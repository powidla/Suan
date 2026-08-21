import argparse
import os
import json
from tqdm import tqdm
from huggingface_hub import upload_file
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from data import formatting


class Inference:
    def __init__(self, base_model, dataset_name, dataset_format, peft_weights, output_dir, repo_id, max_new_tokens, temperature, top_p, do_sample, num_return_sequences,
                 use_sys_prompt=False, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_lora_rank=16, quantization=None):
        self.base_model = base_model
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.peft_weights = peft_weights
        self.output_dir = output_dir
        self.repo_id = repo_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.num_return_sequences = num_return_sequences
        self.use_sys_prompt = use_sys_prompt

        self.id = (f"{self.peft_weights.strip('/').split('/')[-1]}-{self.dataset_name}-{self.dataset_format}"
                   if self.peft_weights else f"{self.base_model.split('/')[-1]}-{self.dataset_name}-{self.dataset_format}")
        self.output_dir = os.path.join(output_dir, self.id)
        os.makedirs(self.output_dir, exist_ok=True)

        if self.dataset_name == "mi":
            self.dataset = formatting.FormatMaliciousInstruct(split='train', use_sys_prompt=self.use_sys_prompt)
        elif self.dataset_name == "ab":
            self.dataset = formatting.FormatAdvBench(split='train', use_sys_prompt=self.use_sys_prompt)
        elif self.dataset_name == "hb":
            self.dataset = formatting.FormatHarmBench(split='train', use_sys_prompt=self.use_sys_prompt)
        elif self.dataset_name == "sb":
            self.dataset = formatting.FormatSorryBench(split='train', use_sys_prompt=self.use_sys_prompt)
        elif self.dataset_name == "xs":
            self.dataset = formatting.FormatXSTest(split='train', use_sys_prompt=self.use_sys_prompt)
        elif self.dataset_name == "or":
            self.dataset = formatting.FormatORBench(split='train', use_sys_prompt=self.use_sys_prompt)
        elif self.dataset_name == "nb":
            self.dataset = formatting.FormatNoveltyBench(split='curated', use_sys_prompt=self.use_sys_prompt)
        elif self.dataset_name == "ae": 
            self.dataset= formatting.FormatAlpacaEval(split='eval', use_sys_prompt=self.use_sys_prompt)
        elif self.dataset_name == "mt": 
            self.dataset= formatting.FormatMTBench(split='train', use_sys_prompt=self.use_sys_prompt)
        elif self.dataset_name == "ah": 
            self.dataset= formatting.FormatArenaHard(split='train', use_sys_prompt=self.use_sys_prompt)
        elif self.dataset_name == "arc":
            self.dataset = formatting.FormatARC(split='test', use_sys_prompt=self.use_sys_prompt)
        elif self.dataset_name == "mmlu":
            self.dataset = formatting.FormatMMLU(split='test', use_sys_prompt=self.use_sys_prompt)
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")

        if self.dataset_format == "sft":
            self.dataset = self.dataset.sft_format
        elif self.dataset_format == "base":
            self.dataset = self.dataset.base_format
        else:
            raise ValueError(f"Invalid dataset format: {self.dataset_format}")

        # vLLM engine
        self.lora_request = None
        llm_kwargs = dict(
            model=self.base_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        if quantization:
            llm_kwargs["quantization"] = quantization
            if quantization == "bitsandbytes":
                llm_kwargs["load_format"] = "bitsandbytes"

        if self.peft_weights:
            print(f"Enabling LoRA adapter from: {self.peft_weights}")
            llm_kwargs.update(enable_lora=True, max_lora_rank=max_lora_rank)
            self.llm = LLM(**llm_kwargs)
            adapter_path = self.peft_weights
            if not os.path.isdir(adapter_path):
                adapter_path = snapshot_download(repo_id=self.peft_weights)
            self.lora_request = LoRARequest("adapter", 1, adapter_path)
        else:
            self.llm = LLM(**llm_kwargs)

        self.sampling_params = SamplingParams(
            n=self.num_return_sequences,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p if self.do_sample else 1.0,
            max_tokens=self.max_new_tokens,
        )

    def run(self):
        output_file = os.path.join(self.output_dir, "responses.jsonl")

        items = list(self.dataset)
        prompts = [item["prompt"] for item in items]

        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            lora_request=self.lora_request,
        )

        with open(output_file, "w", encoding="utf-8") as f:
            for item, output in tqdm(zip(items, outputs), total=len(items)):
                responses = [o.text for o in output.outputs]
                entry = {
                    "prompt": item["prompt"],
                    "responses": responses,
                }
                if "correct_response" in item.keys():
                    entry["correct_response"] = item["correct_response"]
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self.repo_id is not None:
            print(f"Pushing resulting file to {self.repo_id}")
            upload_file(path_or_fileobj=output_file, path_in_repo=f"{self.id}/responses.jsonl",
                        repo_id=self.repo_id, repo_type="dataset")


def main():
    parser = argparse.ArgumentParser(description="Fast batched inference with vLLM, optional LoRA")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_format", type=str, required=True)
    parser.add_argument("--peft_weights", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", choices=["True", "False"], default="True")
    parser.add_argument("--num_return_sequences", type=int, default=10)
    parser.add_argument("--use_sys_prompt", choices=["True", "False"], default="False")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_lora_rank", type=int, default=16)
    parser.add_argument("--quantization", type=str, default=None, help="e.g. bitsandbytes")
    args = parser.parse_args()

    inference = Inference(
        base_model=args.base_model,
        dataset_name=args.dataset,
        dataset_format=args.dataset_format,
        peft_weights=args.peft_weights,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample == "True",
        num_return_sequences=args.num_return_sequences,
        use_sys_prompt=args.use_sys_prompt == "True",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_lora_rank=args.max_lora_rank,
        quantization=args.quantization,
    )

    inference.run()


if __name__ == "__main__":
    main()
