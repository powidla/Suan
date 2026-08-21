import argparse
import os
import json
import re
from huggingface_hub import hf_hub_download, upload_file, snapshot_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm

from data import formatting, utility_names


class Evaluator:
    def __init__(self, judge_model, input_file, output_dir, repo_id, max_new_tokens, peft_weights=None, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_lora_rank=16, quantization=None):

        self.judge_model = judge_model
        self.base_model = judge_model  
        self.input_file = input_file
        self.repo_id = repo_id
        self.max_new_tokens = max_new_tokens
        self.peft_weights = peft_weights

        self.model_id = f"{self.input_file.split('/')[-2]}"
        self.output_dir = os.path.join(output_dir, self.model_id)
        self.subset = self.input_file.split('/')[-2].split('-')[-2]
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Loading judge model: {self.judge_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.judge_model)

        # --- vLLM engine setup ---
        self.lora_request = None
        llm_kwargs = dict(
            model=self.base_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=8192, # for ArenaHard
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

        # Greedy decoding
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
        )

        print(f"Loading data from: {self.input_file}")
        data = self.load_file(self.input_file)
        dataset_format = self.input_file.split('/')[-2].split('-')[-1]

        if self.subset in ('mi', 'ab', 'hb', 'sb'):
            self.data = formatting.FormatEval(data).judge_safe_format
        elif self.subset in ('xs', 'or'):
            self.data = formatting.FormatEval(data).judge_refuse_flow_format
        elif self.subset in ('ae', 'ah', 'mt'):
            self.data = formatting.FormatEval(data).judge_instruct_format
            # add special instruction for FlowJudge v.0.1
            if self.judge_model.split('/')[-1] == utility_names.JUDGES[-1].split('/')[-1]:
                self.data = formatting.FormatEval(data).judge_flow_format
            # default prompt otherwise
            else:
                self.data = formatting.FormatEval(data).judge_arena_format
        else:
            raise ValueError(f"Invalid dataset format: {dataset_format}")
        print(f"Loaded {len(self.data)} instruction-response pairs to score")

    def load_file(self, path):
        if not os.path.isdir(path):
            print(f"Downloading from {path} ...")
            parts = path.strip("/").split("/")
            repo_id, subpath = "/".join(parts[:2]), "/".join(parts[2:])
            path = hf_hub_download(repo_id=repo_id, filename=subpath, repo_type="dataset")
            print(f"Downloaded to: {path}")

        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def extract_rating(self, text):
        # special format for flow judge
        if self.judge_model.split('/')[-1] == utility_names.JUDGES[-1].split('/')[-1]:
            match = re.search(r"<score>\s*\n?\s*([0-9]+(?:\.\d+)?)\s*</score>", text)
        else:
            match = re.search(r"Total rating:\s*([0-9]+(?:\.\d+)?)", text)
        if not match:
            print(f"Warning: Failed to extract rating, generated text snippet: {text}")
            return None
        rating = float(match.group(1))
        return rating if 0 <= rating <= 5 else None

    def score_batch(self, judge_prompts):
        chats = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in judge_prompts
        ]
        outputs = self.llm.generate(chats, self.sampling_params, lora_request=self.lora_request)
        judge_responses = [o.outputs[0].text for o in outputs]
        ratings = [self.extract_rating(r) for r in judge_responses]

        return ratings, judge_responses

    def run(self, chunk_size=100):
        output_file = os.path.join(self.output_dir, "responses_quality_scores.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            pass

        for chunk_start in tqdm(range(0, len(self.data), chunk_size), desc="Scoring responses (chunks)"):
            chunk = self.data[chunk_start:chunk_start + chunk_size]

            flat_prompts = []
            index_map = []  
            for item_idx, item in enumerate(chunk):
                flat_prompts.extend(item['judge_prompt'])
                index_map.extend([item_idx] * len(item['judge_prompt']))

            ratings_flat, judge_responses_flat = self.score_batch(flat_prompts)

            per_item_ratings = {i: [] for i in range(len(chunk))}
            per_item_responses = {i: [] for i in range(len(chunk))}
            for item_idx, rating, response in zip(index_map, ratings_flat, judge_responses_flat):
                per_item_ratings[item_idx].append(rating)
                per_item_responses[item_idx].append(response)

            with open(output_file, 'a', encoding='utf-8') as f:
                for item_idx, item in enumerate(chunk):
                    score_results = {
                        'rating': per_item_ratings[item_idx],
                        'judge_response': per_item_responses[item_idx],
                    }
                    valid_ratings = [r for r in score_results['rating'] if r is not None]
                    score_results['average_rating'] = sum(valid_ratings) / len(valid_ratings) if valid_ratings else None
                    result = item | score_results
                    subset_keys = ['response', 'rating', 'judge_response']
                    result['scored_responses'] = [dict(zip(subset_keys, v)) for v in zip(*(result[k] for k in subset_keys))]
                    result.pop('judge_prompt')
                    result.pop('response')
                    result.pop('rating')
                    result.pop('judge_response')

                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\nScoring complete! Results saved to: {output_file}")

        if self.repo_id is not None:
            print(f"Pushing resulting file to {self.repo_id}")
            upload_file(path_or_fileobj=output_file, path_in_repo=f"{self.model_id}/responses_quality_scores.jsonl",
                        repo_id=self.repo_id, repo_type="dataset")


def main():
    parser = argparse.ArgumentParser(description="Score responses using a judge model with formatting.py")
    parser.add_argument('--input_file', type=str, required=True, help='Path to local JSONL file with prompts and responses')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save scored responses')
    parser.add_argument('--judge_model', type=str, required=True, help='Judge model to use for scoring')
    parser.add_argument('--peft_weights', type=str, default=None, help='Optional LoRA adapter for the judge model')
    parser.add_argument("--repo_id", type=str, default=None, help="HF repository to push to")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="max new tokens")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_lora_rank", type=int, default=16)
    parser.add_argument("--quantization", type=str, default=None, help="e.g. bitsandbytes")
    parser.add_argument("--chunk_size", type=int, default=100, help="chuncks")
    args = parser.parse_args()

    evaluator = Evaluator(
        judge_model=args.judge_model,
        input_file=args.input_file,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        max_new_tokens=args.max_new_tokens,
        peft_weights=args.peft_weights,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_lora_rank=args.max_lora_rank,
        quantization=args.quantization,
    )

    evaluator.run(chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
