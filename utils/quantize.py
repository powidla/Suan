import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class Quantizer:
    def __init__(self, model_name, output_dir, repo_id):

        # --- General settings ---
        self.model_name = model_name
        self.repo_id = repo_id
        self.model_id = f"{model_name.split('/')[-1]}-4bit"
        self.repo_dir = f"{self.repo_id}/{self.model_id}" if self.repo_id else None
        self.output_dir = os.path.join(output_dir, self.model_id)

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage="uint8",
            load_in_8bit=False,
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_has_fp16_weight=False,
            llm_int8_skip_modules=[
                "lm_head",
                "multi_modal_projector",
                "merger",
                "modality_projection",
            ],
            llm_int8_threshold=6.0,
            quant_method="bitsandbytes",
        )


    def quantize(self):
            print(f"Loading model {self.model_name}")
            model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=self.bnb_config)
            model.save_pretrained(self.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer.save_pretrained(self.output_dir)

            # Push to Hub
            if self.repo_dir:
                model.push_to_hub(self.repo_dir, private=True)
                tokenizer.push_to_hub(self.repo_dir, private=True)


def main():
    parser = argparse.ArgumentParser(description="Quantize a model using BitsAndBytes")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--repo_id", type=str, default=None, help="HF repository to push to")
    args = parser.parse_args()

    quantizer = Quantizer(
        model_name=args.model,
        output_dir=args.output_dir,
        repo_id=args.repo_id
    )

    quantizer.quantize()


if __name__ == "__main__":
    main()
