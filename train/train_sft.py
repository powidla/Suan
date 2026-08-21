import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from data import formatting


class Trainer:
    def __init__(self, dataset_name, model_name, output_dir, repo_id, loss_name, max_length):
        # --- General settings ---
        self.lora_rank = 16
        self.max_seq_length = max_length
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.repo_id = repo_id
        self.loss_name = loss_name

        self.model_id = f"{model_name.split('/')[-1]}-{self.dataset_name}-sft-{self.loss_name}"
        self.repo_dir = f"{self.repo_id}/{self.model_id}" if self.repo_id is not None else None
        self.output_dir = os.path.join(output_dir, self.model_id)

        print(f"Loading model {self.model_name}")

        # --- 1. Load model and tokenizer ---
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # --- 2. Prepare model for LoRA fine-tuning ---
        self.model = prepare_model_for_kbit_training(self.model)

        if 'pythia' in self.model_id.lower():
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif 'opt' in self.model_id.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)

        print("Number of trainable params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        # --- 3. Load and format dataset ---
        if self.dataset_name == "alpaca":
            self.dataset = formatting.FormatAlpaca(split='train', tokenizer=self.tokenizer).sft_format
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")

        # --- 4. Training configuration ---
        self.training_args = SFTConfig(
            max_length=self.max_seq_length,
            completion_only_loss=True,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=10,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=self.output_dir,
            report_to="none",
        )

        # --- 5. Choose custom loss ---
        if self.loss_name == 'ce':
            compute_loss_func = None
        else:
            raise ValueError(f"Invalid loss function name: {self.loss_name}")

        # --- 6. Initialize TRL SFTTrainer ---
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.dataset,
            args=self.training_args,
            compute_loss_func=compute_loss_func,
        )

    def train(self):
        print("Starting training...")
        self.trainer.train()
        print("Training completed.")

        # Save locally with loss name
        save_dir = f"{self.output_dir}/sft_saved_lora"
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Push to Hugging Face Hub
        if self.repo_dir is not None:
            print(f"Pushing model to {self.repo_dir}")
            self.model.push_to_hub(self.repo_dir, private=True)
            self.tokenizer.push_to_hub(self.repo_dir, private=True)


def main():
    parser = argparse.ArgumentParser(description="Train a model using TRL SFTTrainer with custom loss functions.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., alpaca, gsm8k, mmlu)")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--repo_id", type=str, default=None, help="HF repository to push to")
    parser.add_argument("--loss", type=str, default="ce", help="Which loss function to use (default: ce).")
    parser.add_argument("--max_length", type=int, default=2048, help="max length")
    args = parser.parse_args()

    trainer = Trainer(
        dataset_name=args.dataset,
        model_name=args.model,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        loss_name=args.loss,
        max_length=args.max_length
    )

    trainer.train()


if __name__ == "__main__":
    main()
