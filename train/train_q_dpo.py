import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, prepare_model_for_kbit_training
from huggingface_hub import HfApi

from data import formatting
from train.dpo_trainer import DPOTrainer
from train.dpo_config import DPOConfig
from train import loss_dpo


class Trainer:
    def __init__(self, dataset_name, loss_name, model_name, peft_weights, output_dir, repo_id, max_length, grad_check):
        # --- General settings ---
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.loss_name = loss_name
        self.model_name = model_name
        self.peft_weights = peft_weights
        self.repo_id = repo_id
        self.grad_check = grad_check

        self.model_id = f"{self.peft_weights.split('/')[-1]}-{self.dataset_name}-{self.loss_name}"
        self.repo_dir = f"{self.repo_id}/{self.model_id}" if self.repo_id is not None else None
        self.output_dir = os.path.join(output_dir, self.model_id)

        # --- 1. Load model and tokenizer ---
        print(f"Loading model {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # --- 2. Prepare model for LoRA fine-tuning ---
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=self.grad_check)

        print(f"Applying PEFT adapter from: {self.peft_weights}")
        self.model = PeftModel.from_pretrained(self.model, self.peft_weights, is_trainable=True)
        self.model.load_adapter(self.peft_weights, adapter_name='reference', is_trainable=False)
        self.model.set_adapter('default')
        print("Number of trainable params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        # --- 3. Load and format dataset ---
        if self.dataset_name == "hh":
            self.dataset = formatting.FormatHHRLHF(split='train').dpo_format
        elif self.dataset_name == "pku":
            self.dataset = formatting.FormatPKUDPO(split='train').dpo_format
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")
        
        # --- 4. Choose custom loss ---
        if self.loss_name == 'dpo':
            loss_func = loss_dpo.DPOLoss().get_loss_fn()
            use_reference = True
        elif self.loss_name == 'ipo':
            loss_func = loss_dpo.IPOLoss().get_loss_fn()
            use_reference = True
        elif self.loss_name == 'siu':
            loss_func = loss_dpo.SiULoss().get_loss_fn()
            use_reference = True
        else:
            raise ValueError(f"Invalid loss function name: {self.loss_name}")

        # --- 5. Training configuration ---
        self.training_args = DPOConfig(
            max_length=self.max_length,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=self.grad_check,
            use_reference=use_reference,
            warmup_steps=50,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=10,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=self.output_dir,
            bf16=torch.cuda.is_bf16_supported(),
            remove_unused_columns=False,
            logging_nan_inf_filter=False
        )


        # --- 6. Initialize TRL DPOTrainer ---
        self.trainer = DPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.dataset,
            args=self.training_args,
            loss_func=loss_func
        )


    def train(self):
        print("Starting training...")
        self.trainer.train()
        print("Training completed.")

        # Save locally with loss name
        save_dir = f"{self.output_dir}/dpo_saved_lora"
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Push to Hugging Face Hub
        if self.repo_dir is not None:
            print(f"Pushing model to {self.repo_dir}")
            self.model.push_to_hub(self.repo_dir, private=True)
            self.tokenizer.push_to_hub(self.repo_dir, private=True)
            # add metrics.log to track rewards
            metrics_path = os.path.join(self.output_dir, "metrics.log")
            if os.path.exists(metrics_path):
                api = HfApi()
                api.upload_file(path_or_fileobj=metrics_path, path_in_repo="metrics.log", repo_id=self.repo_dir, repo_type="model")
            

def main():
    parser = argparse.ArgumentParser(description="Train a model using TRL DPOTrainer.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., ufb)")
    parser.add_argument("--loss", type=str, default="dpo", help="Which loss function to use (default: dpo).")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--peft_weights", type=str, default=None, help="Path to PEFT weights")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--repo_id", type=str, default=None, help="HF repository to push to")
    parser.add_argument("--max_length", type=int, default=2048, help="max length")
    parser.add_argument("--grad_check", choices=["True", "False"], default="False", help="Enable gradient checkpointing")
    args = parser.parse_args()

    trainer = Trainer(
        dataset_name=args.dataset,
        loss_name=args.loss,
        model_name=args.model,
        peft_weights=args.peft_weights,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        max_length=args.max_length,
        grad_check=args.grad_check=="True",
    )

    trainer.train()


if __name__ == "__main__":
    main()
