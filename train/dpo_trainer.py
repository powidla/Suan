from collections import defaultdict
from dataclasses import dataclass

import torch
import json
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from packaging.version import Version
from transformers import Trainer
from transformers.data.data_collator import DataCollatorMixin
from utils import logs


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.
    """
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)

    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values.unsqueeze(-1)  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)

    if squeeze:
        per_token_logps = per_token_logps.squeeze(-1)

    return per_token_logps


def pad(tensors, padding_value=0, padding_side="right"):
    """
    Pads a list of tensors to the same shape along the first dimension.
    """
    max_len = max(t.shape[0] for t in tensors)
    output = torch.full((len(tensors), max_len, *tensors[0].shape[1:]), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)
    for i, t in enumerate(tensors):
        if padding_side == "left":
            output[i, max_len - t.shape[0]:] = t
        elif padding_side == "right":
            output[i, :t.shape[0]] = t
        else:
            raise ValueError("padding_side must be 'left' or 'right'")
    return output


@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """
    Data collator used for preference data. Inputs are dynamically padded to the maximum length of a batch if they are
    not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.
    """

    pad_token_id: int
    max_length: int | None
    return_tensors: str = "pt"

    def torch_call(self, examples):
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]

        # Concatenate the chosen and rejected inputs to avoid doing two forward passes.
        prompt_input_ids = prompt_input_ids + prompt_input_ids
        prompt_loss_mask = [torch.zeros_like(input_ids) for input_ids in prompt_input_ids]
        completion_input_ids = chosen_input_ids + rejected_input_ids
        completion_loss_mask = [torch.ones_like(input_ids) for input_ids in completion_input_ids]

        # Concatenate the prompt and completion inputs
        input_ids = [torch.cat(pair) for pair in zip(prompt_input_ids, completion_input_ids)]
        loss_mask = [torch.cat(pair) for pair in zip(prompt_loss_mask, completion_loss_mask)]

        # Pad
        input_ids = pad(input_ids, padding_value=self.pad_token_id)
        loss_mask = pad(loss_mask, padding_value=0)

        # Truncate
        if self.max_length is not None and self.max_length < loss_mask.size(1):
            input_ids = input_ids[:, :self.max_length]
            loss_mask = loss_mask[:, :self.max_length]
        # For Safe-DPO the dataset looks like D = {x, y_c, y_r, h_c, h_l}, where h_c, h_l are bools. That is why we need to handle them in loss.
        result = {
            "input_ids": input_ids,
            "loss_mask": loss_mask
        }
        if "chosen_is_safe" in examples[0]:
            result["chosen_is_safe"] = torch.tensor([x["chosen_is_safe"] for x in examples], dtype=torch.float32)
            result["rejected_is_safe"] = torch.tensor([x["rejected_is_safe"] for x in examples], dtype=torch.float32)
    
        return result


class DPOTrainer(Trainer):

    def __init__(self, model, args, train_dataset, processing_class, loss_func):
        self.is_peft_model = True
        self.use_amp = args.bf16
        self.max_length = args.max_length
        self.use_reference = args.use_reference
        self.dataset_num_proc = args.dataset_num_proc
        self.loss_func = loss_func

        # Pad token selection
        pad_token = args.pad_token or processing_class.pad_token or processing_class.eos_token
        self.pad_token_id = processing_class.convert_tokens_to_ids(pad_token)
        if self.pad_token_id is None:
            raise ValueError(f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given `processing_class` ({processing_class.__class__.__name__}).")

        # Cast to BF16
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)
        
        # Turn off dropout
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Dataset preparation
        data_collator = DataCollatorForPreference(pad_token_id=self.pad_token_id, max_length=self.max_length)
        train_dataset = self._prepare_dataset(train_dataset, processing_class)
        
        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=processing_class
        )

        self.model_accepts_loss_kwargs = False
        self._metrics_logger = logs.log_metrics(args.output_dir)


    def _prepare_dataset(self, dataset, tokenizer):
        # Build the kwargs for the `map` function
        remove_cols = ["prompt", "chosen", "rejected"]
        existing_cols = dataset.column_names
        remove_cols = [c for c in remove_cols if c in existing_cols]
        map_kwargs = {
            "num_proc": self.dataset_num_proc,
            "writer_batch_size": 10,
            "desc": "Tokenizing train dataset",
            "remove_columns": remove_cols
        }

        with PartialState().main_process_first():
            # Tokenize the dataset
            dataset = dataset.map(
                self.tokenize_row,
                fn_kwargs={"tokenizer": tokenizer},
                **map_kwargs,
            )

        return dataset

    @staticmethod
    def tokenize_row(features, tokenizer):
        """
        Tokenize a row of the dataset.
        """
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        if tokenizer.bos_token_id is not None:
            prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
        
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        result = {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }
        if "chosen_is_safe" in features:
            result["chosen_is_safe"] = features["chosen_is_safe"]
            result["rejected_is_safe"] = features["rejected_is_safe"]
            
        return result

    def concatenated_forward(self, model, batch):
        # Forward
        output = model(batch["input_ids"], use_cache=False, output_hidden_states=True)

        per_token_logps = selective_log_softmax(output['logits'][:, :-1], batch["input_ids"][:, 1:] * batch["loss_mask"][:, 1:])
        all_logps = (per_token_logps*batch["loss_mask"][:, 1:]).sum(-1)

        assert batch["input_ids"].shape[0] % 2 == 0, f"Batch size must be even, got {batch['input_ids'].shape[0]}."
        num_examples = batch["input_ids"].shape[0] // 2

        return {
            "chosen_logps": all_logps[:num_examples],
            "rejected_logps": all_logps[num_examples:],
        }
    
    
    def compute_ref_log_probs(self, model, batch):

        with torch.no_grad():
            model.set_adapter('reference')
            ref_model_output = self.concatenated_forward(model, batch)
            model.set_adapter('default')

        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"]


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            
            outputs = self.concatenated_forward(model, inputs)
            has_sf = "chosen_is_safe" in inputs 

            if self.use_reference:
                ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(model, inputs)
                if has_sf:
                    loss, chosen_rewards, rejected_rewards = self.loss_func(outputs["chosen_logps"], outputs["rejected_logps"], 
                                                                            ref_chosen_logps, ref_rejected_logps, inputs["chosen_is_safe"], inputs["rejected_is_safe"])
                else:
                    loss, chosen_rewards, rejected_rewards = self.loss_func(outputs["chosen_logps"], outputs["rejected_logps"], ref_chosen_logps, ref_rejected_logps)
            else:
                if has_sf: 
                    loss, chosen_rewards, rejected_rewards = self.loss_func(outputs["chosen_logps"], outputs["rejected_logps"], inputs["chosen_is_safe"], inputs["rejected_is_safe"])
                else:
                    loss, chosen_rewards, rejected_rewards = self.loss_func(outputs["chosen_logps"], outputs["rejected_logps"])


        self._stored_metrics["train"]["rewards/chosen"].append(self.accelerator.gather_for_metrics(chosen_rewards).mean().item())
        self._stored_metrics["train"]["rewards/rejected"].append(self.accelerator.gather_for_metrics(rejected_rewards).mean().item())
        self._stored_metrics["train"]["rewards/accuracies"].append(self.accelerator.gather_for_metrics((chosen_rewards > rejected_rewards).float()).mean().item())
        self._stored_metrics["train"]["rewards/margins"].append(self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item())
        self._stored_metrics["train"]["logps/chosen"].append(self.accelerator.gather_for_metrics(outputs["chosen_logps"]).detach().mean().item())
        self._stored_metrics["train"]["logps/rejected"].append(self.accelerator.gather_for_metrics(outputs["rejected_logps"]).detach().mean().item())

        return (loss.mean(), outputs) if return_outputs else loss.mean()


    def log(self, logs, start_time=None):
        """
        Log `logs` on the various objects watching training, including stored metrics.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        if self.accelerator.is_main_process:
            record = {
                "step": self.state.global_step,
                "epoch": round(self.state.epoch, 4) if self.state.epoch is not None else None,
                "split": train_eval,
                **logs}
            self._metrics_logger.info(json.dumps(record))

        return super().log(logs, start_time)
