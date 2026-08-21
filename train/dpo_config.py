from dataclasses import dataclass, field
from typing import Any

from transformers import TrainingArguments


@dataclass
class DPOConfig(TrainingArguments):
    """
    Configuration class for the [`DPOTrainer`].

    Subclasses [`~transformers.TrainingArguments`] and overrides fields that are common across TRL trainers or that
    contain unescaped "%" characters which would cause argparse to raise a `TypeError` when rendering `--help` output.

    Parameters:
        logging_steps (`int` or `float`, *optional*, defaults to `10`):
            Number of update steps between two logs if `logging_strategy="steps"`. Should be an integer or a float in
            range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
        gradient_checkpointing (`bool`, *optional*, defaults to `True`):
            Whether to enable gradient checkpointing to trade compute for memory. Reduces memory usage by clearing
            activations during forward pass and recomputing them during backward pass. Enables training larger models
            or batch sizes at the cost of ~20% slower training.
        bf16 (`bool`, *optional*):
            Whether to use bfloat16 (BF16) mixed precision instead of 32-bit. Generally preferred over FP16 due to
            better numerical stability and no loss scaling required. Requires Ampere or higher NVIDIA architecture or
            Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set.
        lr_scheduler_kwargs (`dict` or `str`, *optional*):
            Additional parameters for the lr_scheduler, such as `{'num_cycles': 1}` for cosine with hard restarts. See
            the documentation of each scheduler for possible values.
        torch_empty_cache_steps (`int`, *optional*):
            Number of steps to wait before calling `torch.<device>.empty_cache()`. If left unset or set to None, cache
            will not be emptied. This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of
            about [10% slower performance](https://github.com/huggingface/transformers/issues/31372).


        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`DPOTrainer`] is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.

        > Parameters that control the data preprocessing

        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        pad_token (`str`, *optional*):
            Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is also `None`,
            it falls back to `processing_class.eos_token`.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated. If `None`, no truncation is applied.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Override fields from TrainingArguments to set defaults.
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training."
        },
    )
    bf16: bool | None = field(
        default=False,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU."
        },
    )

    # Override fields from TrainingArguments whose help strings contain unescaped "%" characters.
    # argparse interprets "%" as a format specifier, raising TypeError when rendering --help output.
    # Fixed upstream in transformers v5.3.0, but overridden here to support older versions.
    # - Introduced in v5.2.0; fixed in v5.3.0
    use_liger_kernel: bool = field(
        default=False,
        metadata={
            "help": "Enable Liger Kernel optimizations. Increases throughput by ~20%% and reduces memory by ~60%%."
        },
    )
    # - Introduced in v4.54.1; fixed in v5.3.0
    torch_empty_cache_steps: int | None = field(
        default=None,
        metadata={
            "help": "Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied."
        },
    )

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `DPOTrainer` is provided as a string."
        },
    )

    use_reference: bool = field(
        default=True,
        metadata={"help": "Whether the usage of reference model is needed."},
    )

    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )

    # Parameters that control the data preprocessing
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    pad_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that "
            "is also `None`, it falls back to `processing_class.eos_token`."
        },
    )
    max_length: int | None = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated. If `None`, no truncation is applied."
        },
    )