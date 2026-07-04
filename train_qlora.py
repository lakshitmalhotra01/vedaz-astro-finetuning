#!/usr/bin/env python3
"""
train_qlora.py
==============
QLoRA fine-tuning script for Qwen2.5-7B-Instruct on astrologer chat data.

Model: Qwen/Qwen2.5-7B-Instruct
Method: QLoRA (4-bit NF4 quantization) via bitsandbytes + PEFT + TRL SFTTrainer

Design Choices:
  - Qwen2.5-7B-Instruct was chosen because:
      • 7B parameter scale fits in ~12-14GB VRAM with 4-bit quantization
      • Strong instruction-following baseline (RLHF-trained)
      • Superior multilingual support (Hindi/Sanskrit terms common in astrology)
      • Native ChatML support with <|im_start|> / <|im_end|> tokens
  - QLoRA vs full fine-tuning:
      • 80-90% VRAM savings; 7B model in 4-bit ≈ 4.5GB, leaving headroom for gradients
      • LoRA adapters (r=16, alpha=32) strike a good balance between
        expressiveness and regularization for domain adaptation
  - SFTTrainer from TRL handles:
      • Packing short examples for efficiency
      • Dataset splitting, logging, checkpointing
      • Gradient accumulation for effective large-batch training

Hardware requirements:
  - Minimum: 1x GPU with 16GB VRAM (e.g., RTX 3080, T4, A10)
  - Recommended: 1x A100 40GB / RTX 4090 24GB (faster, more room)
  - CPU-only: NOT recommended (extremely slow)

Usage:
  python train_qlora.py \
    --dataset_path ./dataset_chatml.jsonl \
    --output_dir ./output/qwen25_astro_qlora \
    --num_epochs 3

  # With Weights & Biases logging:
  python train_qlora.py --dataset_path ./dataset_chatml.jsonl --use_wandb

  # To merge adapter after training:
  python train_qlora.py --merge_only --adapter_path ./output/qwen25_astro_qlora
"""

import os
import json
import logging
import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_SEQ_LENGTH = 2048      # Max tokens per example (Qwen2.5 supports 32k, but 2k covers most conversations)
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen2.5-7B-Instruct on astrologer chat data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    parser.add_argument(
        "--dataset_path", type=str, default="./dataset_chatml.jsonl",
        help="Path to JSONL dataset in ChatML format (default: ./dataset_chatml.jsonl)"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.05,
        help="Fraction of data to use for validation (default: 0.05 = 5%%)"
    )

    # Model
    parser.add_argument(
        "--model_id", type=str, default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Max sequence length (default: {DEFAULT_MAX_SEQ_LENGTH})"
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16,
        help="LoRA rank (default: 16). Higher = more parameters, more capacity")
    parser.add_argument("--lora_alpha", type=int, default=32,
        help="LoRA alpha scaling (default: 32). Rule of thumb: 2x lora_r")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
        help="LoRA dropout (default: 0.05)")

    # Training
    parser.add_argument("--output_dir", type=str, default="./output/qwen25_astro_qlora",
        help="Directory to save checkpoints and final model")
    parser.add_argument("--num_epochs", type=int, default=3,
        help="Number of training epochs (default: 3)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
        help="Batch size per device (default: 2). Reduce if OOM")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
        help="Gradient accumulation steps (default: 8). Effective batch = 2*8=16")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
        help="Learning rate (default: 2e-4). Typical for QLoRA")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
        help="LR warmup ratio (default: 0.05 = 5%% of steps)")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
        choices=["cosine", "linear", "constant", "cosine_with_restarts"],
        help="LR scheduler type (default: cosine)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
        help="Weight decay (default: 0.01)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
        help="Gradient clipping (default: 1.0)")
    parser.add_argument("--logging_steps", type=int, default=10,
        help="Log every N steps (default: 10)")
    parser.add_argument("--save_steps", type=int, default=100,
        help="Save checkpoint every N steps (default: 100)")
    parser.add_argument("--eval_steps", type=int, default=100,
        help="Evaluate every N steps (default: 100)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # Misc
    parser.add_argument("--use_wandb", action="store_true",
        help="Enable Weights & Biases logging")
    parser.add_argument("--merge_only", action="store_true",
        help="Skip training, only merge adapter at --adapter_path")
    parser.add_argument("--adapter_path", type=str, default=None,
        help="Path to existing adapter (for --merge_only)")
    parser.add_argument("--merged_output_dir", type=str,
        default="./output/qwen25_astro_merged",
        help="Where to save merged model (default: ./output/qwen25_astro_merged)")
    parser.add_argument("--push_to_hub", action="store_true",
        help="Push merged model to HuggingFace Hub")
    parser.add_argument("--hub_repo_id", type=str, default=None,
        help="HuggingFace Hub repo ID (e.g., your-username/qwen25-astro)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chatml_dataset(dataset_path: str, val_split: float = 0.05, seed: int = 42):
    """Load JSONL dataset and format as HuggingFace Dataset."""
    logger.info(f"Loading dataset from: {dataset_path}")

    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} conversations")

    # Shuffle and split
    import random
    random.seed(seed)
    random.shuffle(data)

    n_val = max(1, int(len(data) * val_split))
    val_data = data[:n_val]
    train_data = data[n_val:]

    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Model & tokenizer setup
# ---------------------------------------------------------------------------

def load_quantized_model(model_id: str):
    """Load model in 4-bit NF4 quantization for QLoRA."""
    logger.info(f"Loading {model_id} in 4-bit NF4 (QLoRA)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NF4 is better than FP4 for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for compute (less overflow than fp16)
        bnb_4bit_use_double_quant=True,       # Double quantization further reduces memory
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",                   # Automatically distribute across GPUs
        trust_remote_code=True,              # Qwen2.5 requires this
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if _check_flash_attn() else "eager",
    )

    # Required before adding LoRA adapters to quantized model
    model = prepare_model_for_kbit_training(model)

    logger.info(f"Model loaded. Trainable params before LoRA: {count_parameters(model)}")
    return model


def _check_flash_attn() -> bool:
    """Check if Flash Attention 2 is available."""
    try:
        import flash_attn
        logger.info("Flash Attention 2 detected - using for faster training")
        return True
    except ImportError:
        logger.info("Flash Attention 2 not found - using standard attention (install flash-attn for speed)")
        return False


def count_parameters(model) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M ({100*trainable/total:.2f}%)"


def setup_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float):
    """Configure and apply LoRA adapters."""
    
    # Target all linear layers (Q, K, V, O projections + MLP)
    # For Qwen2.5 architecture
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",                    # Don't train biases (saves memory)
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    logger.info(f"LoRA applied. {count_parameters(model)}")
    model.print_trainable_parameters()
    return model


def load_tokenizer(model_id: str, max_seq_length: int):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",           # Right padding for causal LM training
        model_max_length=max_seq_length,
    )
    # Qwen2.5 uses <|endoftext|> as pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    logger.info(f"Chat template: {tokenizer.chat_template[:100] if tokenizer.chat_template else 'None'}...")
    return tokenizer


# ---------------------------------------------------------------------------
# Formatting function (ChatML → string for training)
# ---------------------------------------------------------------------------

def make_formatting_func(tokenizer):
    """
    Returns a function that converts a messages list to a formatted string.
    Qwen2.5 uses ChatML format:
      <|im_start|>system
      ...<|im_end|>
      <|im_start|>user
      ...<|im_end|>
      <|im_start|>assistant
      ...<|im_end|>
    """
    def formatting_prompts_func(examples):
        output_texts = []
        for messages in examples["messages"]:
            # Apply the model's built-in chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # False: include EOS at end (for training)
            )
            output_texts.append(text)
        return output_texts
    return formatting_prompts_func


# ---------------------------------------------------------------------------
# Merge adapter into base model
# ---------------------------------------------------------------------------

def merge_and_save(
    model_id: str,
    adapter_path: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
):
    """Merge LoRA adapter weights into the base model and save."""
    logger.info("="*60)
    logger.info("MERGING LORA ADAPTER INTO BASE MODEL")
    logger.info("="*60)

    logger.info("Loading base model in float16 for merging...")
    # Load in fp16 (not quantized) for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging weights...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    logger.info("✅ Merge complete!")
    logger.info(f"   Model saved to: {output_dir}")

    if push_to_hub and hub_repo_id:
        logger.info(f"Pushing to HuggingFace Hub: {hub_repo_id}")
        model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)
        logger.info(f"✅ Pushed to: https://huggingface.co/{hub_repo_id}")

    return output_dir


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)

    # Setup logging
    if args.use_wandb:
        import wandb
        wandb.init(
            project="vedaz-astro-qlora",
            name=f"qwen25-7b-astro-r{args.lora_r}",
            config=vars(args),
        )
        report_to = "wandb"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"

    # Load data
    train_dataset, val_dataset = load_chatml_dataset(
        args.dataset_path, args.val_split, args.seed
    )

    # Load model and tokenizer
    model = load_quantized_model(args.model_id)
    tokenizer = load_tokenizer(args.model_id, args.max_seq_length)

    # Apply LoRA
    model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    # Training config
    # Compute total steps for logging
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    total_steps = steps_per_epoch * args.num_epochs
    
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"  Model:              {args.model_id}")
    logger.info(f"  Dataset:            {len(train_dataset)} train / {len(val_dataset)} val")
    logger.info(f"  Epochs:             {args.num_epochs}")
    logger.info(f"  Effective batch:    {effective_batch_size}")
    logger.info(f"  Steps per epoch:    {steps_per_epoch}")
    logger.info(f"  Total steps:        {total_steps}")
    logger.info(f"  Learning rate:      {args.learning_rate}")
    logger.info(f"  LoRA r/alpha:       {args.lora_r}/{args.lora_alpha}")
    logger.info(f"  Max seq length:     {args.max_seq_length}")
    logger.info(f"{'='*60}\n")

    sft_config = SFTConfig(
        output_dir=args.output_dir,

        # Training duration
        num_train_epochs=args.num_epochs,
        max_steps=-1,                          # -1 = use num_train_epochs

        # Batch and gradient
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,           # Saves VRAM at cost of ~20% speed
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Optimizer
        optim="paged_adamw_8bit",              # 8-bit paged AdamW from bitsandbytes (saves VRAM)
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,

        # LR schedule
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,

        # Precision
        bf16=torch.cuda.is_bf16_supported(),   # Use bf16 on Ampere+ GPUs
        fp16=not torch.cuda.is_bf16_supported(),

        # Sequence length
        max_seq_length=args.max_seq_length,
        dataset_text_field=None,               # We use formatting_func

        # Packing: combine short conversations to fill context window efficiently
        packing=True,

        # Logging & checkpointing
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,                    # Keep only last 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Report
        report_to=report_to,
        run_name="qwen25-astro-qlora",

        # Misc
        seed=args.seed,
        dataloader_num_workers=0,              # Set >0 if you have fast NVMe storage
        remove_unused_columns=False,
    )

    formatting_func = make_formatting_func(tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save the adapter
    logger.info(f"\nSaving LoRA adapter to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("\n✅ Training complete!")
    logger.info(f"   Adapter saved to: {args.output_dir}")
    logger.info(f"   To merge adapter with base model, run:")
    logger.info(f"   python train_qlora.py --merge_only --adapter_path {args.output_dir}")

    # Auto-merge after training
    merged_dir = args.merged_output_dir
    logger.info(f"\nAuto-merging adapter → {merged_dir}")
    merge_and_save(
        model_id=args.model_id,
        adapter_path=args.output_dir,
        output_dir=merged_dir,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
    )

    return merged_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.merge_only:
        if not args.adapter_path:
            raise ValueError("--adapter_path is required when using --merge_only")
        merge_and_save(
            model_id=args.model_id,
            adapter_path=args.adapter_path,
            output_dir=args.merged_output_dir,
            push_to_hub=args.push_to_hub,
            hub_repo_id=args.hub_repo_id,
        )
    else:
        merged_dir = train(args)
        logger.info(f"\n🎉 All done! Merged model at: {merged_dir}")
        logger.info(f"   Next step: python evaluate_model.py --model_path {merged_dir}")


if __name__ == "__main__":
    main()
