"""Task 1/B3: LoRA fine-tuning of Qwen2.5-1.5B-Instruct on the TrialGPT split.

Designed for Google Colab free T4 (16 GB, no bf16 -> fp16). Runs locally too
(CPU, slow). Uses only the 603 training rows of data/splits/train.jsonl and
validates on data/splits/val.jsonl. The test split is NEVER loaded here.

Outputs:
  adapters/qwen25-1.5b-lora/   LoRA adapter + tokenizer + trainer state
  results/training_log.json    hyperparameters + per-epoch metrics
  results/train_curve.csv      step-level loss/eval loss for plotting

Run:      python code/finetune_lora.py            (or on Colab)
Colab:    !git clone <repo> && cd <repo> && python code/finetune_lora.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from schema import build_messages, build_target, parse_gold_sentences  # noqa: E402

SEED = 42
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_rows(split: str) -> list[dict]:
    path = ROOT / "data" / "splits" / f"{split}.jsonl"
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def chat_ids(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    """Tokenized chat template -> flat id list.

    transformers v4 returns list[int]; v5 returns a BatchEncoding, so both are
    handled here (finetune + any caller) to keep the pipeline version-agnostic.
    """
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    return out if isinstance(out, list) else out["input_ids"]


def build_example(row: dict) -> tuple[str, str]:
    messages = build_messages(row["note"], row["criterion_type"], row["criterion_text"])
    target = build_target(
        row["expert_eligibility"], parse_gold_sentences(row["expert_sentences"])
    )
    return messages, target


def make_dataset(rows: list[dict], tokenizer, max_length: int = 2048) -> list[dict]:
    """Tokenize prompt + target; mask prompt tokens with -100 (loss on target only)."""
    out = []
    for row in rows:
        messages, target = build_example(row)
        prompt_ids = chat_ids(tokenizer, messages, add_generation_prompt=True)
        full_ids = chat_ids(
            tokenizer,
            messages + [{"role": "assistant", "content": target}],
            add_generation_prompt=False,
        )
        prompt_len = len(prompt_ids)
        if len(full_ids) > max_length:
            # Never happens on this dataset (max prompt+target < 700 tokens);
            # kept as a safety net: truncate the tail, clamp the mask boundary.
            full_ids = full_ids[:max_length]
            prompt_len = min(prompt_len, max_length)
        out.append(
            {
                "input_ids": full_ids,
                "labels": [-100] * prompt_len + full_ids[prompt_len:],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--out", default=str(ROOT / "adapters" / "qwen25-1.5b-lora"))
    ap.add_argument("--limit", type=int, default=0, help="debug: cap train rows")
    ap.add_argument(
        "--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
        help="base-weight dtype. auto: fp16 on CUDA (fp32 master + accelerate "
             "fp16 autocast, Colab T4 plan), bf16 on CPU (7.7 GB RAM machine "
             "cannot hold fp32 master weights).",
    )
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute activations (trades ~30% speed for memory)")
    args = ap.parse_args()

    set_seed(SEED)

    import torch
    from contextlib import nullcontext
    from accelerate import Accelerator
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        get_cosine_schedule_with_warmup,
    )
    import transformers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed = "fp16" if device == "cuda" else "no"  # T4 (sm_75) has no bf16
    if args.dtype == "auto":
        load_dtype = torch.float32 if device == "cuda" else torch.bfloat16
    else:
        load_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                      "fp32": torch.float32}[args.dtype]
    # CPU mixed precision: bf16 weights for the frozen base + fp32 LoRA params
    # (PEFT allocates them fp32) must be reconciled by an autocast region.
    cpu_autocast = device == "cpu" and load_dtype == torch.bfloat16
    fwd_ctx = (lambda: torch.autocast("cpu", dtype=torch.bfloat16)) if cpu_autocast else nullcontext
    print(f"[env] device={device} mixed_precision={mixed} load_dtype={load_dtype} "
          f"cpu_autocast={cpu_autocast} torch={torch.__version__} "
          f"transformers={transformers.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **{kw: load_dtype})
    use_ckpt = device == "cuda" or args.grad_checkpoint
    if use_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # frozen base -> grads stop at embeddings
    model.config.use_cache = False  # required with gradient checkpointing

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_rows = load_rows("train")
    val_rows = load_rows("val")
    if args.limit:
        train_rows = train_rows[: args.limit]
        val_rows = val_rows[: max(1, args.limit // 2)]  # debug only
    train_ds = make_dataset(train_rows, tokenizer, args.max_length)
    val_ds = make_dataset(val_rows, tokenizer, args.max_length)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=generator,
        collate_fn=collator, drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * int(args.epochs) // max(args.grad_accum, 1)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision=mixed)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    history: list[dict] = []
    log_path = ROOT / "results" / "train_curve.csv"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("step,split,loss\n")

    def run_eval() -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                with fwd_ctx():
                    out = model(**batch)
                total += out.loss.item() * batch["input_ids"].size(0)
                n += batch["input_ids"].size(0)
        model.train()
        return total / max(n, 1)

    best_val = float("inf")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    t0 = time.time()

    model.train()
    for epoch in range(int(args.epochs)):
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                with fwd_ctx():
                    out = model(**batch)
                accelerator.backward(out.loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            running += out.loss.item()
            seen += 1
            if (step + 1) % args.grad_accum == 0 or (step + 1) == steps_per_epoch:
                global_step += 1
                avg = running / seen
                print(f"epoch {epoch+1} step {global_step} loss {avg:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{global_step},train,{avg:.6f}\n")
                running, seen = 0.0, 0
        val_loss = run_eval()
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{global_step},val,{val_loss:.6f}\n")
        history.append({"epoch": epoch + 1, "val_loss": round(val_loss, 5)})
        print(f"[eval] epoch {epoch+1} val_loss {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)

    elapsed = time.time() - t0
    log = {
        "base_model": args.base_model,
        "method": "LoRA (peft)",
        "seed": SEED,
        "device": device,
        "mixed_precision": mixed,
        "load_dtype": str(load_dtype).replace("torch.", ""),
        "cpu_autocast_bf16": cpu_autocast,
        "gradient_checkpointing": use_ckpt,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "hyperparameters": {
            k: getattr(args, k)
            for k in ["epochs", "batch_size", "grad_accum", "lr", "r", "alpha",
                      "dropout", "max_length", "warmup_ratio", "weight_decay"]
        },
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "global_steps": global_step,
        "best_val_loss": round(best_val, 5),
        "wall_time_sec": round(elapsed, 1),
        "history": history,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (ROOT / "results" / "training_log.json").write_text(json.dumps(log, indent=2))
    print("[done]", json.dumps({"best_val_loss": best_val, "sec": round(elapsed, 1)}))
    print("[save]", out_dir)


if __name__ == "__main__":
    main()
