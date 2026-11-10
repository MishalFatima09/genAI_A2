#!/usr/bin/env python
"""Task 3 Benchmark Runner.

Run on Kaggle/Colab GPU:
    python code/benchmark_kv.py --model adapters/qwen25-1.5b-lora --base Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from kv_cache import run_benchmark, verify_equivalence  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to fine-tuned model (LoRA adapter or merged)")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model name")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--prompt-lengths", nargs="+", type=int, default=[128, 512, 1024])
    ap.add_argument("--gen-lengths", nargs="+", type=int, default=[16, 64, 128])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "kv_benchmark.json"))
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} model={args.model}")

    results = run_benchmark(
        model_path=args.model,
        base_model=args.base,
        device=device,
        prompt_lengths=args.prompt_lengths,
        gen_lengths=args.gen_lengths,
        n_repeats=args.repeats,
        warmup=args.warmup,
    )

    # Also run correctness verification on a few more prompts
    print("\n[verify] Running extended correctness check...")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.pad_token = tokenizer.eos_token
    import transformers
    from peft import PeftModel
    kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, **{kw: compute_dtype}, device_map={"": device} if device == "cuda" else None)
    if Path(args.model).is_dir():
        model = PeftModel.from_pretrained(model, args.model, torch_dtype=compute_dtype, device_map={"": device} if device == "cuda" else None)
    model.eval()

    test_prompts = [
        "Patient record:\n0. 65-year-old female with diabetes.\n1. HbA1c 7.2%.\n\nCriterion type: exclusion\nCriterion: HbA1c > 8%\n\nReturn the JSON decision.",
        "Patient record:\n0. 30-year-old male.\n1. No medical history.\n2. BMI 24.\n\nCriterion type: inclusion\nCriterion: Age 18-65\n\nReturn the JSON decision.",
    ]
    eq = verify_equivalence(model, tokenizer, test_prompts, 64, device)
    results["extended_correctness"] = {
        "matches": eq["matches"],
        "mismatches": eq["mismatches"],
        "details": eq["details"],
    }
    print(f"  Matches: {eq['matches']}, Mismatches: {eq['mismatches']}")

    # Save
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {args.out}")

    # Summary
    print("\n=== SUMMARY ===")
    for c in results["configs"]:
        speedup = c["cached"]["tokens_s_mean"] / c["uncached"]["tokens_s_mean"] if c["uncached"]["tokens_s_mean"] else 0
        print(f"  plen={c['prompt_len']:4d} glen={c['gen_len']:3d} | "
              f"cached={c['cached']['tokens_s_mean']:6.1f} tok/s | "
              f"uncached={c['uncached']['tokens_s_mean']:6.1f} tok/s | "
              f"speedup={speedup:.2f}x | "
              f"match={c['correctness']['match']}")


if __name__ == "__main__":
    main()