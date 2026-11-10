#!/usr/bin/env python
"""Task 5 Benchmark Runner.

Run on Kaggle/Colab GPU:
    python code/benchmark_speculative.py --target adapter_result_kaggle --base Qwen/Qwen2.5-1.5B-Instruct

Run on CPU (correctness only, slow):
    python code/benchmark_speculative.py --target adapter_result_kaggle --base Qwen/Qwen2.5-1.5B-Instruct --device cpu --repeats 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from speculative import run_benchmark  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct", help="Draft model name")
    ap.add_argument("--target", required=True, help="Path to LoRA adapter (fine-tuned target)")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model name")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--gamma", type=int, default=4, help="Draft tokens per iteration")
    ap.add_argument("--prompt-lengths", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--gen-lengths", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "task5_results.json"))
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    results = run_benchmark(
        draft_name=args.draft,
        target_path=args.target,
        base_model=args.base,
        device=device,
        prompt_lengths=args.prompt_lengths,
        gen_lengths=args.gen_lengths,
        gamma=args.gamma,
        n_repeats=args.repeats,
        warmup=args.warmup,
    )

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {args.out}")

    # Summary
    print("\n=== SUMMARY ===")
    for c in results["configs"]:
        speedup = c["greedy"]["tokens_s_mean"] / c["speculative"]["tokens_s_mean"] if c["speculative"]["tokens_s_mean"] else 0
        print(f"  plen={c['prompt_len']:4d} glen={c['gen_len']:3d} | "
              f"greedy={c['greedy']['tokens_s_mean']:6.1f} tok/s | "
              f"spec={c['speculative']['tokens_s_mean']:6.1f} tok/s | "
              f"speedup={speedup:.2f}x | "
              f"acc={c['speculative']['acceptance_rate_mean']:.2%} | "
              f"match={c['correctness']['match']}")

    if "extended_correctness" in results:
        ec = results["extended_correctness"]
        print(f"\nExtended correctness: {ec['matches']} matches, {ec['mismatches']} mismatches")


if __name__ == "__main__":
    import torch  # noqa: E402
    main()