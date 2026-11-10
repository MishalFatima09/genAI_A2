#!/usr/bin/env python
"""Task 4 Benchmark Runner.

Run on Kaggle/Colab GPU:
    python code/benchmark_tiled.py --verify --benchmark --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from tiled_attention import run_all_verification, run_benchmarks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="Run numerical verification")
    ap.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", default=str(ROOT / "results" / "task4_results.json"))
    args = ap.parse_args()

    if not args.verify and not args.benchmark:
        args.verify = True
        args.benchmark = True

    all_results = {}

    if args.verify:
        print("\n=== Numerical Verification ===")
        all_results["verification"] = run_all_verification()

    if args.benchmark:
        print("\n=== Benchmarks ===")
        all_results["benchmarks"] = run_benchmarks(args.device)

    if all_results:
        Path(args.out).parent.mkdir(exist_ok=True)
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\n[saved] {args.out}")

    # Summary
    if "verification" in all_results:
        passed = sum(1 for v in all_results["verification"]["verifications"] if v["passed"])
        total = len(all_results["verification"]["verifications"])
        print(f"\nVerification: {passed}/{total} passed")

    if "benchmarks" in all_results:
        print("\nBenchmark Summary:")
        for b in all_results["benchmarks"]:
            cfg = b["config"]
            print(f"  L={cfg['L']:4d} block={cfg['block_size']:3d} | "
                  f"ref={b['reference']['mean_s']*1000:6.2f}ms "
                  f"tiled={b['tiled']['mean_s']*1000:6.2f}ms "
                  f"speedup={b['speedup']:.2f}x "
                  f"mem_ratio={b['mem_ratio']:.2f}x")


if __name__ == "__main__":
    main()