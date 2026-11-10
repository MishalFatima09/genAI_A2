"""Task 4: Efficient Tiled Attention with Online Softmax.

Implements:
1. Reference scaled dot-product attention (causal, numerically stable)
2. Tiled/blocked attention with running max/normalizer (FlashAttention-style)
3. Numerical verification on small tensors
4. Honest benchmark comparison
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class AttentionResult:
    output: torch.Tensor
    time_s: float
    peak_mem_mb: float


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference scaled dot-product attention with stable softmax.

    Args:
        q: (B, H, L, D)
        k: (B, H, L, D)
        v: (B, H, L, D)
        causal: apply causal mask
        sm_scale: scale factor (default 1/sqrt(D))
    Returns:
        out: (B, H, L, D)
    """
    B, H, L, D = q.shape
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)

    if causal:
        mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))

    # Numerically stable softmax: subtract row max before exp
    row_max = scores.max(dim=-1, keepdim=True).values
    scores_exp = torch.exp(scores - row_max)
    row_sum = scores_exp.sum(dim=-1, keepdim=True)
    attn = scores_exp / row_sum

    out = torch.matmul(attn, v)
    return out


def tiled_attention_online_softmax(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Tiled attention with online softmax (FlashAttention algorithm).

    Processes Q/K/V in blocks to reduce memory from O(L^2) to O(L * block_size).
    Maintains running row max and normalizer for numerically stable softmax.

    Args:
        q: (B, H, L, D)
        k: (B, H, L, D)
        v: (B, H, L, D)
        block_size: tile size for K/V (and Q blocks)
        causal: apply causal mask
        sm_scale: scale factor
    Returns:
        out: (B, H, L, D)
    """
    B, H, L, D = q.shape
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)

    # Output accumulator and online softmax state
    out = torch.zeros_like(q)
    row_max = torch.full((B, H, L), float("-inf"), device=q.device, dtype=q.dtype)
    row_sum = torch.zeros((B, H, L), device=q.device, dtype=q.dtype)

    # Iterate over Q blocks (rows of attention matrix)
    for q_start in range(0, L, block_size):
        q_end = min(q_start + block_size, L)
        q_block = q[:, :, q_start:q_end, :]  # (B, H, Bq, D)

        # Initialize block outputs
        block_out = torch.zeros_like(q_block)
        block_max = torch.full((B, H, q_end - q_start), float("-inf"), device=q.device, dtype=q.dtype)
        block_sum = torch.zeros((B, H, q_end - q_start), device=q.device, dtype=q.dtype)

        # Iterate over K/V blocks (columns of attention matrix)
        for kv_start in range(0, L, block_size):
            kv_end = min(kv_start + block_size, L)
            k_block = k[:, :, kv_start:kv_end, :]  # (B, H, Bkv, D)
            v_block = v[:, :, kv_start:kv_end, :]  # (B, H, Bkv, D)

            # Compute scores for this tile: (B, H, Bq, Bkv)
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            if causal:
                # Causal mask: only allow kv_idx <= q_idx
                q_indices = torch.arange(q_start, q_end, device=q.device).view(1, 1, -1, 1)
                kv_indices = torch.arange(kv_start, kv_end, device=q.device).view(1, 1, 1, -1)
                mask = kv_indices > q_indices  # (1, 1, Bq, Bkv)
                scores.masked_fill_(mask, float("-inf"))

            # Online softmax update for this tile
            # Current tile's row max
            tile_max = scores.max(dim=-1).values  # (B, H, Bq)

            # New row max = max(old_max, tile_max)
            new_max = torch.maximum(block_max, tile_max)

            # Rescale old sum and tile exp
            old_scale = torch.exp(block_max - new_max)
            tile_exp = torch.exp(scores - new_max.unsqueeze(-1))
            tile_sum = tile_exp.sum(dim=-1)  # (B, H, Bq)

            # Update block sum: rescale old + add tile
            block_sum = block_sum * old_scale + tile_sum
            block_max = new_max

            # Update block output: rescale old + add tile contribution
            block_out = block_out * old_scale.unsqueeze(-1) + torch.matmul(tile_exp, v_block)

        # Store block results
        out[:, :, q_start:q_end, :] = block_out
        row_max[:, :, q_start:q_end] = block_max
        row_sum[:, :, q_start:q_end] = block_sum

    # Final normalization
    out = out / row_sum.unsqueeze(-1)
    return out


def verify_numerical(
    B: int = 2,
    H: int = 4,
    L: int = 128,
    D: int = 64,
    block_size: int = 32,
    causal: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-2,
) -> dict:
    """Verify tiled attention matches reference on small tensors.

    Note: rtol is relaxed to 1e-2 because attention outputs contain many
    near-zero values (softmax tail) where relative error blows up despite
    tiny absolute error (~1e-7). This is expected with different accumulation
    order in tiled vs full-matrix computation.
    """
    torch.manual_seed(42)
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    ref = reference_attention(q, k, v, causal=causal)
    tiled = tiled_attention_online_softmax(q, k, v, block_size=block_size, causal=causal)

    max_abs_diff = (ref - tiled).abs().max().item()
    mean_abs_diff = (ref - tiled).abs().mean().item()

    # Relative error only on values above threshold (avoids softmax tail blowup)
    mask = ref.abs() > 1e-3
    if mask.any():
        max_rel_diff = ((ref - tiled).abs()[mask] / ref.abs()[mask]).max().item()
    else:
        max_rel_diff = 0.0

    # Also use torch.allclose for standard comparison
    allclose_passed = torch.allclose(ref, tiled, atol=atol, rtol=rtol)

    passed = max_abs_diff <= atol and max_rel_diff <= rtol and allclose_passed

    return {
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "allclose_passed": allclose_passed,
        "atol": atol,
        "rtol": rtol,
        "config": {"B": B, "H": H, "L": L, "D": D, "block_size": block_size, "causal": causal},
    }


def benchmark_attention(
    B: int = 1,
    H: int = 12,
    L: int = 1024,
    D: int = 128,
    block_size: int = 64,
    causal: bool = True,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    n_warmup: int = 3,
    n_repeat: int = 10,
) -> dict:
    """Benchmark reference vs tiled attention."""
    torch.manual_seed(123)
    q = torch.randn(B, H, L, D, dtype=dtype, device=device)
    k = torch.randn(B, H, L, D, dtype=dtype, device=device)
    v = torch.randn(B, H, L, D, dtype=dtype, device=device)

    # Reference
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ref_times = []
    for _ in range(n_warmup):
        _ = reference_attention(q, k, v, causal=causal)
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = reference_attention(q, k, v, causal=causal)
        torch.cuda.synchronize() if device == "cuda" else None
        ref_times.append(time.perf_counter() - t0)
    ref_peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0.0

    # Tiled
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tiled_times = []
    for _ in range(n_warmup):
        _ = tiled_attention_online_softmax(q, k, v, block_size=block_size, causal=causal)
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = tiled_attention_online_softmax(q, k, v, block_size=block_size, causal=causal)
        torch.cuda.synchronize() if device == "cuda" else None
        tiled_times.append(time.perf_counter() - t0)
    tiled_peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0.0

    return {
        "config": {"B": B, "H": H, "L": L, "D": D, "block_size": block_size, "causal": causal, "dtype": str(dtype)},
        "reference": {
            "mean_s": sum(ref_times) / len(ref_times),
            "std_s": (sum((t - sum(ref_times)/len(ref_times))**2 for t in ref_times) / len(ref_times))**0.5,
            "peak_mem_mb": ref_peak_mem,
        },
        "tiled": {
            "mean_s": sum(tiled_times) / len(tiled_times),
            "std_s": (sum((t - sum(tiled_times)/len(tiled_times))**2 for t in tiled_times) / len(tiled_times))**0.5,
            "peak_mem_mb": tiled_peak_mem,
        },
        "speedup": (sum(ref_times) / len(ref_times)) / (sum(tiled_times) / len(tiled_times)) if tiled_times else 0,
        "mem_ratio": tiled_peak_mem / ref_peak_mem if ref_peak_mem else 0,
    }


def run_all_verification() -> dict:
    """Run numerical verification across multiple configurations."""
    configs = [
        {"B": 1, "H": 4, "L": 64, "D": 32, "block_size": 16, "causal": True},
        {"B": 2, "H": 8, "L": 128, "D": 64, "block_size": 32, "causal": True},
        {"B": 1, "H": 4, "L": 256, "D": 64, "block_size": 64, "causal": True},
        {"B": 1, "H": 4, "L": 128, "D": 64, "block_size": 32, "causal": False},
    ]
    results = []
    for cfg in configs:
        res = verify_numerical(**cfg)
        results.append(res)
        status = "PASS" if res["passed"] else "FAIL"
        print(f"  {status}: L={cfg['L']} B={cfg['B']} H={cfg['H']} D={cfg['D']} "
              f"max_abs={res['max_abs_diff']:.2e} max_rel={res['max_rel_diff']:.2e} "
              f"allclose={res['allclose_passed']}")
    return {"verifications": results}


def run_benchmarks(device: str = "cuda") -> list[dict]:
    """Run benchmarks across multiple sequence lengths."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA not available, running on CPU")

    configs = [
        {"L": 512, "block_size": 64},
        {"L": 1024, "block_size": 64},
        {"L": 2048, "block_size": 128},
    ]
    results = []
    for cfg in configs:
        print(f"\n[bench] L={cfg['L']} block={cfg['block_size']} device={device}")
        res = benchmark_attention(L=cfg["L"], block_size=cfg["block_size"], device=device)
        results.append(res)
        print(f"  ref: {res['reference']['mean_s']*1000:.2f}ms {res['reference']['peak_mem_mb']:.1f}MB")
        print(f"  tiled: {res['tiled']['mean_s']*1000:.2f}ms {res['tiled']['peak_mem_mb']:.1f}MB")
        print(f"  speedup: {res['speedup']:.2f}x mem_ratio: {res['mem_ratio']:.2f}x")
    return results


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Run numerical verification")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out", default="results/task4_results.json")
    args = parser.parse_args()

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