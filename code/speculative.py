"""Task 5: Speculative Decoding.

Greedy speculative decoding with draft/target models sharing tokenizer.

Algorithm:
1. Draft model generates γ tokens autoregressively (with its own KV cache)
2. Target model verifies all γ tokens in ONE forward pass (batched, with its KV cache)
3. Accept matching prefix; on first mismatch, keep target's token, discard rest
4. Sync both KV caches to accepted prefix length
5. Repeat until max_new_tokens or EOS
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SpeculativeResult:
    tokens: list[int]
    prefill_time: float
    decode_time: float
    peak_mem_mb: float
    n_drafted: int
    n_accepted: int
    acceptance_rate: float


def greedy_decode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: str,
) -> tuple[list[int], float, float, float]:
    """Standard greedy decoding with KV cache (reference)."""
    model.eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generated = input_ids.clone()
    prefill_start = time.perf_counter()

    with torch.no_grad():
        out = model(input_ids=generated, use_cache=True)
        past_key_values = out.past_key_values
        prefill_time = time.perf_counter() - prefill_start

        decode_start = time.perf_counter()
        for _ in range(max_new_tokens):
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            out = model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
        decode_time = time.perf_counter() - decode_start

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0.0
    new_tokens = generated[0, input_ids.shape[1]:].tolist()
    return new_tokens, prefill_time, decode_time, peak_mem


def speculative_decode(
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int = 4,
    device: str = "cuda",
) -> SpeculativeResult:
    """Greedy speculative decoding (Leviathan et al., 2023).

    Args:
        draft_model: Smaller model (e.g., Qwen2.5-0.5B)
        target_model: Larger model (e.g., Qwen2.5-1.5B + LoRA)
        tokenizer: Shared tokenizer
        input_ids: (1, L)
        max_new_tokens: Total tokens to generate
        gamma: Number of draft tokens per iteration
        device: "cuda" or "cpu"
    Returns:
        SpeculativeResult with tokens and stats
    """
    draft_model.eval()
    target_model.eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generated = input_ids.clone()
    prefill_start = time.perf_counter()

    with torch.no_grad():
        # === PREFILL: both models process full prompt ===
        draft_out = draft_model(input_ids=generated, use_cache=True)
        draft_pkv = draft_out.past_key_values

        target_out = target_model(input_ids=generated, use_cache=True)
        target_pkv = target_out.past_key_values

        prefill_time = time.perf_counter() - prefill_start

        # === SPECULATIVE DECODE LOOP ===
        decode_start = time.perf_counter()
        total_drafted = 0
        total_accepted = 0

        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # --- DRAFT: generate γ tokens ---
            draft_tokens = []
            for _ in range(gamma):
                logits = draft_out.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                draft_tokens.append(next_token)
                generated = torch.cat([generated, next_token], dim=-1)

                draft_out = draft_model(
                    input_ids=next_token,
                    use_cache=True,
                    past_key_values=draft_pkv,
                )
                draft_pkv = draft_out.past_key_values

                if next_token.item() == tokenizer.eos_token_id:
                    break

            if not draft_tokens:
                break

            draft_sequence = torch.cat(draft_tokens, dim=1)  # (1, γ)
            total_drafted += draft_sequence.shape[1]

            # --- TARGET VERIFICATION: single batched forward ---
            # Target processes all draft tokens at once with its KV cache
            target_out = target_model(
                input_ids=draft_sequence,
                use_cache=True,
                past_key_values=target_pkv,
            )
            target_pkv = target_out.past_key_values

            # --- FIND ACCEPTANCE PREFIX ---
            target_logits = target_out.logits  # (1, γ, V)
            target_greedy = torch.argmax(target_logits, dim=-1)  # (1, γ)

            # Compare draft vs target greedy token-by-token
            match = (draft_sequence == target_greedy).squeeze(0)  # (γ,)
            accept_len = 0
            for m in match:
                if m.item():
                    accept_len += 1
                else:
                    break

            total_accepted += accept_len

            # --- SYNC: keep accepted prefix, correct first mismatch ---
            if accept_len > 0:
                # Keep accepted tokens in generated (already there from draft)
                # Update target KV cache to accepted length
                target_pkv = _truncate_kv_cache(target_pkv, input_ids.shape[1] + accept_len)
                draft_pkv = _truncate_kv_cache(draft_pkv, input_ids.shape[1] + accept_len)

            if accept_len < draft_sequence.shape[1]:
                # First mismatch: replace with target's token
                correct_token = target_greedy[:, accept_len:accept_len+1]
                generated = generated[:, :input_ids.shape[1] + accept_len]
                generated = torch.cat([generated, correct_token], dim=-1)

                # Advance both caches by 1 (the corrected token)
                draft_out = draft_model(
                    input_ids=correct_token,
                    use_cache=True,
                    past_key_values=draft_pkv,
                )
                draft_pkv = draft_out.past_key_values

                target_out = target_model(
                    input_ids=correct_token,
                    use_cache=True,
                    past_key_values=target_pkv,
                )
                target_pkv = target_out.past_key_values

                # If target also predicts EOS, we're done
                if correct_token.item() == tokenizer.eos_token_id:
                    break
            else:
                # All γ accepted: continue with target's last logits for next draft round
                draft_out.logits = target_out.logits[:, -1:, :]

        decode_time = time.perf_counter() - decode_start

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0.0
    new_tokens = generated[0, input_ids.shape[1]:].tolist()
    acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0

    return SpeculativeResult(
        tokens=new_tokens,
        prefill_time=prefill_time,
        decode_time=decode_time,
        peak_mem_mb=peak_mem,
        n_drafted=total_drafted,
        n_accepted=total_accepted,
        acceptance_rate=acceptance_rate,
    )


def _truncate_kv_cache(past_key_values, new_seq_len: int):
    """Truncate KV cache to new_seq_len (keep prefix)."""
    if past_key_values is None:
        return None
    # past_key_values: tuple of layers, each (K, V) with shape (B, H, L, D)
    truncated = []
    for layer_kv in past_key_values:
        k, v = layer_kv
        truncated.append((k[:, :, :new_seq_len, :], v[:, :, :new_seq_len, :]))
    return tuple(truncated)


def verify_token_equality(
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    gamma: int,
    device: str,
) -> dict:
    """Verify speculative decoding produces identical tokens to greedy target."""
    results = {"matches": 0, "mismatches": 0, "details": []}

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        # Greedy target
        greedy_tokens, _, _, _ = greedy_decode(target_model, tokenizer, input_ids, max_new_tokens, device)

        # Speculative
        spec_result = speculative_decode(draft_model, target_model, tokenizer, input_ids, max_new_tokens, gamma, device)

        match = greedy_tokens == spec_result.tokens
        results["matches" if match else "mismatches"] += 1
        results["details"].append({
            "prompt_len": input_ids.shape[1],
            "match": match,
            "greedy_tokens": greedy_tokens,
            "speculative_tokens": spec_result.tokens,
            "acceptance_rate": spec_result.acceptance_rate,
            "n_drafted": spec_result.n_drafted,
            "n_accepted": spec_result.n_accepted,
        })

    return results


def load_models(
    draft_name: str,
    target_path: str,
    base_model: str,
    device: str,
    dtype: torch.dtype,
) -> tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """Load draft, target (LoRA), and shared tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    import transformers
    kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

    draft = AutoModelForCausalLM.from_pretrained(draft_name, **{kw: dtype}, device_map=device)
    draft.eval()

    target = AutoModelForCausalLM.from_pretrained(base_model, **{kw: dtype})
    from peft import PeftModel
    target = PeftModel.from_pretrained(target, target_path).to(device)
    target.eval()

    # Verify tokenizer compatibility
    draft_tok = AutoTokenizer.from_pretrained(draft_name)
    assert draft_tok.vocab_size == tokenizer.vocab_size, "Tokenizer vocab mismatch"
    assert draft_tok.get_vocab() == tokenizer.get_vocab(), "Tokenizer vocab mismatch"

    return draft, target, tokenizer


def run_benchmark(
    draft_name: str,
    target_path: str,
    base_model: str,
    device: str = "cuda",
    prompt_lengths: list[int] = None,
    gen_lengths: list[int] = None,
    gamma: int = 4,
    n_repeats: int = 5,
    warmup: int = 2,
) -> dict:
    """Full benchmark suite."""
    if prompt_lengths is None:
        prompt_lengths = [128, 512]
    if gen_lengths is None:
        gen_lengths = [64, 128]

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32
    draft, target, tokenizer = load_models(draft_name, target_path, base_model, device, dtype)

    # Generate test prompts
    base_prompt = (
        "Patient record:\n0. Patient is a 45-year-old male with hypertension.\n"
        "1. He takes lisinopril 10mg daily.\n2. Blood pressure controlled at 130/80.\n\n"
        "Criterion type: inclusion\nCriterion: Age >= 18 years\n\nReturn the JSON decision."
    )
    prompts = {}
    for plen in prompt_lengths:
        tokens = tokenizer(base_prompt, add_special_tokens=False).input_ids
        repeat = max(1, plen // len(tokens))
        prompt_text = base_prompt * repeat
        prompts[plen] = prompt_text[:tokenizer.model_max_length]

    all_results = {
        "draft_model": draft_name,
        "target_model": target_path,
        "base_model": base_model,
        "gamma": gamma,
        "device": device,
        "configs": [],
    }

    for plen in prompt_lengths:
        for glen in gen_lengths:
            prompt = prompts[plen]
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            actual_plen = input_ids.shape[1]

            config_result = {
                "prompt_len": actual_plen,
                "gen_len": glen,
                "gamma": gamma,
                "greedy": {"prefill_s": [], "decode_s": [], "tokens_s": [], "peak_mem_mb": []},
                "speculative": {"prefill_s": [], "decode_s": [], "tokens_s": [], "peak_mem_mb": [],
                               "acceptance_rate": [], "n_drafted": [], "n_accepted": []},
            }

            # Warmup
            for _ in range(warmup):
                _ = greedy_decode(target, tokenizer, input_ids, glen, device)
                _ = speculative_decode(draft, target, tokenizer, input_ids, glen, gamma, device)

            # Timed runs
            for _ in range(n_repeats):
                g_tokens, g_prefill, g_decode, g_mem = greedy_decode(target, tokenizer, input_ids, glen, device)
                s_result = speculative_decode(draft, target, tokenizer, input_ids, glen, gamma, device)

                config_result["greedy"]["prefill_s"].append(g_prefill)
                config_result["greedy"]["decode_s"].append(g_decode)
                config_result["greedy"]["tokens_s"].append(glen / g_decode)
                config_result["greedy"]["peak_mem_mb"].append(g_mem)

                config_result["speculative"]["prefill_s"].append(s_result.prefill_time)
                config_result["speculative"]["decode_s"].append(s_result.decode_time)
                config_result["speculative"]["tokens_s"].append(glen / s_result.decode_time)
                config_result["speculative"]["peak_mem_mb"].append(s_result.peak_mem_mb)
                config_result["speculative"]["acceptance_rate"].append(s_result.acceptance_rate)
                config_result["speculative"]["n_drafted"].append(s_result.n_drafted)
                config_result["speculative"]["n_accepted"].append(s_result.n_accepted)

            # Aggregate
            for method in ["greedy", "speculative"]:
                for key in ["prefill_s", "decode_s", "tokens_s", "peak_mem_mb"]:
                    vals = config_result[method][key]
                    config_result[method][f"{key}_mean"] = round(sum(vals) / len(vals), 4)
                    config_result[method][f"{key}_std"] = round(
                        (sum((v - config_result[method][f"{key}_mean"])**2 for v in vals) / len(vals))**0.5, 4
                    )

            # Correctness check
            eq = verify_token_equality(draft, target, tokenizer, [prompt], glen, gamma, device)
            config_result["correctness"] = eq["details"][0]

            all_results["configs"].append(config_result)
            print(f"  plen={actual_plen} glen={glen} | "
                  f"greedy={config_result['greedy']['tokens_s_mean']:.1f} tok/s | "
                  f"spec={config_result['speculative']['tokens_s_mean']:.1f} tok/s | "
                  f"acc={config_result['speculative']['acceptance_rate_mean']:.2%} | "
                  f"match={eq['details'][0]['match']}")

    return all_results


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target", required=True, help="Path to LoRA adapter")
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--gen-lengths", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="results/task5_results.json")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} draft={args.draft} target={args.target}")

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

    # Extended correctness
    print("\n[verify] Extended correctness check...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    draft, target, tokenizer = load_models(args.draft, args.target, args.base, device, dtype)

    test_prompts = [
        "Patient record:\n0. 65-year-old female with diabetes.\n1. HbA1c 7.2%.\n\nCriterion type: exclusion\nCriterion: HbA1c > 8%\n\nReturn the JSON decision.",
        "Patient record:\n0. 30-year-old male.\n1. No medical history.\n2. BMI 24.\n\nCriterion type: inclusion\nCriterion: Age 18-65\n\nReturn the JSON decision.",
    ]
    eq = verify_token_equality(draft, target, tokenizer, test_prompts, 64, args.gamma, device)
    results["extended_correctness"] = {
        "matches": eq["matches"],
        "mismatches": eq["mismatches"],
        "details": eq["details"],
    }
    print(f"  Matches: {eq['matches']}, Mismatches: {eq['mismatches']}")

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