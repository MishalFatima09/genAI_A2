"""Task 5: Speculative Decoding.

Greedy speculative decoding with explicit KV cache management.
Based on the reference implementation with ExplicitKVCache.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is in sys.path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kv_cache import ExplicitKVCache, generate_with_kv_cache  # noqa: E402

# Monkey-patch to avoid accelerate dispatch error with single GPU
import transformers.modeling_utils
transformers.modeling_utils.PreTrainedModel.to = lambda self, *args, **kwargs: nn.Module.to(self, *args, **kwargs)

DRAFT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class SpeculativeResult:
    tokens: list[int]
    prefill_time: float
    decode_time: float
    peak_mem_mb: float
    n_drafted: int
    n_accepted: int
    acceptance_rate: float
    step_details: List[Dict[str, Any]] = field(default_factory=list)


def load_models(
    target_path: str,
    base_model: str = BASE_MODEL_ID,
    draft_name: str = DRAFT_MODEL_ID,
    device: str = "cuda",
    draft_in_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """Load draft, target (LoRA), and shared tokenizer. Verify tokenizer compatibility."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    compute_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load shared tokenizer
    target_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_name, trust_remote_code=True)

    # Verify tokenizer compatibility
    assert len(target_tokenizer) == len(draft_tokenizer), "Tokenizer vocab size mismatch"
    assert target_tokenizer.eos_token_id == draft_tokenizer.eos_token_id, "EOS token mismatch"
    assert target_tokenizer.pad_token_id == draft_tokenizer.pad_token_id, "PAD token mismatch"

    # Load target model (base + LoRA)
    target_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=(device == "cuda"),
        torch_dtype=compute_dtype,
        device_map={"": device} if device == "cuda" else None,
        trust_remote_code=True,
    )

    if Path(target_path).is_dir():
        target_model = PeftModel.from_pretrained(
            target_model,
            target_path,
            torch_dtype=compute_dtype,
            device_map={"": device} if device == "cuda" else None,
        )

    target_model.eval()

    # Load draft model
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_name,
        load_in_4bit=(draft_in_4bit and device == "cuda"),
        torch_dtype=compute_dtype,
        device_map={"": device} if device == "cuda" else None,
        trust_remote_code=True,
    )
    draft_model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    return target_model, draft_model, target_tokenizer


def speculative_decode(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    gamma: int = 4,
    device: str = "cuda",
) -> SpeculativeResult:
    """Greedy speculative decoding with explicit KV cache rollback."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    target_model.eval()
    draft_model.eval()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    prompt_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids, device=device)
    eos_id = tokenizer.eos_token_id

    generated_tokens: List[int] = []
    step_details: List[Dict[str, Any]] = []

    total_start = time.perf_counter()

    # --- 1. PREFILL PHASE ---
    target_cache = ExplicitKVCache()
    draft_cache = ExplicitKVCache()

    pos_ids_prefill = torch.arange(0, prompt_len, dtype=torch.long, device=device).unsqueeze(0)
    cache_pos_prefill = torch.arange(0, prompt_len, dtype=torch.long, device=device)

    prefill_start = time.perf_counter()
    with torch.no_grad():
        t_out = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos_ids_prefill,
            past_key_values=target_cache,
            use_cache=True,
            cache_position=cache_pos_prefill,
            return_dict=True,
        )
        _ = draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos_ids_prefill,
            past_key_values=draft_cache,
            use_cache=True,
            cache_position=cache_pos_prefill,
            return_dict=True,
        )

    if device == "cuda":
        torch.cuda.synchronize()

    prefill_time = time.perf_counter() - prefill_start

    # First token from target
    first_logits = t_out.logits[:, -1, :].clone()
    del t_out
    first_token = torch.argmax(first_logits, dim=-1, keepdim=True)
    first_token_id = first_token.item()
    generated_tokens.append(first_token_id)

    x_last = first_token
    x_last_id = first_token_id

    total_draft_time = 0.0
    total_verify_time = 0.0
    total_proposed = 0
    total_accepted = 0
    verification_step_count = 0

    # --- 2. SPECULATIVE GENERATION LOOP ---
    while len(generated_tokens) < max_new_tokens:
        if x_last_id == eos_id:
            break

        verification_step_count += 1
        base_cache_len = target_cache.get_seq_length()
        assert base_cache_len == draft_cache.get_seq_length(), "Cache length mismatch"

        current_gamma = min(gamma, max_new_tokens - len(generated_tokens))
        if current_gamma <= 0:
            break

        # --- A. Candidate-Token Drafting ---
        draft_start = time.perf_counter()
        draft_tokens: List[torch.Tensor] = []
        cur_input = x_last

        for j in range(current_gamma):
            cur_pos = base_cache_len + j
            d_pos_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
            d_cache_pos = torch.tensor([cur_pos], dtype=torch.long, device=device)
            d_mask = torch.ones((1, cur_pos + 1), dtype=torch.long, device=device)

            with torch.no_grad():
                d_out = draft_model(
                    input_ids=cur_input,
                    attention_mask=d_mask,
                    position_ids=d_pos_ids,
                    past_key_values=draft_cache,
                    use_cache=True,
                    cache_position=d_cache_pos,
                    return_dict=True,
                )

            next_draft = torch.argmax(d_out.logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens.append(next_draft)
            cur_input = next_draft
            del d_out

            if next_draft.item() == eos_id:
                break

        if device == "cuda":
            torch.cuda.synchronize()
        total_draft_time += time.perf_counter() - draft_start

        num_drafted = len(draft_tokens)
        total_proposed += num_drafted

        # --- B. Target Model Parallel Verification ---
        verify_inputs = torch.cat([x_last] + draft_tokens, dim=-1)
        k_plus_1 = num_drafted + 1

        v_pos_ids = torch.arange(
            base_cache_len, base_cache_len + k_plus_1, dtype=torch.long, device=device
        ).unsqueeze(0)
        v_cache_pos = torch.arange(
            base_cache_len, base_cache_len + k_plus_1, dtype=torch.long, device=device
        )
        v_mask = torch.ones((1, base_cache_len + k_plus_1), dtype=torch.long, device=device)

        verify_start = time.perf_counter()
        with torch.no_grad():
            t_v_out = target_model(
                input_ids=verify_inputs,
                attention_mask=v_mask,
                position_ids=v_pos_ids,
                past_key_values=target_cache,
                use_cache=True,
                cache_position=v_cache_pos,
                return_dict=True,
            )

        if device == "cuda":
            torch.cuda.synchronize()
        total_verify_time += time.perf_counter() - verify_start

        target_predictions = torch.argmax(t_v_out.logits[0, :k_plus_1, :], dim=-1)
        del t_v_out

        # --- C. Prefix Acceptance & Cache Rollback ---
        num_accepted = 0
        eos_stopped = False
        for i in range(num_drafted):
            if len(generated_tokens) >= max_new_tokens:
                break
            target_pred_id = target_predictions[i].item()
            draft_id = draft_tokens[i].item()

            if target_pred_id == draft_id:
                num_accepted += 1
                generated_tokens.append(draft_id)
                if draft_id == eos_id:
                    eos_stopped = True
                    break
            else:
                break

        total_accepted += num_accepted

        # Correction or bonus token
        chosen_next_token_id = target_predictions[num_accepted].item()
        chosen_next_token = target_predictions[num_accepted].view(1, 1)

        if not eos_stopped and len(generated_tokens) < max_new_tokens:
            generated_tokens.append(chosen_next_token_id)
            if chosen_next_token_id == eos_id:
                eos_stopped = True

        # Rollback caches
        new_valid_cache_len = base_cache_len + 1 + num_accepted
        target_cache.crop(new_valid_cache_len)

        if num_accepted == num_drafted:
            # Sync draft cache
            last_draft_pos = base_cache_len + num_drafted
            sync_pos_ids = torch.tensor([[last_draft_pos]], dtype=torch.long, device=device)
            sync_cache_pos = torch.tensor([last_draft_pos], dtype=torch.long, device=device)
            sync_mask = torch.ones((1, last_draft_pos + 1), dtype=torch.long, device=device)
            with torch.no_grad():
                _ = draft_model(
                    input_ids=draft_tokens[-1],
                    attention_mask=sync_mask,
                    position_ids=sync_pos_ids,
                    past_key_values=draft_cache,
                    use_cache=True,
                    cache_position=sync_cache_pos,
                    return_dict=True,
                )
        else:
            draft_cache.crop(new_valid_cache_len)

        x_last = chosen_next_token
        x_last_id = chosen_next_token_id

        step_details.append({
            "step": verification_step_count,
            "drafted": num_drafted,
            "accepted": num_accepted,
            "bonus_or_correction_token": chosen_next_token_id,
            "tokens_generated_so_far": len(generated_tokens),
        })

        if eos_stopped or len(generated_tokens) >= max_new_tokens:
            break

    if device == "cuda":
        torch.cuda.synchronize()

    decode_time = time.perf_counter() - total_start - prefill_time

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0.0
    acceptance_rate = total_accepted / max(total_proposed, 1)

    return SpeculativeResult(
        tokens=generated_tokens,
        prefill_time=prefill_time,
        decode_time=decode_time,
        peak_mem_mb=peak_mem,
        n_drafted=total_proposed,
        n_accepted=total_accepted,
        acceptance_rate=acceptance_rate,
        step_details=step_details,
    )


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

        greedy_tokens, _, _, _ = greedy_decode(target_model, tokenizer, input_ids, max_new_tokens, device)
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
    draft, target, tokenizer = load_models(target_path, base_model, draft_name, device)

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
                for key in ["prefill_s", "decode_s", "tokens_s", "peak_mem_mb",
                           "acceptance_rate", "n_drafted", "n_accepted"]:
                    if key in config_result[method]:
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
    parser.add_argument("--draft", default=DRAFT_MODEL_ID)
    parser.add_argument("--target", required=True, help="Path to LoRA adapter")
    parser.add_argument("--base", default=BASE_MODEL_ID)
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
    draft, target, tokenizer = load_models(args.target, args.base, args.draft, device)

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