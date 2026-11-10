"""Task 3: KV-Cached Autoregressive Generation.

Implements manual KV cache handling for Qwen2 (Llama-style) architecture.
Verifies token-level equivalence with uncached reference under greedy decoding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DecodeResult:
    tokens: list[int]
    prefill_time: float
    decode_time: float
    peak_mem_mb: float


def _to_device(obj, device: str):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device) for x in obj)
    return obj


def decode_without_cache(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: str,
) -> DecodeResult:
    """Reference: re-run full sequence each step (no cache reuse)."""
    model.eval()
    torch.cuda.reset_peak_memory_stats(device) if device == "cuda" else None

    generated = input_ids.clone()
    prefill_start = time.perf_counter()

    with torch.no_grad():
        # First forward (prefill)
        out = model(input_ids=generated, use_cache=False)
        prefill_time = time.perf_counter() - prefill_start

        decode_start = time.perf_counter()
        for _ in range(max_new_tokens):
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            out = model(input_ids=generated, use_cache=False)
        decode_time = time.perf_counter() - decode_start

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0.0
    new_tokens = generated[0, input_ids.shape[1]:].tolist()
    return DecodeResult(new_tokens, prefill_time, decode_time, peak_mem)


def decode_with_cache(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: str,
) -> DecodeResult:
    """KV-cache aware: prefill once, then feed only last token + past_key_values."""
    model.eval()
    torch.cuda.reset_peak_memory_stats(device) if device == "cuda" else None

    generated = input_ids.clone()
    prefill_start = time.perf_counter()

    with torch.no_grad():
        # Prefill: full sequence, get past_key_values
        out = model(input_ids=generated, use_cache=True)
        past_key_values = out.past_key_values
        prefill_time = time.perf_counter() - prefill_start

        decode_start = time.perf_counter()
        for _ in range(max_new_tokens):
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Decode step: only last token + cache
            out = model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
        decode_time = time.perf_counter() - decode_start

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0.0
    new_tokens = generated[0, input_ids.shape[1]:].tolist()
    return DecodeResult(new_tokens, prefill_time, decode_time, peak_mem)


def verify_equivalence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: str,
) -> dict:
    """Run both decode paths, assert token equality."""
    results = {"matches": 0, "mismatches": 0, "details": []}

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        r_cached = decode_with_cache(model, input_ids, max_new_tokens, device)
        r_uncached = decode_without_cache(model, input_ids, max_new_tokens, device)

        match = r_cached.tokens == r_uncached.tokens
        results["matches" if match else "mismatches"] += 1
        results["details"].append({
            "prompt_len": input_ids.shape[1],
            "gen_len": max_new_tokens,
            "match": match,
            "cached_tokens": r_cached.tokens,
            "uncached_tokens": r_uncached.tokens,
            "cached_prefill_s": round(r_cached.prefill_time, 4),
            "cached_decode_s": round(r_cached.decode_time, 4),
            "uncached_prefill_s": round(r_uncached.prefill_time, 4),
            "uncached_decode_s": round(r_uncached.decode_time, 4),
        })

    return results


def run_benchmark(
    model_path: str,
    base_model: str,
    device: str = "cuda",
    prompt_lengths: list[int] = None,
    gen_lengths: list[int] = None,
    n_repeats: int = 5,
    warmup: int = 2,
) -> dict:
    """Full benchmark suite with multiple configs."""
    if prompt_lengths is None:
        prompt_lengths = [128, 512, 1024]
    if gen_lengths is None:
        gen_lengths = [16, 64, 128]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load fine-tuned model (LoRA merged or adapter)
    import transformers
    kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, **{kw: torch.float16}, device_map=device
    )
    model.eval()

    # Generate test prompts of varying lengths
    base_prompt = "Patient record:\n0. Patient is a 45-year-old male with hypertension.\n1. He takes lisinopril 10mg daily.\n2. Blood pressure controlled at 130/80.\n\nCriterion type: inclusion\nCriterion: Age >= 18 years\n\nReturn the JSON decision."
    prompts = {}
    for plen in prompt_lengths:
        # Repeat base prompt to reach target length
        tokens = tokenizer(base_prompt, add_special_tokens=False).input_ids
        repeat = max(1, plen // len(tokens))
        prompt_text = base_prompt * repeat
        prompts[plen] = prompt_text[:tokenizer.model_max_length]

    all_results = {
        "model": model_path,
        "base_model": base_model,
        "device": device,
        "n_repeats": n_repeats,
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
                "cached": {"prefill_s": [], "decode_s": [], "tokens_s": [], "peak_mem_mb": []},
                "uncached": {"prefill_s": [], "decode_s": [], "tokens_s": [], "peak_mem_mb": []},
            }

            # Warmup
            for _ in range(warmup):
                _ = decode_with_cache(model, input_ids, glen, device)
                _ = decode_without_cache(model, input_ids, glen, device)

            # Timed runs
            for _ in range(n_repeats):
                r_c = decode_with_cache(model, input_ids, glen, device)
                r_u = decode_without_cache(model, input_ids, glen, device)

                config_result["cached"]["prefill_s"].append(r_c.prefill_time)
                config_result["cached"]["decode_s"].append(r_c.decode_time)
                config_result["cached"]["tokens_s"].append(glen / r_c.decode_time)
                config_result["cached"]["peak_mem_mb"].append(r_c.peak_mem_mb)

                config_result["uncached"]["prefill_s"].append(r_u.prefill_time)
                config_result["uncached"]["decode_s"].append(r_u.decode_time)
                config_result["uncached"]["tokens_s"].append(glen / r_u.decode_time)
                config_result["uncached"]["peak_mem_mb"].append(r_u.peak_mem_mb)

            # Aggregate
            for method in ["cached", "uncached"]:
                for key in ["prefill_s", "decode_s", "tokens_s", "peak_mem_mb"]:
                    vals = config_result[method][key]
                    config_result[method][f"{key}_mean"] = round(sum(vals) / len(vals), 4)
                    config_result[method][f"{key}_std"] = round(
                        (sum((v - config_result[method][f"{key}_mean"])**2 for v in vals) / len(vals))**0.5, 4
                    )

            # Correctness check (last run)
            eq = verify_equivalence(model, tokenizer, [prompt], glen, device)
            config_result["correctness"] = eq["details"][0]

            all_results["configs"].append(config_result)
            print(f"  plen={actual_plen} glen={glen} | cached: {config_result['cached']['tokens_s_mean']:.1f} tok/s | uncached: {config_result['uncached']['tokens_s_mean']:.1f} tok/s | match={eq['details'][0]['match']}")

    return all_results


# ============================================================================
# ExplicitKVCache for Task 5 Speculative Decoding
# ============================================================================

class ExplicitKVCache:
    """Explicit KV cache with crop/rollback capability for speculative decoding.
    
    Implements the DynamicCache interface expected by transformers models.
    """
    
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0
    
    def __len__(self):
        return len(self.key_cache)
    
    def get_seq_length(self) -> int:
        return self._seen_tokens
    
    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Get query offset for causal masking."""
        return self._seen_tokens
    
    def get_max_cache_shape(self) -> Optional[Tuple[int, int]]:
        return None
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a specific layer."""
        # Extend layer lists if needed
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0))
            self.value_cache.append(torch.empty(0))
        
        if self.key_cache[layer_idx].numel() == 0:
            # First update for this layer
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Append to existing
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def crop(self, new_seq_len: int):
        """Crop cache to new_seq_len (keep prefix)."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel() > 0:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, :new_seq_len, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, :new_seq_len, :]
        self._seen_tokens = new_seq_len
    
    def reorder_cache(self, beam_idx: torch.Tensor):
        """Reorder cache for beam search (no-op for greedy)."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel() > 0:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(self.key_cache[layer_idx].device))
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(self.value_cache[layer_idx].device))
    
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to legacy tuple format."""
        return tuple(
            (self.key_cache[i], self.value_cache[i]) 
            for i in range(len(self.key_cache))
        )
    
    @classmethod
    def from_legacy_cache(cls, legacy_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> "ExplicitKVCache":
        """Create ExplicitKVCache from legacy tuple format."""
        cache = cls()
        for i, (k, v) in enumerate(legacy_cache):
            cache.key_cache.append(k)
            cache.value_cache.append(v)
        if cache.key_cache:
            cache._seen_tokens = cache.key_cache[0].shape[-2]
        return cache


def generate_with_kv_cache(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: str,
) -> Tuple[List[int], float, float, float]:
    """Generate using ExplicitKVCache (for compatibility)."""
    # This is a simplified version - actual implementation uses the model's built-in cache
    # For Task 3 benchmarking, use decode_with_cache instead
    return decode_with_cache(model, input_ids, max_new_tokens, device)