# Complete fix for speculative.py on Kaggle - run this cell
with open('/kaggle/working/genAI_A2/code/speculative.py', 'r') as f:
    content = f.read()

# 1. Ensure DynamicCache import
if 'from transformers.cache_utils import DynamicCache' not in content:
    content = content.replace(
        'from transformers import AutoModelForCausalLM, AutoTokenizer',
        'from transformers import AutoModelForCausalLM, AutoTokenizer\nfrom transformers.cache_utils import DynamicCache'
    )

# 2. Replace the entire speculative_decode function with fixed version
old_function_start = 'def speculative_decode('
old_function_end = 'def verify_token_equality('

start_idx = content.find(old_function_start)
end_idx = content.find(old_function_end, start_idx)

new_function = '''def speculative_decode(
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
            target_out = target_model(
                input_ids=draft_sequence,
                use_cache=True,
                past_key_values=target_pkv,
            )
            target_pkv = target_out.past_key_values

            # --- FIND ACCEPTANCE PREFIX ---
            target_logits = target_out.logits  # (1, γ, V)
            target_greedy = torch.argmax(target_logits, dim=-1)  # (1, γ)

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
                # Update both caches to accepted length using DynamicCache
                target_pkv = _truncate_kv_cache(target_pkv, input_ids.shape[1] + accept_len)
                draft_pkv = _truncate_kv_cache(draft_pkv, input_ids.shape[1] + accept_len)

            if accept_len < draft_sequence.shape[1]:
                # First mismatch: replace with target's token
                correct_token = target_greedy[:, accept_len:accept_len+1]
                generated = generated[:, :input_ids.shape[1] + accept_len]
                generated = torch.cat([generated, correct_token], dim=-1)

                # Advance both caches by 1 with the corrected token
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

                if correct_token.item() == tokenizer.eos_token_id:
                    break
            else:
                # All γ accepted: run target on last token to get fresh logits + cache
                last_token = generated[:, -1:]
                target_out = target_model(
                    input_ids=last_token,
                    use_cache=True,
                    past_key_values=target_pkv,
                )
                target_pkv = target_out.past_key_values
                # Use target's output for next draft round
                draft_out = target_out

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
    """Truncate KV cache to new_seq_len (keep prefix).

    Uses DynamicCache to preserve cache class methods (get_seq_length, etc.).
    """
    if past_key_values is None:
        return None
    
    # Convert to DynamicCache if it's a tuple
    if isinstance(past_key_values, tuple):
        cache = DynamicCache.from_legacy_cache(past_key_values)
    else:
        cache = past_key_values
    
    # Truncate each layer
    for layer_idx in range(len(cache.key_cache)):
        cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :new_seq_len, :]
        cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :new_seq_len, :]
    
    # Update seen tokens
    cache._seen_tokens = new_seq_len
    
    return cache


def verify_token_equality('''

content = content[:start_idx] + new_function + content[end_idx:]

with open('/kaggle/working/genAI_A2/code/speculative.py', 'w') as f:
    f.write(content)

print("Fixed!")