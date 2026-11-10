# Patch speculative.py on Kaggle - run this in a cell BEFORE the benchmark
import re

with open('/kaggle/working/genAI_A2/code/speculative.py', 'r') as f:
    content = f.read()

# Fix 1: Add DynamicCache import
if 'from transformers.cache_utils import DynamicCache' not in content:
    content = content.replace(
        'from transformers import AutoModelForCausalLM, AutoTokenizer',
        'from transformers import AutoModelForCausalLM, AutoTokenizer\nfrom transformers.cache_utils import DynamicCache'
    )

# Fix 2: Replace _truncate_kv_cache with DynamicCache version
old_truncate = '''def _truncate_kv_cache(past_key_values, new_seq_len: int):
    """Truncate KV cache to new_seq_len (keep prefix).

    Handles different KV cache formats:
    - Standard: tuple of (key, value) per layer
    - Some models: tuple of (key, value, ...) per layer
    """
    if past_key_values is None:
        return None
    truncated = []
    for layer_kv in past_key_values:
        # layer_kv is typically (key, value) but could have more
        # Key and value are at indices 0 and 1
        k = layer_kv[0][:, :, :new_seq_len, :]
        v = layer_kv[1][:, :, :new_seq_len, :]
        # Preserve any additional elements (e.g., for some attention variants)
        if len(layer_kv) > 2:
            truncated.append((k, v) + layer_kv[2:])
        else:
            truncated.append((k, v))
    return tuple(truncated)'''

new_truncate = '''def _truncate_kv_cache(past_key_values, new_seq_len: int):
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
    
    return cache'''

content = content.replace(old_truncate, new_truncate)

# Fix 3: Fix the speculative_decode function - replace the "All γ accepted" branch
old_accepted = '''            else:
                # All γ accepted: continue with target's last logits for next draft round
                draft_out.logits = target_out.logits[:, -1:, :]'''

new_accepted = '''            else:
                # All γ accepted: run target on last accepted token to get fresh logits + cache
                last_token = generated[:, -1:]
                target_out = target_model(
                    input_ids=last_token,
                    use_cache=True,
                    past_key_values=target_pkv,
                )
                target_pkv = target_out.past_key_values
                # Use target's logits for next draft round (draft will generate from these)
                draft_out = target_out'''

content = content.replace(old_accepted, new_accepted)

with open('/kaggle/working/genAI_A2/code/speculative.py', 'w') as f:
    f.write(content)

print("Patched!")