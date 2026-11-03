# Model Selection (Task 1, A2) — Candidates and Justification

Requirements from the assignment: open-weight decoder-only Transformer,
locally executable, supports autoregressive inference work (KV cache, tiled
attention, speculative decoding), fits free compute (Google Colab free T4 =
16 GB VRAM; student's local machine is **CPU-only**), licensed so paid APIs or
paid GPUs are never necessary.

## 1. Candidates considered

| Property | **Qwen2.5-1.5B-Instruct (chosen)** | Qwen2.5-0.5B-Instruct | Llama-3.2-1B-Instruct | Phi-3.5-mini-instruct | Mistral-7B-Instruct-v0.3 |
|---|---|---|---|---|---|
| Parameters | **1.54 B** | 0.49 B | 1.24 B | 3.82 B | 7.29 B |
| Architecture | decoder-only, pre-norm RMSNorm, SwiGLU, RoPE, **GQA (12 Q / 2 KV heads)**, 28 layers, d_model 1536 | same family, 24 layers, d 896 | decoder-only, RMSNorm, SwiGLU, RoPE, GQA (32 Q / 4 KV), 16 layers, d 2048 | decoder-only, RMSNorm, SwiGLU, RoPE, dense, 32 layers, d 3072 | decoder-only, RMSNorm, SwiGLU, RoPE, GQA (32 Q / 8 KV), 32 layers, d 4096 |
| Context window | 32,768 tokens (native) | 32,768 | 131,072 | 131,072 | 32,768 |
| License | **Apache-2.0** | Apache-2.0 | Llama 3.2 Community License (custom terms, gated download) | MIT | Apache-2.0 |
| Weights size (bf16/fp16) | ≈ 3.1 GB | ≈ 1.0 GB | ≈ 2.5 GB | ≈ 7.6 GB | ≈ 14.6 GB |
| Colab T4 (16 GB) feasibility | **Easy**: fp16 inference + full LoRA fit with batch 4; 4-bit optional | Very easy | Easy | Tight for long-context + LoRA | Infeasible in fp16 (>14 GB + activations + optimizer); only 4-bit QLoRA, very slow |
| Local CPU feasibility | OK for eval/inference (short generations) | Best | OK | Slow | Impractical |

## 2. Decision: Qwen2.5-1.5B-Instruct

**Final choice: `Qwen/Qwen2.5-1.5B-Instruct`, Apache-2.0.**

Justification points (A2):

1. **License:** Apache-2.0 — no gated acceptance, no usage-restriction clause
   (unlike Llama 3.2's community license), so the GitHub repository and hosted
   adapters can be redistributed freely.
2. **Parameters / memory:** 1.54 B → ~3.1 GB in fp16, leaving >11 GB of the
   T4 for activations, KV cache, and LoRA optimizer state. Mistral-7B (7.29 B)
   would require 4-bit QLoRA and still leave negligible room for long
   patient notes × many criteria per batch; rejected on memory.
3. **Architecture:** GQA with 2 KV heads materially shrinks KV-cache memory
   and speed (Task 3) while keeping quality — a better test-bed for the
   cache-aware decoding work than a MHA model; 128-dim head / 12 heads, RoPE
   for explicit positional handling to explain in the report.
4. **Context window:** native 32 k tokens ≫ our longest (note + criterion)
   prompt (see tokenization report: P99 ≈ 1.5 k tokens), so truncation is a
   safety net, not a routine event — an important claim for evidence recall.
5. **Hardware fit:** fine-tunes with LoRA in fp16 on a free Colab T4 in
   minutes-to-low-hours for our ~880 training rows; inference experiments
   (KV cache, speculative decoding with a 0.5B draft model from the same
   family/tokenizer) all run locally on CPU and on T4.
6. **Speculative-decoding compatibility:** `Qwen2.5-0.5B-Instruct` shares the
   exact same tokenizer (vocab 152,064, byte-level BPE) as the 1.5B target —
   a required property for Task 5 draft/target pairing.

**Rejected alternatives (A3):**
- *Mistral-7B (Apache-2.0)*: quality ceiling higher, but 7.3 B params exceed
  free-T4 comfortable limits for our multi-epoch run; slower for the 6-task
  iteration loop; rejected on compute feasibility, kept as an extension.
- *Llama-3.2-1B*: competitive quality, but gated/custom license conflicts
  with redistribution requirements; rejected on licensing.
- *Phi-3.5-mini (3.8 B, MIT)*: strong reasoning, but ~7.6 GB fp16 weights
  squeeze the T4 when combined with long-context activations; rejected on
  memory headroom.
- *Qwen2.5-0.5B*: kept as **draft model** (Task 5) instead of target — too
  small to be the primary clinical reasoner.

## 3. Compute and memory plan

| Item | Setting | Est. memory |
|---|---|---|
| Training platform | Google Colab free T4 (16 GB, sm_75, **no bf16 — use fp16**) | — |
| Base model | Qwen2.5-1.5B-Instruct, frozen | ~3.1 GB (fp16) |
| Method | LoRA (r=16, α=32, dropout 0.05, q/k/v/o + MLP projections) | ~20–60 MB trainable |
| Precision | fp16 autocast (T4 lacks bf16); gradient checkpointing ON | activations fit batch 4 × 2048 tok |
| Max seq len | 2048 (documented in truncation report) | KV activations OK |
| Local evaluation | CPU torch 2.x, fp32 for eval determinism | 8–12 GB RAM |

## 4. Alternatives considered / difficulties (A3 log)

| Decision | Alternatives | Evidence for final choice |
|---|---|---|
| LoRA vs full FT vs prompt-tuning | Full FT (memory-heavy, overfits 880 rows), prompt-tuning (weaker on structured output), BitFit (less expressive) | S9/S10/S11 in `literature_review.md`; LoRA merges to zero inference overhead |
| r=16 vs r=4 / r=64 | r=4 underfit structured schema in pilot; r=64 added params with no val gain (pilot run) | pilot logs (results/) |
| fp16 vs 4-bit QLoRA on T4 | bf16 unsupported on T4; 4-bit saved VRAM but slowed step time and complicated numerics | measured step time; 1.5B does not need quantization |
| Greedy decoding for eval | temperature sampling for diversity | determinism required for B4/D2/F3 token-agreement tests |
