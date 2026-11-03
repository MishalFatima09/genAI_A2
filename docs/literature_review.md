# Literature Review (Task 1, A1) — Sources and Critical Notes

Research themes: (T1) clinical eligibility reasoning with LLMs, (T2) Transformer
architecture and attention, (T3) supervised / parameter-efficient adaptation,
(T4) efficient autoregressive inference, (T5) clinical NLP evaluation practice.
All entries verified against arXiv / publisher metadata (Oct 2026). This file is
the working source for the Related Work section of the IEEE report.

---

## Theme 1 — Clinical trial eligibility matching with LLMs

### [S1] TrialGPT (primary system paper)
Q. Jin, Z. Wang, C. S. Floudas, F. Chen, C. Gong, D. Bracken-Clarke, E. Xue,
Y. Yang, J. Sun, and Z. Lu, "Matching patients to clinical trials with large
language models," *Nature Communications*, vol. 15, art. no. 9074, pp. 1–14, 2024.
DOI: 10.1038/s41467-024-53081-z (arXiv:2307.15051).

- Zero-shot patient–trial matching with three modules: Retrieval (candidate
  filtering), Matching (criterion-level label + explanation + evidence-sentence
  location), Ranking (trial-level aggregation). Evaluated on 183 synthetic
  patients, >75,000 annotations, and a 1,015 patient–criterion pair physician
  study (the exact rows of our mandatory dataset).
- **Relevance:** Defines the task, label space, and output schema we reproduce
  (`included / not included / excluded / not excluded / not enough information /
  not applicable`, plus sentence-ID evidence). Our contribution is supervised
  adaptation of an open model instead of zero-shot GPT-4.
- **Citable numbers:** criterion accuracy 87.3% (human 88.7–90.0%); inclusion
  89.9% / exclusion 85.9%; evidence P 90.1 / R 87.9 / **F1 88.6**; ranking 0.7314
  vs 0.5085 baseline; 42.6% screening-time reduction.

### [S2] Zero-shot matching on n2c2
M. Wornow, A. Lozano, D. Dash, J. Jindal, K. W. Mahaffey, and N. H. Shah,
"Zero-shot clinical trial patient matching with LLMs," arXiv:2402.05125, 2024.

- Zero-shot LLM pipeline for free-text criterion satisfaction; SOTA on n2c2
  2018; two-stage retrieval cuts tokens processed by up to one third;
  clinicians judged justifications coherent for 97% of correct decisions.
- **Relevance:** Second domain benchmark; motivates our justification/evidence
  head and the token-efficiency angle (prefix reuse, caching) used later.
- **Citable numbers:** up to 33% token reduction; explanation faithfulness
  97% (correct) / 75% (incorrect).

### [S3] Trial-LLAMA (distillation / open fine-tuning)
M. Nievas, A. Basu, Y. Wang, and H. Singh, "Distilling large language models
for matching patients to clinical trials," *JAMIA*, vol. 31, no. 9,
pp. 1953–1963, 2024. DOI: 10.1093/jamia/ocae073.

- Benchmarks GPT-3.5/GPT-4/LLaMA-7B/13B/70B on criterion-level eligibility;
  builds synthetic GPT-4 training data and fine-tunes open models into
  Trial-LLAMA.
- **Relevance:** Closest published precedent for *our* approach — supervised
  fine-tuning of an open decoder on trial-matching annotations; justifies
  local fine-tuning over API calls (privacy, reproducibility, cost).
- **Citable numbers:** fine-tuned open models reach GPT-3.5 parity on limited
  synthetic data.

### [S4] 2025 systematic review
B. A. Morrison, M. Sushil, and J. S. Young, "A systematic review of
trial-matching pipelines using large language models," arXiv:2509.19327, 2025.

- 126 articles screened, 31 included; taxonomy patient-to-criterion /
  patient-to-trial / trial-to-patient / binary eligibility. GPT-4 strongest but
  costliest; dataset and metric heterogeneity blocks comparison.
- **Relevance:** Related-work backbone; motivates explicit, standard metric
  reporting (accuracy, macro-F1, evidence P/R/F1) and reporting latency/cost
  alongside accuracy — the niche of our efficient-inference tasks.
- **Citable numbers:** only 31/126 studies comparable; 16 used synthetic vs 14
  real patient data.

### [S5] QLoRA for eligibility classification
S. Ray, A. N. Sarker, N. Chatterjee, K. Bhowmik, and S. Dey, "Leveraging large
language models for clinical trial eligibility criteria classification,"
*Digital*, vol. 5, no. 2, art. no. 12, 2025. DOI: 10.3390/digital5020012.

- QLoRA fine-tunes LLaMA-2-7B-chat on 700k eligibility statements for binary
  patient-statement classification.
- **Relevance:** Direct evidence that LoRA-family adaptation works for trial
  eligibility text; supports our choice of PEFT over full fine-tuning.
- **Citable numbers:** 93% accuracy / F1 0.93.

---

## Theme 2 — Transformer architecture and efficient attention

### [S6] The Transformer
A. Vaswani et al., "Attention is all you need," in *NeurIPS*, vol. 30, 2017,
pp. 5998–6008 (arXiv:1706.03762).

- Scaled dot-product multi-head attention; O(N²) memory/compute in sequence
  length is the root cause of every efficiency problem addressed in Tasks 3–5.
- **Relevance:** Background equations: softmax(QKᵀ/√d_k)V, causal masking,
  positional encodings; basis for our KV-cache discussion.

### [S7] FlashAttention (online softmax + tiling)
T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and
memory-efficient exact attention with IO-awareness," in *NeurIPS*, vol. 35,
2022, pp. 16344–16359 (arXiv:2205.14135).

- Tiling Q/K/V between HBM and SRAM with a streaming (online) softmax that
  maintains running row maxima/normalizers; **exact** attention in O(N) extra
  memory.
- **Relevance:** Canonical algorithm behind our Task 4 tiled online-softmax
  attention; proves the speed/memory gain costs zero accuracy — essential for
  a clinical screening claim.
- **Citable numbers:** 3× speedup (GPT-2, 1K), 10–20× memory savings.

### [S8] FlashAttention-2
T. Dao, "FlashAttention-2: Faster attention with better parallelism and work
partitioning," arXiv:2307.08691, 2023.

- Fewer non-matmul ops (log-sum-exp bookkeeping only), parallelism over
  sequence length, better warp partitioning; ~2× over FlashAttention; 50–73%
  of A100 peak FLOPs.
- **Relevance:** Implementation detail reference for our kernel loop order and
  causal-block skipping; also the argument that fused kernels beat educational
  PyTorch loops (expected in our honest benchmark).

---

## Theme 3 — Parameter-efficient supervised adaptation

### [S9] LoRA
E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and
W. Chen, "LoRA: Low-rank adaptation of large language models," in *ICLR*,
2022 (arXiv:2106.09685).

- Frozen base weights + trainable low-rank updates ΔW = BA, rank r ≪ d;
  matches full fine-tuning at a fraction of the trainable parameters; adapters
  merge into base weights → **zero added inference latency**.
- **Relevance:** Our fine-tuning method. The mergeability property keeps the
  LoRA-tuned model compatible with the KV-cache / speculative-decoding
  inference paths in Tasks 3 and 5.
- **Citable numbers:** 10,000× fewer trainable params vs GPT-3 175B full FT;
  ranks r = 1–2 often suffice.

### [S10] QLoRA
T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient
finetuning of quantized LLMs," in *NeurIPS*, vol. 36, 2023 (arXiv:2305.14314).

- 4-bit NF4 quantized frozen base + LoRA gradients, double quantization,
  paged optimizers; 65B FT memory >780 GB → <48 GB.
- **Relevance:** Fallback path if we scale to 7B+; also the memory-efficiency
  argument pair for our inference-side efficiency claims.
- **Citable numbers:** Guanaco-65B = 99.3% of ChatPT (Vicuna) in 24 h on one GPU.

### [S11] PEFT survey
V. Lialin, V. Deshpande, X. Yao, and A. Rumshisky, "Scaling down to scale up:
A guide to parameter-efficient fine-tuning," arXiv:2303.15647, 2023 (v2 2024).

- Taxonomy of 30 PEFT methods; head-to-head of 14–15 methods up to 11B params;
  methods reported to beat LoRA often fail under constrained tuning budgets;
  LoRA is the robust default.
- **Relevance:** Justifies selecting LoRA among adapters, BitFit, prompt
  tuning, (IA)³ given our fixed free-tier compute budget.
- **Citable numbers:** comparison of 14–15 methods; LoRA strongest under
  limited hyperparameter search.

---

## Theme 4 — Efficient autoregressive inference

### [S12] Speculative decoding (origin)
Y. Leviathan, M. Kalman, and Y. Matias, "Fast inference from transformers via
speculative decoding," in *ICML*, vol. 202, 2023, pp. 19274–19286
(arXiv:2211.17192).

- Small draft model proposes γ tokens; large target verifies in one batched
  pass; speculative sampling preserves the target distribution exactly;
  expected speedup (1 − α^{γ+1}) / ((1 − α)(γc + 1)).
- **Relevance:** Core of Task 5. The distribution-preservation guarantee means
  eligibility accuracy cannot regress under acceleration.
- **Citable numbers:** 2–3× wall-clock (up to 3.4× at temp 0) with identical
  outputs.

### [S13] Speculative sampling (DeepMind)
C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper,
"Accelerating large language model decoding with speculative sampling,"
arXiv:2302.01318, DeepMind, 2023.

- Independent draft-and-verify formulation; explains that decoding is
  memory-bandwidth bound, so verifying K draft tokens ≈ cost of sampling 1.
- **Relevance:** Systems argument for our report: why drafting + batched
  verification is nearly free and when it slows down (low acceptance rate).
- **Citable numbers:** 2–2.5× on Chinchilla 70B, unchanged quality.

### [S14] KV cache and PagedAttention
W. Kwon, Z. Li, S. Zhuang, L. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez,
H. Zhang, and I. Stoica, "Efficient memory management for large language
model serving with PagedAttention," in *SOSP '23*, 2023, pp. 611–626
(arXiv:2309.06180).

- Autoregressive serving stores per-token K/V states; contiguous allocation
  wastes 61–80% of the cache; paged blocks + prefix sharing fix this.
- **Relevance:** Foundational definition of the KV cache for Task 3; prefix
  sharing is exactly our repeated patient-note structure (same patient across
  many criteria) — motivates cache reuse experiments.
- **Citable numbers:** 2–4× throughput at equal latency; only 20.4–38.2% of
  KV memory useful in prior systems; KV cache ≈ 30% of A100 serving 13B.

---

## Theme 5 — Clinical NLP evaluation practice

### [S15] Shaky foundations of clinical foundation models
M. Wornow, Y. Xu, R. Thapa, B. Patel, E. Steinberg, S. L. Fleming, M. A.
Pfeffer, J. Fries, and N. H. Shah, "The shaky foundations of large language
models and foundation models for electronic health records," *npj Digital
Medicine*, vol. 6, art. no. 135, 2023. DOI: 10.1038/s41746-023-00879-8.

- Reviews 84 EHR foundation models; most evaluated on narrow tasks with little
  health-system evidence; lexical metrics (ROUGE/BLEU) correlate poorly with
  human judgment; recommends calibration, fairness, ranking-based metrics.
- **Relevance:** Justifies our metric suite — criterion accuracy + macro-F1,
  per-class P/R/F1, confusion matrices, evidence-sentence P/R/F1, and manual
  error analysis — instead of a single headline number.
- **Citable numbers:** 84 models reviewed; explicit warning against lexical
  overlap metrics.

### [S16] Evaluation protocol conventions (from [S1] + [S2] + [S4])
- De-facto conventions across TrialGPT, Wornow et al., and the Morrison review:
  (a) criterion-level accuracy/F1 vs physician consensus; (b) evidence
  retrieval scored as P/R/F1 over sentence IDs; (c) trial-level ranking
  (NDCG-style); (d) human grading of explanations as correct/partial/incorrect.
- **Relevance:** Fixes our protocol: same common metrics for all design
  choices; empty-gold-evidence rows handled explicitly (see Task 2 notes).

---

## Coverage map

| Requirement | Sources |
|---|---|
| Clinical eligibility reasoning | S1, S2, S3, S4, S5 |
| Transformer / attention | S6, S7, S8 |
| Supervised adaptation (PEFT) | S9, S10, S11 |
| Efficient inference (KV cache, speculative) | S12, S13, S14 |
| Evaluation practice | S15, S16 (+S1, S2, S4) |

**Total: 16 verified sources (assignment minimum: 5).**

## AI-use disclosure (for appendix)
Literature search, metadata verification, and drafting of these notes were
performed with the assistance of an AI research assistant (web search +
arXiv/publisher cross-checks). All claims will be re-verified against the
primary sources before inclusion in the final IEEE report; all implementations
are written and tested by the student.
