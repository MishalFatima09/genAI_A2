import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
})

# -----------------------------------------------------------------------------
# Figure 1: Training Loss Curve
# -----------------------------------------------------------------------------
def plot_training_curve():
    csv_path = RESULTS_DIR / "train_curve.csv"
    if not csv_path.exists():
        print("Skipping Fig 1: train_curve.csv not found")
        return
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(train_df["step"], train_df["loss"], label="Train Loss (LoRA)", color="#1f77b4", alpha=0.85, linewidth=1.5)
    if not val_df.empty:
        ax.scatter(val_df["step"], val_df["loss"], color="#d62728", s=50, zorder=5, label="Validation Checkpoint Loss")
        for _, row in val_df.iterrows():
            ax.annotate(f"{row['loss']:.4f}", (row["step"], row["loss"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#d62728")

    ax.set_xlabel("Optimization Step (Batch Size=1, Grad Accum=16)")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Qwen2.5-1.5B LoRA Fine-Tuning Convergence")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_training_curve.pdf")
    plt.savefig(FIG_DIR / "fig1_training_curve.png")
    plt.close()
    print("Generated Fig 1: Training Curve")

# -----------------------------------------------------------------------------
# Figure 2: Confusion Matrix & Per-Class F1
# -----------------------------------------------------------------------------
def plot_confusion_matrices():
    test_json = RESULTS_DIR / "metrics_after_lora_test.json"
    if not test_json.exists():
        return
    with open(test_json) as f:
        data = json.load(f)

    cm = data["label"]["overall"]["confusion_matrix"]
    matrix = np.array(cm["rows_gold_cols_pred"])
    labels = ["INC", "NOT INC", "EXC", "NOT EXC", "NI", "NA", "INV"]

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > matrix.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold" if i==j and val > 0 else "normal")

    fig.colorbar(cax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Class", labelpad=10)
    ax.set_ylabel("Physician Gold Consensus")
    ax.set_title("Held-Out Test Confusion Matrix (Task 2)", pad=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_confusion_matrix.pdf")
    plt.savefig(FIG_DIR / "fig2_confusion_matrix.png")
    plt.close()
    print("Generated Fig 2: Confusion Matrix")

# -----------------------------------------------------------------------------
# Figure 3: Task 4 Tiled Attention vs Reference SDPA
# -----------------------------------------------------------------------------
def plot_tiled_attention():
    t4_json = RESULTS_DIR / "task4_results.json"
    if not t4_json.exists():
        return
    with open(t4_json) as f:
        t4_data = json.load(f)

    benchmarks = t4_data.get("benchmarks", [])
    lengths = [b["config"]["L"] for b in benchmarks]
    ref_mem = [b["reference"]["peak_mem_mb"] for b in benchmarks]
    tiled_mem = [b["tiled"]["peak_mem_mb"] for b in benchmarks]
    mem_saved = [((1.0 - b["mem_ratio"]) * 100) for b in benchmarks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    x = np.arange(len(lengths))
    width = 0.35
    ax1.bar(x - width/2, ref_mem, width, label="Reference SDPA", color="#94a3b8")
    ax1.bar(x + width/2, tiled_mem, width, label="Tiled Online Softmax", color="#10b981")
    ax1.set_xlabel("Sequence Length (L)")
    ax1.set_ylabel("Peak Activation Memory (MB)")
    ax1.set_title("Peak Attention Memory")
    ax1.set_xticks(x)
    ax1.set_xticklabels(lengths)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(lengths, mem_saved, marker="o", color="#059669", linewidth=2, markersize=6)
    ax2.set_xlabel("Sequence Length (L)")
    ax2.set_ylabel("Memory Reduction (%)")
    ax2.set_title("Memory Savings vs Sequence Length")
    ax2.set_ylim(0, 100)
    for l, s in zip(lengths, mem_saved):
        ax2.annotate(f"{s:.1f}%", (l, s), textcoords="offset points", xytext=(0, 7), ha="center", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_tiled_attention.pdf")
    plt.savefig(FIG_DIR / "fig3_tiled_attention.png")
    plt.close()
    print("Generated Fig 3: Tiled Attention Scaling")

# -----------------------------------------------------------------------------
# Figure 4: Task 5 Speculative Decoding Throughput & Acceptance
# -----------------------------------------------------------------------------
def plot_speculative_decoding():
    t5_json = RESULTS_DIR / "task5_results.json"
    if not t5_json.exists():
        return
    with open(t5_json) as f:
        t5_data = json.load(f)

    configs = t5_data.get("configs", [])
    labels = [f"L={c['prompt_len']}\nGen={c['gen_len']}" for c in configs]
    g_tok = [c["greedy"]["tokens_s_mean"] for c in configs]
    s_tok = [c["speculative"]["tokens_s_mean"] for c in configs]
    acc = [c["speculative"]["acceptance_rate_mean"] * 100 for c in configs]
    speedup = [s / g for s, g in zip(s_tok, g_tok)]

    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax1.bar(x - width/2, g_tok, width, label="Target Greedy (1.5B LoRA)", color="#64748b")
    rects2 = ax1.bar(x + width/2, s_tok, width, label="Speculative (Draft 0.5B, γ=4)", color="#3b82f6")

    ax1.set_ylabel("Throughput (Tokens / Second)")
    ax1.set_title("Task 5: Speculative vs Greedy Decoding Throughput & Speedup")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Secondary axis for speedup line
    ax2 = ax1.twinx()
    ax2.plot(x, speedup, color="#ef4444", marker="s", linewidth=2, label="Speedup Factor (x)")
    ax2.set_ylabel("Speedup Ratio (Speculative / Greedy)", color="#ef4444")
    ax2.tick_params(axis="y", labelcolor="#ef4444")
    ax2.set_ylim(0.5, 1.6)
    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.7)

    for i, (sp, ac) in enumerate(zip(speedup, acc)):
        ax2.annotate(f"{sp:.2f}x\n(α={ac:.1f}%)", (x[i], sp), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, fontweight="bold", color="#b91c1c")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_speculative_decoding.pdf")
    plt.savefig(FIG_DIR / "fig4_speculative_decoding.png")
    plt.close()
    print("Generated Fig 4: Speculative Decoding Performance")

# -----------------------------------------------------------------------------
# Figure 5: Task 3 KV-Cache Throughput Scaling vs Uncached
# -----------------------------------------------------------------------------
def plot_kv_cache():
    kv_json = RESULTS_DIR / "kv_benchmark.json"
    if not kv_json.exists():
        return
    with open(kv_json) as f:
        kv_data = json.load(f)

    configs = kv_data.get("configs", [])
    # Group by prompt length for gen_len = 128
    long_gen = [c for c in configs if c.get("gen_len") == 128]
    if not long_gen:
        long_gen = configs[:3]

    labels = [f"Prompt={c['prompt_len']}\n(Gen=128)" for c in long_gen]
    cached_tok = [c["cached"]["tokens_s_mean"] for c in long_gen]
    uncached_tok = [c["uncached"]["tokens_s_mean"] for c in long_gen]
    speedups = [c / u for c, u in zip(cached_tok, uncached_tok)]

    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width/2, uncached_tok, width, label="Uncached Reference", color="#94a3b8")
    ax1.bar(x + width/2, cached_tok, width, label="KV-Cached (ExplicitKVCache)", color="#10b981")

    ax1.set_ylabel("Decoding Throughput (Tokens / Second)")
    ax1.set_title("Task 3: KV-Cached vs. Uncached Decoding Scaling (Qwen2.5-1.5B LoRA)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, speedups, color="#b91c1c", marker="o", linewidth=2, label="KV Speedup Factor")
    ax2.set_ylabel("Speedup Ratio (Cached / Uncached)", color="#b91c1c")
    ax2.tick_params(axis="y", labelcolor="#b91c1c")
    ax2.set_ylim(0, 8.0)

    for i, sp in enumerate(speedups):
        ax2.annotate(f"{sp:.2f}x", (x[i], sp), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color="#b91c1c")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_kv_cache.pdf")
    plt.savefig(FIG_DIR / "fig5_kv_cache.png")
    plt.close()
    print("Generated Fig 5: KV-Cache Scaling")

if __name__ == "__main__":
    plot_training_curve()
    plot_confusion_matrices()
    plot_tiled_attention()
    plot_speculative_decoding()
    plot_kv_cache()
    print("All publication figures generated successfully in report/figures/")
