"""Task 1/B1: dataset download, fixed patient-grouped split, manifest, stats.

Split protocol (mandated): sort unique patient IDs -> shuffle with
numpy.random.default_rng(42) -> first 37 patients train, next 8 validation,
remaining 8 test. Every annotation row of a patient stays in that patient's
split. GPT-4 prediction columns are dropped (never used as inputs/targets).

Outputs:
  data/split_manifest.json      patient IDs, counts, seed, protocol
  data/split_manifest.csv       one row per patient (id, split, n_rows, n_trials)
  data/splits/{train,val,test}.jsonl   cleaned rows (no GPT-4 columns)
  results/data_stats.json       label distribution, token-length stats
  results/truncation_report.json token lengths vs max_prompt_tokens, dropped gold

Run:  python code/prepare_data.py [--max-prompt-tokens 2048]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import (  # noqa: E402
    FORBIDDEN_INPUT_COLUMNS,
    LABELS_BY_TYPE,
    build_messages,
    build_target,
    parse_gold_sentences,
    split_note_sentences,
    truncate_note_sentences,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT / "results"
HF_DATASET = "ncbi/TrialGPT-Criterion-Annotations"
SEED = 42
N_TRAIN, N_VAL, N_TEST = 37, 8, 8

KEEP_COLUMNS = [
    "annotation_id",
    "patient_id",
    "note",
    "trial_id",
    "trial_title",
    "criterion_type",
    "criterion_text",
    "expert_sentences",
    "expert_eligibility",
    "training",
]


def load_raw() -> pd.DataFrame:
    """Load the mandatory dataset (datasets lib if present, else HF parquet)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(HF_DATASET, split="train")
        df = ds.to_pandas()
        print(f"[load] via datasets: {len(df)} rows")
        return df
    except ImportError:
        pass

    import httpx

    base = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": HF_DATASET,
        "config": "default",
        "split": "train",
        "offset": 0,
        "length": 100,
    }
    rows: list[dict] = []
    while True:
        r = httpx.get(base, params=params, timeout=60, follow_redirects=True)
        r.raise_for_status()
        payload = r.json()
        batch = [item["row"] for item in payload["rows"]]
        if not batch:
            break
        rows.extend(batch)
        params["offset"] += len(batch)
        if params["offset"] >= payload["num_rows_total"]:
            break
    df = pd.DataFrame(rows)
    print(f"[load] via HF rows API: {len(df)} rows")
    return df


def patient_split(patient_ids: list[str]) -> dict[str, list[str]]:
    """Mandated split: sorted IDs, default_rng(42) shuffle, 37/8/8."""
    ids = sorted(patient_ids)
    assert len(ids) == N_TRAIN + N_VAL + N_TEST, f"expected 53 patients, got {len(ids)}"
    rng = np.random.default_rng(SEED)
    order = np.array(ids, dtype=object)
    rng.shuffle(order)  # in-place shuffle of the sorted list
    return {
        "train": order[:N_TRAIN].tolist(),
        "val": order[N_TRAIN : N_TRAIN + N_VAL].tolist(),
        "test": order[N_TRAIN + N_VAL :].tolist(),
    }


def token_stats(split_dfs: dict[str, pd.DataFrame], max_prompt_tokens: int) -> tuple[dict, dict, "object"]:
    """Token length of every prompt/target under the Qwen tokenizer + truncation audit."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    stats: dict = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "max_prompt_tokens": max_prompt_tokens}
    trunc_report = {
        "max_prompt_tokens": max_prompt_tokens,
        "policy": "longest prefix of numbered sentences + final consent sentence; IDs never renumbered",
        "per_split": {},
    }

    for split, df in split_dfs.items():
        lens: list[int] = []
        target_lens: list[int] = []
        n_truncated = 0
        dropped_gold_rows = 0
        dropped_id_counter: Counter[int] = Counter()
        for row in df.itertuples(index=False):
            sentences = split_note_sentences(row.note)
            kept, dropped_ids, truncated = truncate_note_sentences(
                sentences, tokenizer, max_prompt_tokens
            )
            messages = build_messages(
                format_from_tuples(kept) if truncated else row.note,
                row.criterion_type,
                row.criterion_text,
            )
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            lens.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
            target = build_target(
                row.expert_eligibility, parse_gold_sentences(row.expert_sentences)
            )
            target_lens.append(len(tokenizer(target, add_special_tokens=False)["input_ids"]))
            if truncated:
                n_truncated += 1
                dropped_id_counter.update(dropped_ids)
                gold = parse_gold_sentences(row.expert_sentences)
                if any(g in dropped_ids for g in gold):
                    dropped_gold_rows += 1
        arr = np.array(lens)
        stats[split] = {
            "n_rows": len(lens),
            "prompt_tokens_min": int(arr.min()),
            "prompt_tokens_p50": int(np.percentile(arr, 50)),
            "prompt_tokens_p95": int(np.percentile(arr, 95)),
            "prompt_tokens_max": int(arr.max()),
            "target_tokens_mean": round(float(np.mean(target_lens)), 2),
            "target_tokens_max": int(np.max(target_lens)),
            "rows_over_limit_before_truncation": n_truncated,
            "rows_dropping_gold_evidence": dropped_gold_rows,
        }
        trunc_report["per_split"][split] = {
            "n_rows": len(lens),
            "n_truncated_rows": n_truncated,
            "dropped_sentence_id_counts": dict(dropped_id_counter),
            "rows_where_gold_evidence_was_dropped": dropped_gold_rows,
        }
    return stats, trunc_report, tokenizer


def format_from_tuples(sentences: list[tuple[int, str]]) -> str:
    return "\n".join(f"{sid}. {text}" for sid, text in sentences)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--skip-token-stats", action="store_true")
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    SPLITS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_raw()

    # --- integrity checks ------------------------------------------------
    assert len(df) == 1015, f"unexpected row count {len(df)}"
    assert df["patient_id"].nunique() == 53, "unexpected patient count"
    assert not df.duplicated(["patient_id", "trial_id", "criterion_type", "criterion_text"]).any()
    for c in FORBIDDEN_INPUT_COLUMNS:
        assert c in df.columns, f"expected dataset column {c}"
    df["criterion_type"] = df["criterion_type"].str.lower().str.strip()

    # --- split ------------------------------------------------------------
    splits = patient_split(df["patient_id"].unique().tolist())
    all_ids = [p for s in splits.values() for p in s]
    assert len(all_ids) == len(set(all_ids)) == 53, "patient leakage in split"

    split_of = {p: s for s, plist in splits.items() for p in plist}
    df["split"] = df["patient_id"].map(split_of)
    assert df["split"].notna().all()

    # --- manifest ---------------------------------------------------------
    rows = []
    for split, plist in splits.items():
        for pid in plist:
            sub = df[df["patient_id"] == pid]
            rows.append(
                {
                    "patient_id": pid,
                    "split": split,
                    "n_rows": int(len(sub)),
                    "n_trials": int(sub["trial_id"].nunique()),
                    "n_sentences": int(len(split_note_sentences(sub.iloc[0]["note"]))),
                }
            )
    manifest = {
        "dataset": HF_DATASET,
        "seed": SEED,
        "protocol": (
            "sort unique patient IDs; shuffle with numpy.random.default_rng(42); "
            "first 37 -> train, next 8 -> val, last 8 -> test"
        ),
        "n_patients": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST},
        "n_rows": {s: int((df["split"] == s).sum()) for s in splits},
        "n_trials": {s: int(df[df["split"] == s]["trial_id"].nunique()) for s in splits},
        "patients": {s: plist for s, plist in splits.items()},
        "forbidden_input_columns_dropped": list(FORBIDDEN_INPUT_COLUMNS),
        "leakage_check": "PASS: patient IDs disjoint across splits",
    }
    (DATA_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(rows).to_csv(DATA_DIR / "split_manifest.csv", index=False)

    # --- cleaned splits (GPT-4 columns removed) ---------------------------
    split_dfs: dict[str, pd.DataFrame] = {}
    for split, plist in splits.items():
        sub = df[df["split"] == split].copy()
        drop = [c for c in FORBIDDEN_INPUT_COLUMNS if c in sub.columns]
        sub = sub.drop(columns=drop).drop(columns=["split"], errors="ignore")
        sub = sub[KEEP_COLUMNS]
        sub.to_json(SPLITS_DIR / f"{split}.jsonl", orient="records", lines=True)
        split_dfs[split] = sub
        print(f"[split] {split}: {len(sub)} rows, {len(plist)} patients, "
              f"{sub['trial_id'].nunique()} trials")

    # --- label distribution ----------------------------------------------
    label_dist: dict = {}
    for split, sub in split_dfs.items():
        c = Counter(zip(sub["criterion_type"], sub["expert_eligibility"]))
        label_dist[split] = {
            f"{t}|{l}": int(n) for (t, l), n in sorted(c.items())
        }
    for t, labels in LABELS_BY_TYPE.items():
        for l in labels:
            for split in split_dfs:
                label_dist[split].setdefault(f"{t}|{l}", 0)

    # --- example sanity render -------------------------------------------
    ex = split_dfs["train"].iloc[0]
    example = {
        "messages": build_messages(ex["note"], ex["criterion_type"], ex["criterion_text"]),
        "target": build_target(ex["expert_eligibility"],
                               parse_gold_sentences(ex["expert_sentences"])),
    }

    out = {"label_distribution": label_dist, "n_forbidden_columns_dropped": len(FORBIDDEN_INPUT_COLUMNS)}
    if not args.skip_token_stats:
        stats, trunc, _ = token_stats(split_dfs, args.max_prompt_tokens)
        stats["max_prompt_tokens"] = args.max_prompt_tokens
        out["token_stats"] = stats
        (RESULTS_DIR / "truncation_report.json").write_text(json.dumps(trunc, indent=2))
    (RESULTS_DIR / "data_stats.json").write_text(json.dumps(out, indent=2))
    (RESULTS_DIR / "example_rendered.json").write_text(json.dumps(example, indent=2))

    print(json.dumps(manifest["n_rows"]))
    print("manifest:", DATA_DIR / "split_manifest.json")
    print("splits:", SPLITS_DIR)


if __name__ == "__main__":
    main()
