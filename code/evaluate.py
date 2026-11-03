"""Task 1/B4 + Task 2 metrics: greedy evaluation of base vs fine-tuned model.

For a fixed split (default: validation) runs:
  (a) the base model zero-shot   (BEFORE)
  (b) the base model + LoRA adapter (AFTER)
on the identical rows, with identical decoding settings (greedy,
max_new_tokens=64), and reports:
  - label: accuracy, macro-F1, per-class P/R/F1, confusion matrix
           (overall, inclusion-only, exclusion-only)
  - evidence sentence IDs: per-row set P/R/F1 (both-empty -> 1.0, documented),
           micro P/R/F1 pooled over rows (both-empty rows excluded),
           exact-match accuracy
  - JSON parse failure rate, invalid-label rate

Outputs:
  results/preds_<tag>_<split>.jsonl   per-row predictions (both tags)
  results/metrics_<tag>_<split>.json  metrics dict
  results/before_after_<split>.json   disagreement summary (with --compare)

Run:  python code/evaluate.py --split val --compare adapters/qwen25-1.5b-lora
Test: python code/evaluate.py --split test --allow-test   (final eval only!)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import (  # noqa: E402
    LABELS_BY_TYPE,
    build_messages,
    parse_gold_sentences,
    parse_model_output,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_causal_lm(name: str, dtype):
    """torch_dtype (v4) vs dtype (v5) API rename handling."""
    import transformers
    from transformers import AutoModelForCausalLM

    kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    return AutoModelForCausalLM.from_pretrained(name, **{kw: dtype})


# ---------------------------------------------------------------- metrics ---
def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "support": tp + fn}


def label_metrics(golds: list[str], preds: list[str], labels: list[str]) -> dict:
    """accuracy + macro-F1 + per-class P/R/F1 + confusion matrix.

    Predictions outside the known label set (e.g. unparseable output) are
    routed to an explicit '<invalid>' confusion-matrix bucket, never silently
    mapped onto a real class.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    cm_labels = list(labels) + ["<invalid>"]
    valid = set(labels)
    preds_safe = [p if p in valid else "<invalid>" for p in preds]
    n = len(golds)
    correct = sum(g == p for g, p in zip(golds, preds_safe))
    per_class = {}
    for l in labels:
        tp = sum(1 for g, p in zip(golds, preds_safe) if g == l and p == l)
        fp = sum(1 for g, p in zip(golds, preds_safe) if g != l and p == l)
        fn = sum(1 for g, p in zip(golds, preds_safe) if g == l and p != l)
        per_class[l] = prf(tp, fp, fn)
    return {
        "n": n,
        "accuracy": round(accuracy_score(golds, preds_safe), 4) if n else 0.0,
        "macro_f1": round(f1_score(golds, preds_safe, average="macro",
                                   labels=labels, zero_division=0), 4),
        "per_class": per_class,
        "invalid_predictions": int(sum(1 for p in preds_safe if p == "<invalid>")),
        "confusion_matrix": {
            "labels": cm_labels,
            "rows_gold_cols_pred": confusion_matrix(golds, preds_safe,
                                                    labels=cm_labels).tolist(),
        },
    }


def evidence_metrics(gold_sets: list[set[int]], pred_sets: list[set[int]]) -> dict:
    """Sentence-ID retrieval. Documented conventions:
    - per-row: empty gold AND empty pred -> row scores 1.0 (both agree on 'no evidence');
               empty gold, non-empty pred -> 0.0 (hallucinated evidence)
    - micro: TP/FP/FN pooled; rows with both-empty are excluded (contribute 0)
    """
    row_f1s, row_ps, row_rs = [], [], []
    tp = fp = fn = 0
    exact = 0
    for g, p in zip(gold_sets, pred_sets):
        if g == p:
            exact += 1
        if not g and not p:
            row_ps.append(1.0); row_rs.append(1.0); row_f1s.append(1.0)
            continue
        t = len(g & p)
        pr = t / len(p) if p else 0.0
        rc = t / len(g) if g else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        row_ps.append(pr); row_rs.append(rc); row_f1s.append(f)
        tp += t
        fp += len(p - g)
        fn += len(g - p)
    return {
        "n_rows": len(gold_sets),
        "per_row_mean": {
            "precision": round(float(np.mean(row_ps)), 4) if row_ps else 0.0,
            "recall": round(float(np.mean(row_rs)), 4) if row_rs else 0.0,
            "f1": round(float(np.mean(row_f1s)), 4) if row_f1s else 0.0,
        },
        "micro": prf(tp, fp, fn),
        "exact_match_rate": round(exact / len(gold_sets), 4) if gold_sets else 0.0,
        "empty_gold_rows": int(sum(1 for g in gold_sets if not g)),
        "empty_pred_rows": int(sum(1 for p in pred_sets if not p)),
        "both_empty_rows": int(sum(1 for g, p in zip(gold_sets, pred_sets) if not g and not p)),
    }


def full_report(rows: list[dict]) -> dict:
    golds = [r["gold_label"] for r in rows]
    preds = [r["pred_label"] for r in rows]
    labels = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]
    incl_idx = [i for i, r in enumerate(rows) if r["criterion_type"] == "inclusion"]
    excl_idx = [i for i, r in enumerate(rows) if r["criterion_type"] == "exclusion"]

    def subset(idxs: list[int]) -> dict:
        return label_metrics([golds[i] for i in idxs], [preds[i] for i in idxs], labels)

    parsed = [r for r in rows if r["parse_ok"]]
    return {
        "label": {
            "overall": label_metrics(golds, preds, labels),
            "inclusion": subset(incl_idx),
            "exclusion": subset(excl_idx),
        },
        "evidence": {
            "overall": evidence_metrics(
                [set(r["gold_evidence"]) for r in rows],
                [set(r["pred_evidence"]) for r in rows],
            ),
            "inclusion": evidence_metrics(
                [set(rows[i]["gold_evidence"]) for i in incl_idx],
                [set(rows[i]["pred_evidence"]) for i in incl_idx],
            ),
            "exclusion": evidence_metrics(
                [set(rows[i]["gold_evidence"]) for i in excl_idx],
                [set(rows[i]["pred_evidence"]) for i in excl_idx],
            ),
        },
        "generation": {
            "n": len(rows),
            "parse_ok": len(parsed),
            "parse_fail_rate": round(1 - len(parsed) / len(rows), 4) if rows else 0.0,
            "invalid_label_count": sum(1 for r in rows if not r["parse_ok"] and r["raw_text"].strip()),
        },
    }


# ------------------------------------------------------------- generation ---
def generate(model, tokenizer, rows: list[dict], device: str,
             batch_size: int = 8, max_new_tokens: int = 64) -> list[str]:
    import torch

    texts = []
    for r in rows:
        msgs = build_messages(r["note"], r["criterion_type"], r["criterion_text"])
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True))
    outputs: list[str] = []
    tokenizer.padding_side = "left"  # decoder-only batched generation
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=2048).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        if i % (batch_size * 8) == 0:
            print(f"  [gen] {i+len(batch)}/{len(texts)}")
    return outputs


def run_one(model, tokenizer, rows: list[dict], device: str, tag: str,
            batch_size: int, max_new_tokens: int) -> list[dict]:
    t0 = time.time()
    raws = generate(model, tokenizer, rows, device, batch_size, max_new_tokens)
    elapsed = time.time() - t0
    out = []
    for r, raw in zip(rows, raws):
        parsed = parse_model_output(raw, r["criterion_type"])
        out.append({
            "annotation_id": r.get("annotation_id"),
            "patient_id": r["patient_id"],
            "trial_id": r["trial_id"],
            "criterion_type": r["criterion_type"],
            "criterion_text": r["criterion_text"],
            "gold_label": r["expert_eligibility"],
            "gold_evidence": parse_gold_sentences(r["expert_sentences"]),
            "pred_label": parsed.label or "<parse_fail>",
            "pred_evidence": parsed.evidence,
            "parse_ok": parsed.parse_ok,
            "raw_text": raw.strip(),
            "tag": tag,
        })
    print(f"[{tag}] {len(rows)} rows in {elapsed:.1f}s "
          f"({len(rows)/max(elapsed,1e-9):.2f} rows/s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--compare", default="",
                    help="path to LoRA adapter; enables BEFORE/AFTER on identical rows")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-test", action="store_true",
                    help="required to evaluate on the held-out test split")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    if args.split == "test" and not args.allow_test:
        sys.exit("Refusing to evaluate on test before final evaluation "
                 "(pass --allow-test to override).")

    import torch
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in
            open(ROOT / "data" / "splits" / f"{args.split}.jsonl", encoding="utf-8")
            if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"[env] device={device} split={args.split} n={len(rows)}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tags: dict[str, list[dict]] = {}

    # BEFORE: base model, zero-shot
    base = load_causal_lm(args.base_model, dtype).to(device)
    base.eval()
    tags["before_base"] = run_one(base, tok, rows, device, "before_base",
                                  args.batch_size, args.max_new_tokens)
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    # AFTER: base + LoRA adapter (merged weights)
    if args.compare:
        from peft import PeftModel

        tuned = load_causal_lm(args.base_model, dtype)
        tuned = PeftModel.from_pretrained(tuned, args.compare).to(device)
        tuned.eval()
        tags["after_lora"] = run_one(tuned, tok, rows, device, "after_lora",
                                     args.batch_size, args.max_new_tokens)
        del tuned

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    for tag, preds in tags.items():
        path = results_dir / f"preds_{tag}_{args.split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for p in preds:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        metrics = full_report(preds)
        mpath = results_dir / f"metrics_{tag}_{args.split}.json"
        mpath.write_text(json.dumps(metrics, indent=2))
        lab = metrics["label"]
        print(f"[{tag}] acc={lab['overall']['accuracy']} "
              f"macroF1={lab['overall']['macro_f1']} "
              f"evF1={metrics['evidence']['overall']['micro']['f1']} "
              f"parse_fail={metrics['generation']['parse_fail_rate']}")

    if len(tags) == 2:
        before = {p["annotation_id"]: p for p in tags["before_base"]}
        after = {p["annotation_id"]: p for p in tags["after_lora"]}
        diffs = []
        flipped_ok, flipped_bad = 0, 0
        for aid in before:
            b, a = before[aid], after[aid]
            if b["pred_label"] != a["pred_label"]:
                fixed = a["pred_label"] == a["gold_label"]
                broken = b["pred_label"] == b["gold_label"]
                flipped_ok += int(fixed and not broken)
                flipped_bad += int(broken and not fixed)
                diffs.append({
                    "annotation_id": aid,
                    "gold": b["gold_label"],
                    "before": b["pred_label"],
                    "after": a["pred_label"],
                    "before_correct": b["pred_label"] == b["gold_label"],
                    "after_correct": a["pred_label"] == a["gold_label"],
                    "criterion_type": b["criterion_type"],
                    "criterion_text": b["criterion_text"],
                })
        summary = {
            "split": args.split,
            "n_changed_labels": len(diffs),
            "fixed_by_finetune": flipped_ok,
            "broken_by_finetune": flipped_bad,
            "before_acc": tags["before_base"] and full_report(tags["before_base"])["label"]["overall"]["accuracy"],
            "after_acc": full_report(tags["after_lora"])["label"]["overall"]["accuracy"],
            "changes": diffs,
        }
        (results_dir / f"before_after_{args.split}.json").write_text(
            json.dumps(summary, indent=2))
        print(f"[diff] {len(diffs)} label changes "
              f"({flipped_ok} fixed / {flipped_bad} broken)")


if __name__ == "__main__":
    main()
