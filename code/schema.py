"""Shared prompt schema for the TrialGPT criterion-eligibility task (Task 1).

Input : (patient note with numbered sentences, criterion type, criterion text)
Output: strict JSON {"label": <criterion-type-specific string>,
                     "evidence": [<original sentence ints>]}

Label vocabulary is fixed to the dataset's expert categories; inclusion and
exclusion criteria use different positive/negative terms and must never be
collapsed to binary. GPT-4 prediction columns are never used as inputs.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

LABELS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "inclusion": ("included", "not included", "not enough information", "not applicable"),
    "exclusion": ("excluded", "not excluded", "not enough information", "not applicable"),
}

ALL_LABELS = tuple(
    sorted({lbl for labels in LABELS_BY_TYPE.values() for lbl in labels})
)

# Columns released by the dataset that are GPT-4 outputs / judgments on them.
# They must never be model inputs (assignment: no leakage).
FORBIDDEN_INPUT_COLUMNS: tuple[str, ...] = (
    "gpt4_explanation",
    "gpt4_sentences",
    "gpt4_eligibility",
    "explanation_correctness",
)

SYSTEM_PROMPT = (
    "You are a clinical research assistant that screens patient records against "
    "individual clinical trial criteria. You see a patient record whose sentences "
    "are numbered with their original IDs (e.g. '3. ...'). For the given criterion, "
    "decide the eligibility label and cite the original sentence IDs that support "
    "the decision. Answer with a single JSON object and nothing else, using the "
    'schema {"label": string, "evidence": [int, ...]}. '
    "Use only these labels for inclusion criteria: included, not included, "
    "not enough information, not applicable. "
    "Use only these labels for exclusion criteria: excluded, not excluded, "
    "not enough information, not applicable. "
    "Cite evidence sentence IDs exactly as numbered in the record. "
    "If the record does not contain the information needed, use "
    "'not enough information' and cite the most relevant sentences or an empty list. "
    "If the criterion cannot apply to this patient, use 'not applicable'."
)

_SENTENCE_RE = re.compile(r"(?m)^(\d+)\.\s+")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ParsedPrediction:
    label: str | None = None
    evidence: list[int] = field(default_factory=list)
    parse_ok: bool = False
    raw: str = ""


def split_note_sentences(note: str) -> list[tuple[int, str]]:
    """Parse 'N. sentence' lines from the dataset note into (id, text) pairs."""
    matches = list(_SENTENCE_RE.finditer(note))
    sentences: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        sid = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note)
        text = note[start:end].strip()
        sentences.append((sid, text))
    return sentences


def format_note(sentences: list[tuple[int, str]]) -> str:
    return "\n".join(f"{sid}. {text}" for sid, text in sentences)


def build_messages(
    note: str, criterion_type: str, criterion_text: str
) -> list[dict[str, str]]:
    """Chat messages for one (patient, criterion) example."""
    note_fmt = note if "0." in note[:8] else format_note(split_note_sentences(note))
    user = (
        f"Patient record:\n{note_fmt}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n\n"
        "Return the JSON decision."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_target(expert_eligibility: str, evidence_ids: list[int]) -> str:
    """Canonical target string: compact JSON, sorted evidence IDs."""
    obj = {"label": expert_eligibility, "evidence": sorted(int(i) for i in evidence_ids)}
    return json.dumps(obj, separators=(",", ":"))


def parse_model_output(text: str, criterion_type: str | None = None) -> ParsedPrediction:
    """Parse a model response into label + evidence IDs (lenient but validated)."""
    raw = text or ""
    result = ParsedPrediction(raw=raw)
    m = _JSON_RE.search(raw)
    if not m:
        return result
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return result
    if not isinstance(obj, dict):
        return result
    label = str(obj.get("label", "")).strip().lower()
    evidence_raw = obj.get("evidence", [])
    evidence: list[int] = []
    if isinstance(evidence_raw, list):
        for e in evidence_raw:
            try:
                evidence.append(int(e))
            except (TypeError, ValueError):
                continue
    valid_labels = LABELS_BY_TYPE.get(criterion_type, ALL_LABELS) if criterion_type else ALL_LABELS
    if label not in valid_labels:
        return result
    result.label = label
    result.evidence = sorted(set(evidence))
    result.parse_ok = True
    return result


def parse_gold_sentences(expert_sentences: str) -> list[int]:
    """Dataset stores physician evidence as a string repr of an int list."""
    if expert_sentences is None:
        return []
    s = str(expert_sentences).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(val, list):
        return []
    return sorted({int(x) for x in val})


def truncate_note_sentences(
    sentences: list[tuple[int, str]],
    tokenizer,
    max_prompt_tokens: int,
    reserve: int = 256,
) -> tuple[list[tuple[int, str]], list[int], bool]:
    """Truncation policy (documented in docs/truncation.md).

    Keep the longest *prefix* of numbered sentences (IDs stay original and
    contiguous from 0), plus the final synthetic consent sentence if it was
    dropped. Never renumber IDs. Returns (kept_sentences, dropped_ids,
    dropped_gold_matters) where the last flag is filled in by the caller
    against gold evidence.
    """
    if not sentences:
        return [], [], False
    body = sentences[:-1] if len(sentences) > 1 else sentences
    tail = sentences[-1] if len(sentences) > 1 else None

    def tok_len(sents: list[tuple[int, str]]) -> int:
        return len(tokenizer(format_note(sents), add_special_tokens=False)["input_ids"])

    # Fast path: everything fits.
    full = sentences if tail is None else body + [tail]
    if tok_len(full) <= max_prompt_tokens - reserve:
        return full, [], False

    kept: list[tuple[int, str]] = []
    for s in body:
        trial = kept + [s] + ([tail] if tail is not None else [])
        if tok_len(trial) > max_prompt_tokens - reserve:
            break
        kept.append(s)
    if tail is not None and tok_len(kept + [tail]) <= max_prompt_tokens - reserve:
        kept.append(tail)
    dropped_ids = [sid for sid, _ in sentences if sid not in {k[0] for k in kept}]
    return kept, dropped_ids, True
