from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..types import SampleInput
from .anchors import build_anchor_payload


@dataclass
class MemoryIndex:
    entries: List[Dict[str, Any]]
    idf_by_scenario: Dict[str, Dict[str, float]]
    avg_len_by_scenario: Dict[str, float]
    meta: Dict[str, Any]


def _idf(doc_freq: int, num_docs: int) -> float:
    return math.log((num_docs + 1.0) / (doc_freq + 1.0)) + 1.0


def _bm25(
    q_tf: Dict[str, int],
    d_tf: Dict[str, int],
    doc_len: int,
    avg_doc_len: float,
    idf_map: Dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not q_tf or not d_tf:
        return 0.0
    denom_norm = max(avg_doc_len, 1.0)
    score = 0.0
    for term, q_weight in q_tf.items():
        tf = float(d_tf.get(term, 0))
        if tf <= 0:
            continue
        idf = float(idf_map.get(term, 0.0))
        denom = tf + k1 * (1.0 - b + b * (float(doc_len) / denom_norm))
        score += idf * ((tf * (k1 + 1.0)) / max(denom, 1e-9)) * float(q_weight)
    return float(score)


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union) if union else 0.0


def build_memory_index(samples: List[SampleInput], index_name: str = "rjg_v1_unlabeled") -> MemoryIndex:
    entries: List[Dict[str, Any]] = []
    df_by_scenario: Dict[str, Dict[str, int]] = {}
    total_len_by_scenario: Dict[str, int] = {}
    count_by_scenario: Dict[str, int] = {}

    for sample in samples:
        media_manifest = {
            "audio_caption": str((sample.media or {}).get("audio_caption", "") or ""),
        }
        anchors = build_anchor_payload(sample, media_manifest=media_manifest)
        tf = anchors.get("token_freq", {}) or {}
        tokens = list(tf.keys())
        scenario = str(sample.scenario or "").strip().lower()
        if not scenario:
            continue
        entry = {
            "sample_id": sample.sample_id,
            "scenario": scenario,
            "text": sample.text,
            "audio_caption": media_manifest.get("audio_caption", ""),
            "subject_options": list(sample.subject_options),
            "target_options": list(sample.target_options),
            "token_freq": tf,
            "tokens": tokens,
            "doc_len": int(sum(int(v) for v in tf.values())),
            "anchor_keywords": anchors.get("keyword_tokens", []) or [],
        }
        entries.append(entry)
        count_by_scenario[scenario] = count_by_scenario.get(scenario, 0) + 1
        total_len_by_scenario[scenario] = total_len_by_scenario.get(scenario, 0) + int(entry["doc_len"])
        doc_df = df_by_scenario.setdefault(scenario, {})
        for t in set(tokens):
            doc_df[t] = int(doc_df.get(t, 0)) + 1

    idf_by_scenario: Dict[str, Dict[str, float]] = {}
    avg_len_by_scenario: Dict[str, float] = {}
    for scenario, df_map in df_by_scenario.items():
        n = int(count_by_scenario.get(scenario, 0))
        avg_len_by_scenario[scenario] = float(total_len_by_scenario.get(scenario, 0) / max(n, 1))
        idf_by_scenario[scenario] = {t: _idf(df, n) for t, df in df_map.items()}

    return MemoryIndex(
        entries=entries,
        idf_by_scenario=idf_by_scenario,
        avg_len_by_scenario=avg_len_by_scenario,
        meta={"index_name": index_name, "entry_count": len(entries)},
    )


def save_memory_index(index: MemoryIndex, out_path: Path) -> None:
    payload = {
        "meta": index.meta,
        "entries": index.entries,
        "idf_by_scenario": index.idf_by_scenario,
        "avg_len_by_scenario": index.avg_len_by_scenario,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_memory_index(path: Path) -> MemoryIndex:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"memory index root must be object: {path}")
    entries = data.get("entries", []) or []
    idf_by_scenario = data.get("idf_by_scenario", {}) or {}
    avg_len_by_scenario = data.get("avg_len_by_scenario", {}) or {}
    meta = data.get("meta", {}) or {}
    return MemoryIndex(entries=entries, idf_by_scenario=idf_by_scenario, avg_len_by_scenario=avg_len_by_scenario, meta=meta)


def _norm_minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def retrieve_similar_entries(
    index: MemoryIndex,
    scenario: str,
    query_tf: Dict[str, int],
    query_subject_options: List[str],
    query_target_options: List[str],
    query_sample_id: str,
    top_k: int = 40,
    rerank_k: int = 12,
    loo: bool = True,
) -> List[Dict[str, Any]]:
    scenario = str(scenario or "").strip().lower()
    idf_map = index.idf_by_scenario.get(scenario, {}) or {}
    avg_len = float(index.avg_len_by_scenario.get(scenario, 1.0) or 1.0)
    pre: List[Tuple[float, float, Dict[str, Any]]] = []
    query_tokens = list((query_tf or {}).keys())

    for entry in index.entries:
        if str(entry.get("scenario", "")).strip().lower() != scenario:
            continue
        if loo and str(entry.get("sample_id", "")) == str(query_sample_id):
            continue
        d_tf = entry.get("token_freq", {}) or {}
        bm = _bm25(query_tf, d_tf, int(entry.get("doc_len", 0)), avg_len, idf_map)
        jac = _jaccard(query_tokens, entry.get("tokens", []) or [])
        pre.append((bm, jac, entry))

    if not pre:
        return []
    pre.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = pre[: max(top_k, rerank_k)]
    bm_norm = _norm_minmax([x[0] for x in top])
    jac_norm = _norm_minmax([x[1] for x in top])

    reranked: List[Dict[str, Any]] = []
    q_sub = set(str(x).strip().lower() for x in (query_subject_options or []) if str(x).strip())
    q_tgt = set(str(x).strip().lower() for x in (query_target_options or []) if str(x).strip())
    for i, (_, _, entry) in enumerate(top):
        e_sub = set(str(x).strip().lower() for x in (entry.get("subject_options", []) or []) if str(x).strip())
        e_tgt = set(str(x).strip().lower() for x in (entry.get("target_options", []) or []) if str(x).strip())
        opt_overlap = 0.0
        if q_sub:
            opt_overlap += len(q_sub & e_sub) / max(len(q_sub), 1)
        if q_tgt:
            opt_overlap += len(q_tgt & e_tgt) / max(len(q_tgt), 1)
        opt_overlap *= 0.5
        score = 0.65 * bm_norm[i] + 0.25 * jac_norm[i] + 0.10 * opt_overlap
        reranked.append(
            {
                "sample_id": entry.get("sample_id"),
                "text": entry.get("text", ""),
                "audio_caption": entry.get("audio_caption", ""),
                "subject_options": entry.get("subject_options", []) or [],
                "target_options": entry.get("target_options", []) or [],
                "anchor_keywords": entry.get("anchor_keywords", []) or [],
                "retrieval_score": float(score),
            }
        )
    reranked.sort(key=lambda x: float(x.get("retrieval_score", 0.0)), reverse=True)
    return reranked[:rerank_k]
