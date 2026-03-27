from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from ..types import SampleInput


def load_samples(path: Path, scenario_filter: Optional[str] = None, limit: Optional[int] = None) -> List[SampleInput]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input json root must be list: {path}")
    out: List[SampleInput] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        inp = row.get("input", {}) or {}
        scenario = str(inp.get("scenario", "")).strip().lower()
        if scenario_filter and scenario != scenario_filter:
            continue
        options = row.get("options", {}) or {}
        sample = SampleInput(
            sample_id=str(row.get("id", "")).strip(),
            scenario=scenario,
            text=str(inp.get("text", "")),
            media=inp.get("media", {}) or {},
            subject_options=[str(x) for x in (options.get("subject", []) or [])],
            target_options=[str(x) for x in (options.get("target", []) or [])],
            diversity=row.get("diversity", {}) or {},
            ground_truth=row.get("ground_truth", {}) or {},
        )
        if not sample.sample_id:
            continue
        out.append(sample)
        if limit is not None and len(out) >= limit:
            break
    return out


def iter_samples(path: Path, scenario_filter: Optional[str] = None, limit: Optional[int] = None) -> Iterable[SampleInput]:
    return load_samples(path=path, scenario_filter=scenario_filter, limit=limit)

