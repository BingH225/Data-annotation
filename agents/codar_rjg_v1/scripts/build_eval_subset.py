#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build deterministic eval subset with optional baseline-id preference.")
    p.add_argument("--input-json", required=True, help="Full dataset JSON list.")
    p.add_argument("--output-json", required=True, help="Subset output JSON.")
    p.add_argument("--size", type=int, default=267, help="Target subset size.")
    p.add_argument("--baseline-detailed-json", help="Optional baseline detailed file containing preferred ids.")
    p.add_argument("--report-json", help="Optional report output.")
    return p.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output_json).resolve()
    report_path = Path(args.report_json).resolve() if args.report_json else output_path.with_suffix(".report.json")
    size = int(args.size)
    if size <= 0:
        raise ValueError("--size must be > 0")

    full_data = _load_json(input_path)
    if not isinstance(full_data, list):
        raise ValueError("input dataset root must be list")
    id_to_row: Dict[str, dict] = {}
    for row in full_data:
        sid = str(row.get("id", "")).strip()
        if sid:
            id_to_row[sid] = row

    baseline_ids: List[str] = []
    if args.baseline_detailed_json:
        baseline_path = Path(args.baseline_detailed_json).resolve()
        baseline = _load_json(baseline_path)
        if isinstance(baseline, list):
            baseline_ids = [str(x.get("id", "")).strip() for x in baseline if str(x.get("id", "")).strip()]

    selected: List[dict] = []
    selected_ids = set()
    overlap = 0

    # Prefer baseline IDs when present in current input.
    for sid in baseline_ids:
        if sid in id_to_row and sid not in selected_ids:
            selected.append(id_to_row[sid])
            selected_ids.add(sid)
            overlap += 1
            if len(selected) >= size:
                break

    # Backfill deterministically by original input order.
    if len(selected) < size:
        for row in full_data:
            sid = str(row.get("id", "")).strip()
            if not sid or sid in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(sid)
            if len(selected) >= size:
                break

    if len(selected) < size:
        raise RuntimeError(f"Unable to build subset with requested size={size}; only got {len(selected)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "input_json": str(input_path),
        "output_json": str(output_path),
        "target_size": size,
        "final_size": len(selected),
        "baseline_detailed_json": str(Path(args.baseline_detailed_json).resolve()) if args.baseline_detailed_json else None,
        "preferred_baseline_id_count": len(baseline_ids),
        "baseline_overlap_used": overlap,
        "backfilled_count": len(selected) - overlap,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
