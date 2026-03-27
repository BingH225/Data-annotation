#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_SCENARIOS = ["affection", "attitude", "intent"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic fixed-size subset per scenario.")
    parser.add_argument("--input-json", required=True, help="Full dataset json file.")
    parser.add_argument("--output-json", required=True, help="Output subset json file.")
    parser.add_argument("--per-scenario", type=int, default=100, help="Sample count per scenario.")
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario order. Default: affection,attitude,intent",
    )
    parser.add_argument("--report-json", help="Optional report output path.")
    return parser.parse_args()


def _load_rows(path: Path) -> List[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input dataset root must be a list.")
    return rows


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output_json).resolve()
    report_path = Path(args.report_json).resolve() if args.report_json else output_path.with_suffix(".report.json")
    per_scenario = int(args.per_scenario)
    if per_scenario <= 0:
        raise ValueError("--per-scenario must be > 0")
    scenarios = [s.strip().lower() for s in str(args.scenarios).split(",") if s.strip()]
    if not scenarios:
        raise ValueError("No scenarios provided.")

    rows = _load_rows(input_path)
    selected_by_scenario: Dict[str, List[dict]] = {s: [] for s in scenarios}
    for row in rows:
        if not isinstance(row, dict):
            continue
        scenario = str((row.get("input", {}) or {}).get("scenario", "")).strip().lower()
        if scenario not in selected_by_scenario:
            continue
        cur = selected_by_scenario[scenario]
        if len(cur) < per_scenario:
            cur.append(row)
        if all(len(selected_by_scenario[s]) >= per_scenario for s in scenarios):
            break

    missing = {s: per_scenario - len(selected_by_scenario[s]) for s in scenarios if len(selected_by_scenario[s]) < per_scenario}
    if missing:
        raise RuntimeError(f"Insufficient samples for scenarios: {missing}")

    selected_rows: List[dict] = []
    for scenario in scenarios:
        selected_rows.extend(selected_by_scenario[scenario][:per_scenario])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "input_json": str(input_path),
        "output_json": str(output_path),
        "scenarios": scenarios,
        "per_scenario": per_scenario,
        "total": len(selected_rows),
        "counts": {s: len(selected_by_scenario[s]) for s in scenarios},
        "sample_ids": {
            s: [str((row.get("id", ""))).strip() for row in selected_by_scenario[s]]
            for s in scenarios
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
