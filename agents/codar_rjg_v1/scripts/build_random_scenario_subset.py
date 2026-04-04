#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build random scenario subset from full dataset.")
    p.add_argument("--input-json", required=True, help="Full dataset JSON (list).")
    p.add_argument("--output-json", required=True, help="Output subset JSON.")
    p.add_argument("--scenario", required=True, choices=["affection", "attitude", "intent"], help="Scenario filter.")
    p.add_argument("--size", type=int, default=133, help="Subset size.")
    p.add_argument("--seed", type=int, help="Random seed (default: current unix time).")
    p.add_argument("--report-json", help="Optional report path (default: output_json + .report.json).")
    return p.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output_json).resolve()
    report_path = Path(args.report_json).resolve() if args.report_json else output_path.with_suffix(".report.json")
    scenario = str(args.scenario).strip().lower()
    size = int(args.size)
    if size <= 0:
        raise ValueError("--size must be > 0")

    all_rows = _load_json(input_path)
    if not isinstance(all_rows, list):
        raise ValueError("input dataset root must be list")

    scenario_rows: List[Dict] = []
    for x in all_rows:
        sc = str(x.get("scenario", "")).strip().lower()
        if not sc:
            sc = str((x.get("input", {}) or {}).get("scenario", "")).strip().lower()
        if not sc:
            sc = str((x.get("meta", {}) or {}).get("scenario", "")).strip().lower()
        if sc == scenario:
            scenario_rows.append(x)
    if len(scenario_rows) < size:
        raise RuntimeError(
            f"scenario '{scenario}' only has {len(scenario_rows)} rows, smaller than requested size={size}"
        )

    seed = int(args.seed) if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    picked = rng.sample(scenario_rows, size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(picked, ensure_ascii=False, indent=2), encoding="utf-8")

    picked_ids = [str(x.get("id", "")).strip() for x in picked if str(x.get("id", "")).strip()]
    report = {
        "input_json": str(input_path),
        "output_json": str(output_path),
        "scenario": scenario,
        "requested_size": size,
        "final_size": len(picked),
        "pool_size": len(scenario_rows),
        "seed": seed,
        "sample_ids": picked_ids,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
