#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List


def normalize(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def sample_scenario(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""

    if isinstance(item.get("output"), dict):
        out = item["output"]
        scenario = out.get("scenario", out.get("situation", ""))
        if scenario:
            return normalize(scenario)

    inp = item.get("input")
    if isinstance(inp, dict):
        scenario = inp.get("scenario", "")
        if scenario:
            return normalize(scenario)

    gt = item.get("ground_truth")
    if isinstance(gt, dict):
        scenario = gt.get("scenario", gt.get("situation", ""))
        if scenario:
            return normalize(scenario)

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a random scenario subset from a JSON dataset")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--scenario", default="intent", help="Scenario name to keep")
    parser.add_argument("--count", type=int, default=133, help="Number of samples to draw")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: unix time)")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    scenario = normalize(args.scenario)
    count = max(1, int(args.count))

    payload = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        data: List[Dict[str, Any]] = payload["data"]
    elif isinstance(payload, list):
        data = payload
    else:
        raise ValueError("Unsupported JSON structure. Expected list or dict with key 'data'.")

    pool = [item for item in data if sample_scenario(item) == scenario]
    if not pool:
        raise ValueError(f"No samples found for scenario={scenario}")

    seed = int(args.seed) if args.seed is not None else int(time.time())
    rng = random.Random(seed)

    if len(pool) <= count:
        chosen = list(pool)
    else:
        chosen = rng.sample(pool, count)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "input": str(src),
        "output": str(dst),
        "scenario": scenario,
        "available": len(pool),
        "selected": len(chosen),
        "seed": seed,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
