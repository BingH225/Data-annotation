#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path

# =========================
# CONFIG (use your paths)
# =========================
INPUT_POOL_JSON = Path(r"C:\Users\xwhhh\Desktop\data_deal\Data\clean.intent_relabel_new1.json")
OUT_ROOT = Path(r"C:\Users\xwhhh\Desktop\data_deal\Task")
SEED = 42
ENCODING = "utf-8"

PER_SITUATION_QUOTA = {"attitude": 100, "intent": 100, "affection": 100}

# Name mapping: A->You, B->A, C->Zuo, D->Tan, E->Yu, F->Wu
PEOPLE = ["You", "A", "Zuo", "Tan", "Yu", "Wu"]

# Primary schedule:
# D1: You, A, Zuo, Tan, Yu, Wu (all primary)
# D2: A, Zuo, Tan, Yu, Wu (primary)
# D3: A, Zuo, Tan, Yu, Wu (primary)
# D4: A, Zuo, Tan, Yu, Wu (primary)
PRIMARY_WORKERS_BY_DAY = {
    1: ["You", "A", "Zuo", "Tan", "Yu", "Wu"],
    2: ["A", "Zuo", "Tan", "Yu", "Wu"],
    3: ["A", "Zuo", "Tan", "Yu", "Wu"],
    4: ["A", "Zuo", "Tan", "Yu", "Wu"],
}

# Leader review schedule (You is leader):
# D2: review A, Zuo (from D1 primary_done exports)
# D3: review Tan, Yu (from D2 primary_done exports)
# D4: review Wu, A   (from D3 primary_done exports)
QA_MAP = {
    2: (1, ["A", "Zuo"]),
    3: (2, ["Tan", "Yu"]),
    4: (3, ["Wu", "A"]),
}


def mkdirs(root: Path):
    (root / "00_source").mkdir(parents=True, exist_ok=True)
    (root / "01_plan").mkdir(parents=True, exist_ok=True)
    (root / "02_primary_assignments").mkdir(parents=True, exist_ok=True)
    (root / "03_review_tasks").mkdir(parents=True, exist_ok=True)
    (root / "04_outputs" / "primary_done").mkdir(parents=True, exist_ok=True)
    (root / "04_outputs" / "review_done").mkdir(parents=True, exist_ok=True)
    (root / "04_outputs" / "adjudication").mkdir(parents=True, exist_ok=True)
    (root / "05_logs" / "daily_stats").mkdir(parents=True, exist_ok=True)

    for d in [1, 2, 3, 4]:
        (root / "02_primary_assignments" / f"day{d:02d}").mkdir(parents=True, exist_ok=True)
        (root / "04_outputs" / "primary_done" / f"day{d:02d}").mkdir(parents=True, exist_ok=True)
    for d in [2, 3, 4]:
        (root / "03_review_tasks" / f"day{d:02d}").mkdir(parents=True, exist_ok=True)
        (root / "04_outputs" / "review_done" / f"day{d:02d}").mkdir(parents=True, exist_ok=True)


def normalize_situation(x) -> str:
    if not isinstance(x, str):
        return ""
    return x.strip().lower()


def load_pool(path: Path):
    with open(path, "r", encoding=ENCODING) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input pool must be a JSON list, got {type(data)}")
    return data


def bucket_by_situation(pool):
    buckets = {"attitude": [], "intent": [], "affection": []}
    for item in pool:
        if not isinstance(item, dict):
            continue
        out = item.get("output", {})
        if not isinstance(out, dict):
            continue
        s = normalize_situation(out.get("situation"))
        if s in buckets:
            buckets[s].append(item)
    return buckets


def take_items(buckets, quota):
    taken = []
    for s, n in quota.items():
        if len(buckets[s]) < n:
            raise RuntimeError(f"Not enough '{s}' items: need {n}, have {len(buckets[s])}")
        for _ in range(n):
            taken.append(buckets[s].pop())
    return taken


def add_meta(items, assigned_to: str, day: int, role: str):
    for it in items:
        if not isinstance(it, dict):
            continue
        meta = it.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta.update({
            "assigned_to": assigned_to,
            "day": day,
            "role": role,
            "source_pool": str(INPUT_POOL_JSON),
        })
        it["meta"] = meta


def write_json(path: Path, obj):
    with open(path, "w", encoding=ENCODING) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.write_text(text, encoding=ENCODING)


def build_plan_files(root: Path):
    schedule_lines = [
        "# Schedule (Leader: You)",
        "",
        "- D1: You,A,Zuo,Tan,Yu,Wu primary",
        "- D2: A,Zuo,Tan,Yu,Wu primary; You reviews A & Zuo (from D1 exports)",
        "- D3: A,Zuo,Tan,Yu,Wu primary; You reviews Tan & Yu (from D2 exports)",
        "- D4: A,Zuo,Tan,Yu,Wu primary; You reviews Wu & A (from D3 exports)",
        "",
        "Primary quota per person-day: attitude=100, intent=100, affection=100 (total=300).",
        "Review tasks are placeholder empty JSON files named: You_review_<Target>_dayXX.json",
    ]
    write_text(root / "01_plan" / "schedule.md", "\n".join(schedule_lines) + "\n")

    quotas = {
        "primary_per_person_day": PER_SITUATION_QUOTA,
        "seed": SEED,
        "encoding": ENCODING,
        "note": "Review files are empty JSON arrays [] placeholders.",
    }
    write_json(root / "01_plan" / "quotas.json", quotas)

    qa_lines = ["QA_MAP (Leader: You)"]
    for qa_day in sorted(QA_MAP.keys()):
        src_day, targets = QA_MAP[qa_day]
        qa_lines.append(f"day{qa_day:02d}: source=day{src_day:02d}, targets={','.join(targets)}")
    write_text(root / "01_plan" / "qa_map.txt", "\n".join(qa_lines) + "\n")


def build_primary_assignments(root: Path, buckets):
    created = []
    for day in [1, 2, 3, 4]:
        for person in PRIMARY_WORKERS_BY_DAY[day]:
            items = take_items(buckets, PER_SITUATION_QUOTA)
            add_meta(items, assigned_to=person, day=day, role="primary")
            out_path = root / "02_primary_assignments" / f"day{day:02d}" / f"{person}_primary_day{day:02d}.json"
            write_json(out_path, items)
            created.append(out_path)
    return created


def build_review_placeholders(root: Path):
    created = []
    for qa_day in sorted(QA_MAP.keys()):
        _, targets = QA_MAP[qa_day]
        for t in targets:
            out_path = root / "03_review_tasks" / f"day{qa_day:02d}" / f"You_review_{t}_day{qa_day:02d}.json"
            write_json(out_path, [])  # empty JSON list placeholder
            created.append(out_path)
    return created


def main():
    random.seed(SEED)
    mkdirs(OUT_ROOT)

    write_text(
        OUT_ROOT / "00_source" / "pool_meta.txt",
        f"pool={INPUT_POOL_JSON}\nseed={SEED}\nencoding={ENCODING}\n"
    )

    build_plan_files(OUT_ROOT)

    pool = load_pool(INPUT_POOL_JSON)
    buckets = bucket_by_situation(pool)

    for k in buckets:
        random.shuffle(buckets[k])

    # Required counts
    primary_files_needed = sum(len(PRIMARY_WORKERS_BY_DAY[d]) for d in [1, 2, 3, 4])  # 21 files
    needed_per_situation = {s: primary_files_needed * PER_SITUATION_QUOTA[s] for s in PER_SITUATION_QUOTA}  # 2100 each

    for s, n in needed_per_situation.items():
        have = len(buckets[s])
        if have < n:
            raise RuntimeError(
                f"Pool insufficient for situation '{s}': need {n}, have {have}. "
                f"Please increase pool or reduce quotas."
            )

    primary_paths = build_primary_assignments(OUT_ROOT, buckets)
    review_paths = build_review_placeholders(OUT_ROOT)

    # Summary
    print("=== GENERATED STRUCTURE ===")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print("")
    print(f"Primary assignment files: {len(primary_paths)}")
    print("  - Each file: 300 items (attitude=100, intent=100, affection=100)")
    print(f"  - Total unique primary items assigned: {len(primary_paths) * sum(PER_SITUATION_QUOTA.values())}")
    print("")
    print("Review placeholder files (empty JSON arrays):")
    for p in review_paths:
        print(f"  - {p}")
    print("")
    print("Plan files:")
    print(f"  - {OUT_ROOT / '01_plan' / 'schedule.md'}")
    print(f"  - {OUT_ROOT / '01_plan' / 'quotas.json'}")
    print(f"  - {OUT_ROOT / '01_plan' / 'qa_map.txt'}")
    print("")
    print("[OK] Done.")


if __name__ == "__main__":
    main()