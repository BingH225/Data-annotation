#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


FIELDS = [
    "subject",
    "target",
    "subject1",
    "subject2",
    "subject3",
    "target1",
    "target2",
    "target3",
]

PLACEHOLDER_PATTERNS = [
    r"replace\s+with",
    r"^use\b",
    r"\buse\s+[^\n]*\bas\b",
    r"^avoid\b",
    r"placeholder",
    r"operation\s+instruction",
    r"e\.g\.",
    r"for example",
    r"non-overlapping",
    r"distractor",
]

PLACEHOLDER_REGEX = [re.compile(p, re.IGNORECASE) for p in PLACEHOLDER_PATTERNS]


def read_csv_with_fallback(path: Path) -> Tuple[str, List[str], List[Dict[str, str]]]:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                headers = reader.fieldnames or []
                return encoding, headers, rows
        except UnicodeDecodeError:
            continue
    raise RuntimeError("cannot decode csv with utf-8-sig/utf-8/gbk")


def parse_requested_range(stem: str, total_rows: int) -> Tuple[int, int]:
    # Example: Tan_primary_day08_151_300 -> only review rows 151..300
    m = re.search(r"day\d+_(\d+)_(\d+)$", stem, flags=re.IGNORECASE)
    if not m:
        return 1, total_rows

    start = int(m.group(1))
    end = int(m.group(2))
    if start > end:
        start, end = end, start
    return start, end


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def check_row(row: Dict[str, str]) -> Tuple[bool, Dict[str, List[str]], Dict[str, str]]:
    field_reasons: Dict[str, List[str]] = defaultdict(list)
    field_values: Dict[str, str] = {}

    for field in FIELDS:
        value = str(row.get(field, "") or "").strip()
        field_values[field] = value

        if not value:
            field_reasons[field].append("empty")
        if "/" in value or any(ch in value for ch in "()（）"):
            field_reasons[field].append("forbidden_symbol")
        if any(p.search(value) for p in PLACEHOLDER_REGEX):
            field_reasons[field].append("placeholder")
        if word_count(value) >= 5:
            field_reasons[field].append("word_ge_5")

    has_issue = any(field_reasons.values())
    return has_issue, field_reasons, field_values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Review all CSV files in a directory with 4 rules: "
            "empty, forbidden symbol (/ or parentheses), placeholder text, word count >= 5."
        )
    )
    parser.add_argument(
        "--dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing CSV files. Default: current script directory.",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir).resolve()
    csv_files = sorted(base_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"no csv files found in: {base_dir}")

    report = {
        "review_time": datetime.now().isoformat(timespec="seconds"),
        "rules": [
            "empty",
            "forbidden_symbol(/ or parentheses)",
            "placeholder",
            "word_ge_5",
        ],
        "directory": str(base_dir),
        "file_count": len(csv_files),
        "files": [],
    }

    for csv_path in csv_files:
        file_result: Dict[str, object] = {
            "file_name": csv_path.name,
            "file_path": str(csv_path),
        }
        try:
            encoding, headers, rows = read_csv_with_fallback(csv_path)
            missing = [c for c in FIELDS if c not in headers]
            if missing:
                file_result["error"] = f"missing required columns: {missing}"
                report["files"].append(file_result)
                continue

            total_rows = len(rows)
            req_start, req_end = parse_requested_range(csv_path.stem, total_rows)
            eff_start = max(1, min(req_start, total_rows)) if total_rows else 0
            eff_end = max(0, min(req_end, total_rows))
            if eff_start > eff_end:
                checked_rows = []
            else:
                checked_rows = rows[eff_start - 1 : eff_end]

            rule_row_sets = {
                "empty": set(),
                "forbidden_symbol": set(),
                "placeholder": set(),
                "word_ge_5": set(),
            }
            issues = []

            for i, row in enumerate(checked_rows, start=eff_start):
                has_issue, field_reasons, field_values = check_row(row)
                if not has_issue:
                    continue

                fields_issue = []
                for field in FIELDS:
                    reasons = field_reasons.get(field, [])
                    if reasons:
                        fields_issue.append(
                            {
                                "field": field,
                                "reasons": reasons,
                                "value": field_values[field],
                            }
                        )
                        for r in reasons:
                            rule_row_sets[r].add(i)

                issues.append(
                    {
                        "file_row": i,
                        "id": str(row.get("id", "") or ""),
                        "field_issues": fields_issue,
                    }
                )

            file_result.update(
                {
                    "encoding": encoding,
                    "total_rows_in_file": total_rows,
                    "requested_range": [req_start, req_end],
                    "checked_range": [eff_start, eff_end],
                    "checked_rows": len(checked_rows),
                    "non_compliant_rows": len(issues),
                    "rule_row_counts": {
                        "empty": len(rule_row_sets["empty"]),
                        "forbidden_symbol": len(rule_row_sets["forbidden_symbol"]),
                        "placeholder": len(rule_row_sets["placeholder"]),
                        "word_ge_5": len(rule_row_sets["word_ge_5"]),
                    },
                    "issues": issues,
                }
            )
        except Exception as exc:  # noqa: BLE001
            file_result["error"] = str(exc)

        report["files"].append(file_result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = base_dir / f"review_4rules_{timestamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    total_non_compliant = sum(
        int(f.get("non_compliant_rows", 0)) for f in report["files"] if "error" not in f
    )
    print(f"Reviewed files: {len(csv_files)}")
    print(f"Total non-compliant rows: {total_non_compliant}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
