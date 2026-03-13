from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import pandas as pd


DEFAULT_CSV_PATH = Path("Data/review_assign_all_reindexed_scenario_sampled_300_merged.csv")
DEFAULT_JSON_PATH = Path("export/exported_labels.json")
DEFAULT_SOURCE_JSON_PATH = Path("Data/review_assign_all_reindexed_scenario_sampled_300.json")


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def _to_scenario_lower(v: str) -> str:
    s = v.strip().lower()
    if s in {"affection", "intent", "attitude"}:
        return s
    return s


def _generic_label(rec: Dict[str, Any]) -> str:
    scenario = _to_scenario_lower(_safe_text(rec.get("scenario", "")))
    if scenario == "affection":
        return _safe_text(rec.get("label_Affection", ""))
    if scenario == "intent":
        return _safe_text(rec.get("label_Intent", ""))
    if scenario == "attitude":
        return _safe_text(rec.get("label_Attitude", ""))
    return ""


def _generic_mechanism(rec: Dict[str, Any]) -> str:
    scenario = _to_scenario_lower(_safe_text(rec.get("scenario", "")))
    if scenario == "affection":
        value = _safe_text(rec.get("mechanism_Affection", ""))
        return value if value else _safe_text(rec.get("mechanism", ""))
    if scenario == "intent":
        value = _safe_text(rec.get("mechanism_Intent", ""))
        return value if value else _safe_text(rec.get("mechanism", ""))
    if scenario == "attitude":
        value = _safe_text(rec.get("mechanism_Attitude", ""))
        return value if value else _safe_text(rec.get("mechanism", ""))
    return _safe_text(rec.get("mechanism", ""))


def _filename_from_item(item: Dict[str, Any]) -> str:
    input_obj = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
    url = _safe_text(input_obj.get("url", "")).strip()
    if url:
        return Path(url).name
    media_path = _safe_text(input_obj.get("media_path", "")).strip()
    if media_path:
        return Path(media_path).name
    path = _safe_text(input_obj.get("path", "")).strip()
    if path:
        return Path(path).name
    media_path_local = _safe_text(input_obj.get("media_path_local", "")).strip()
    if media_path_local:
        return Path(media_path_local).name
    return ""


def _source_id_from_input(input_obj: Dict[str, Any]) -> str:
    return _safe_text(input_obj.get("id", "") or input_obj.get("samples_id", "")).strip()


def _source_lookup_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    input_obj = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
    return (
        _filename_from_item(item),
        _source_id_from_input(input_obj),
        _safe_text(input_obj.get("text", "")).strip(),
    )


def _csv_lookup_key(rec: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        _safe_text(rec.get("filename", "")).strip(),
        _safe_text(rec.get("id", "")).strip(),
        _safe_text(rec.get("input_text", "")).strip(),
    )


def _build_matcher(source_items: List[Dict[str, Any]]) -> Tuple[
    Dict[Tuple[str, str, str], Deque[int]],
    Dict[str, Deque[int]],
    Dict[str, Deque[int]],
]:
    by_key: Dict[Tuple[str, str, str], Deque[int]] = defaultdict(deque)
    by_id: Dict[str, Deque[int]] = defaultdict(deque)
    by_filename: Dict[str, Deque[int]] = defaultdict(deque)
    for idx, item in enumerate(source_items):
        key = _source_lookup_key(item)
        by_key[key].append(idx)
        sid = key[1]
        fname = key[0]
        if sid:
            by_id[sid].append(idx)
        if fname:
            by_filename[fname].append(idx)
    return by_key, by_id, by_filename


def _pop_unconsumed(queue: Optional[Deque[int]], consumed: Set[int]) -> Optional[int]:
    if queue is None:
        return None
    while queue and queue[0] in consumed:
        queue.popleft()
    if not queue:
        return None
    return queue.popleft()


def _load_source_items(path: Path) -> List[Dict[str, Any]]:
    def _load_one_file(src: Path) -> List[Dict[str, Any]]:
        suffix = src.suffix.lower()
        if suffix == ".jsonl":
            out: List[Dict[str, Any]] = []
            for line in src.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw:
                    continue
                item = json.loads(raw)
                if isinstance(item, dict):
                    out.append(item)
            return out

        data = json.loads(src.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                return [x for x in data["data"] if isinstance(x, dict)]
            return [data]
        return []

    if not path.exists():
        return []

    if path.is_file():
        return _load_one_file(path)

    if path.is_dir():
        items: List[Dict[str, Any]] = []
        for p in sorted(path.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".json", ".jsonl"}:
                continue
            items.extend(_load_one_file(p))
        return items

    return []


def _copy_source_item(item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not item:
        return {"input": {}, "output": {}}
    # Deep copy through JSON to avoid mutating loaded source structures.
    return json.loads(json.dumps(item, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export video_labels.csv back to input/output JSON format."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON_PATH))
    parser.add_argument(
        "--output-abandoned-json",
        default="abandoned_records.json",
        help="Output JSON file path for rows filtered by abandon/skipped.",
    )
    parser.add_argument(
        "--source-json",
        default=str(DEFAULT_SOURCE_JSON_PATH),
        help=(
            "Original source JSON/JSONL file or directory used as template to preserve "
            "input fields (supports mixed image/video sources). Default: "
            f"{DEFAULT_SOURCE_JSON_PATH}"
        ),
    )
    parser.add_argument(
        "--base-image-url",
        default="",
        help="Deprecated, ignored. Input fields are preserved from source JSON.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Include rows with abandon/skipped=True.",
    )
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    out_path = Path(args.output_json)
    abandoned_json_path = Path(args.output_abandoned_json)
    source_json_path = Path(args.source_json)

    df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)
    source_items = _load_source_items(source_json_path)

    by_key, by_id, by_filename = _build_matcher(source_items)
    consumed_source_indices: Set[int] = set()
    unmatched_rows = 0

    result: List[Dict[str, Any]] = []
    abandoned_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        abandoned = str(rec.get("abandon", "")).strip().lower() in {"true", "1", "yes"}
        skipped = str(rec.get("skipped", "")).strip().lower() in {"true", "1", "yes"}

        key = _csv_lookup_key(rec)
        source_idx = _pop_unconsumed(by_key.get(key), consumed_source_indices)
        if source_idx is None and key[1]:
            source_idx = _pop_unconsumed(by_id.get(key[1]), consumed_source_indices)
        if source_idx is None and key[0]:
            source_idx = _pop_unconsumed(by_filename.get(key[0]), consumed_source_indices)
        if source_idx is not None:
            consumed_source_indices.add(source_idx)
        else:
            unmatched_rows += 1

        source_item = source_items[source_idx] if source_idx is not None else None
        item = _copy_source_item(source_item)
        if not isinstance(item.get("input"), dict):
            item["input"] = {}
        if not isinstance(item.get("output"), dict):
            item["output"] = {}

        output_obj = item["output"]

        output_obj["subject"] = _safe_text(rec.get("subject", "")).strip()
        output_obj["target"] = _safe_text(rec.get("target", "")).strip()
        output_obj["subject1"] = _safe_text(rec.get("subject1", "")).strip()
        output_obj["subject2"] = _safe_text(rec.get("subject2", "")).strip()
        output_obj["subject3"] = _safe_text(rec.get("subject3", "")).strip()
        output_obj["target1"] = _safe_text(rec.get("target1", "")).strip()
        output_obj["target2"] = _safe_text(rec.get("target2", "")).strip()
        output_obj["target3"] = _safe_text(rec.get("target3", "")).strip()
        output_obj["scenario"] = _to_scenario_lower(_safe_text(rec.get("scenario", "")))
        output_obj["mechanism_Affection"] = _safe_text(rec.get("mechanism_Affection", "")).strip()
        output_obj["mechanism_Intent"] = _safe_text(rec.get("mechanism_Intent", "")).strip()
        output_obj["mechanism_Attitude"] = _safe_text(rec.get("mechanism_Attitude", "")).strip()
        output_obj["mechanism"] = _generic_mechanism(rec).strip()
        output_obj["label_Affection"] = _safe_text(rec.get("label_Affection", "")).strip()
        output_obj["label_Intent"] = _safe_text(rec.get("label_Intent", "")).strip()
        output_obj["label_Attitude"] = _safe_text(rec.get("label_Attitude", "")).strip()
        output_obj["label"] = _generic_label(rec).strip()
        output_obj["domain"] = _safe_text(rec.get("domain", "")).strip()
        output_obj["culture"] = _safe_text(rec.get("culture", "")).strip()
        output_obj["rationale"] = _safe_text(rec.get("rationale", "")).strip()

        filtered_out = abandoned or skipped
        if filtered_out:
            abandoned_rows.append(item)
            if not args.include_skipped:
                continue

        result.append(item)

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    abandoned_json_path.write_text(
        json.dumps(abandoned_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Exported: {len(result)}")
    print(f"Output JSON: {out_path}")
    print(f"Filtered rows: {len(abandoned_rows)}")
    print(f"Filtered rows JSON: {abandoned_json_path}")
    if unmatched_rows:
        print(f"Warning: source template not found for {unmatched_rows} CSV rows.")


if __name__ == "__main__":
    main()
