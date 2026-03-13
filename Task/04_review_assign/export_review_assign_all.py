from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = Path(__file__).resolve().parent
PRIMARY_ASSIGN_DIR = PROJECT_ROOT / "Task" / "02_primary_assignments"
OUTPUT_JSON_PATH = PROJECT_ROOT / "export" / "review_assign_all.json"


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def _to_scenario_lower(v: str) -> str:
    return v.strip().lower()


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
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return [x for x in data["data"] if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _copy_source_item(item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not item:
        return {"input": {}, "output": {}}
    return json.loads(json.dumps(item, ensure_ascii=False))


def _row_range_from_filename(csv_path: Path) -> Optional[Tuple[int, int]]:
    m = re.search(r"day\d+_(\d+)_(\d+)$", csv_path.stem, flags=re.IGNORECASE)
    if not m:
        return None
    start = int(m.group(1))
    end = int(m.group(2))
    if start <= 0 or end < start:
        return None
    return start, end


def _slice_df_by_name_range(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    r = _row_range_from_filename(csv_path)
    if not r:
        return df
    start, end = r
    return df.iloc[start - 1 : end].copy()


def _list_csv_files(csv_dir: Path) -> List[Path]:
    return sorted([p for p in csv_dir.glob("*.csv") if p.is_file()], key=lambda x: x.name.lower())


def _day_token_from_name(name: str) -> Optional[str]:
    m = re.search(r"(day\d+)", name, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower()


def _source_stem_for_csv(csv_path: Path) -> str:
    # Tan_primary_day08_001_150 -> Tan_primary_day08
    m = re.match(r"^(.*day\d+)_\d+_\d+$", csv_path.stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return csv_path.stem


def _resolve_source_json_for_csv(csv_path: Path, primary_assign_dir: Path) -> Path:
    day_token = _day_token_from_name(csv_path.stem)
    if not day_token:
        raise RuntimeError(f"Cannot parse day token from filename: {csv_path.name}")

    source_stem = _source_stem_for_csv(csv_path)
    day_dir = primary_assign_dir / day_token
    candidates = [
        day_dir / f"{source_stem}.json",
        day_dir / f"{source_stem}.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Source template not found for {csv_path.name}. Tried: {tried}"
    )


def export_all(
    csv_dir: Path,
    primary_assign_dir: Path,
    output_json: Path,
    include_skipped: bool,
) -> None:
    csv_files = _list_csv_files(csv_dir)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {csv_dir}")

    output_json.parent.mkdir(parents=True, exist_ok=True)

    all_result: List[Dict[str, Any]] = []
    total_filtered = 0
    total_unmatched = 0

    print(f"CSV dir: {csv_dir}")
    print(f"Primary assign dir: {primary_assign_dir}")
    print(f"CSV files: {len(csv_files)}")

    for csv_path in csv_files:
        source_json = _resolve_source_json_for_csv(csv_path, primary_assign_dir)
        source_items = _load_source_items(source_json)
        if not source_items:
            raise RuntimeError(f"Source JSON is empty: {source_json}")

        df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)
        original_rows = len(df)
        df = _slice_df_by_name_range(df, csv_path)

        by_key, by_id, by_filename = _build_matcher(source_items)
        consumed_source_indices: Set[int] = set()
        matched_rows = 0
        matched_by_row_index = 0
        unmatched_rows = 0
        file_filtered = 0
        file_result: List[Dict[str, Any]] = []

        for row_index, row in df.iterrows():
            rec = row.to_dict()
            abandoned = str(rec.get("abandon", "")).strip().lower() in {"true", "1", "yes"}
            skipped = str(rec.get("skipped", "")).strip().lower() in {"true", "1", "yes"}

            key = _csv_lookup_key(rec)
            source_idx = _pop_unconsumed(by_key.get(key), consumed_source_indices)
            if source_idx is None and key[1]:
                source_idx = _pop_unconsumed(by_id.get(key[1]), consumed_source_indices)
            if source_idx is None and key[0]:
                source_idx = _pop_unconsumed(by_filename.get(key[0]), consumed_source_indices)

            # Final fallback: align by original row index from CSV (0-based).
            if source_idx is None:
                try:
                    idx_int = int(row_index)
                except Exception:
                    idx_int = -1
                if 0 <= idx_int < len(source_items):
                    source_idx = idx_int
                    matched_by_row_index += 1

            if source_idx is not None:
                matched_rows += 1
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
            output_obj["mechanism_Affection"] = _safe_text(
                rec.get("mechanism_Affection", "")
            ).strip()
            output_obj["mechanism_Intent"] = _safe_text(rec.get("mechanism_Intent", "")).strip()
            output_obj["mechanism_Attitude"] = _safe_text(
                rec.get("mechanism_Attitude", "")
            ).strip()
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
                file_filtered += 1
                if not include_skipped:
                    continue

            file_result.append(item)

        all_result.extend(file_result)
        total_filtered += file_filtered
        total_unmatched += unmatched_rows
        sliced_rows = len(df)
        print(
            f"- {csv_path.name}: source={source_json.name}, rows={original_rows}, used={sliced_rows}, "
            f"matched={matched_rows}, row_fallback={matched_by_row_index}, unmatched={unmatched_rows}, "
            f"filtered={file_filtered}, exported={len(file_result)}"
        )

    output_json.write_text(json.dumps(all_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported total: {len(all_result)}")
    print(f"Total filtered rows: {total_filtered}")
    print(f"Total unmatched rows: {total_unmatched}")
    print(f"Output JSON: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export all CSVs under Task/04_review_assign into one combined JSON. "
            "If filename ends with dayXX_<start>_<end>, only rows [start, end] are exported."
        )
    )
    parser.add_argument("--csv-dir", default=str(CSV_DIR))
    parser.add_argument("--primary-assign-dir", default=str(PRIMARY_ASSIGN_DIR))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON_PATH))
    parser.add_argument("--include-skipped", action="store_true")
    args = parser.parse_args()

    export_all(
        csv_dir=Path(args.csv_dir),
        primary_assign_dir=Path(args.primary_assign_dir),
        output_json=Path(args.output_json),
        include_skipped=bool(args.include_skipped),
    )


if __name__ == "__main__":
    main()
