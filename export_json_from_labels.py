from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DEFAULT_CSV_PATH = Path("video_labels.csv")
DEFAULT_JSON_PATH = Path("exported_labels.json")
DEFAULT_SOURCE_JSON_PATH = Path(r"Data\prelabel_slim_mm_seed42.json")


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def _to_situation_lower(v: str) -> str:
    s = v.strip().lower()
    if s in {"affection", "intent", "attitude"}:
        return s
    return s


def _generic_label(rec: Dict[str, Any]) -> str:
    situation = _to_situation_lower(_safe_text(rec.get("situation", "")))
    if situation == "affection":
        return _safe_text(rec.get("label_Affection", ""))
    if situation == "intent":
        return _safe_text(rec.get("label_Intent", ""))
    if situation == "attitude":
        return _safe_text(rec.get("label_Attitude", ""))
    return ""


def _generic_mechanism(rec: Dict[str, Any]) -> str:
    situation = _to_situation_lower(_safe_text(rec.get("situation", "")))
    if situation == "affection":
        value = _safe_text(rec.get("mechanism_Affection", ""))
        return value if value else _safe_text(rec.get("mechanism", ""))
    if situation == "intent":
        value = _safe_text(rec.get("mechanism_Intent", ""))
        return value if value else _safe_text(rec.get("mechanism", ""))
    if situation == "attitude":
        value = _safe_text(rec.get("mechanism_Attitude", ""))
        return value if value else _safe_text(rec.get("mechanism", ""))
    return _safe_text(rec.get("mechanism", ""))


def _build_image_url(filename: str, base_image_url: str) -> str:
    if not filename:
        return ""
    base = base_image_url.strip()
    if not base:
        return filename
    return f"{base.rstrip('/')}/{filename}"


def _filename_from_item(item: Dict[str, Any]) -> str:
    input_obj = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
    url = _safe_text(input_obj.get("url", "")).strip()
    if url:
        return Path(url).name
    path = _safe_text(input_obj.get("path", "")).strip()
    if path:
        return Path(path).name
    return ""


def _load_source_items(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return [x for x in data["data"] if isinstance(x, dict)]
        return [data]
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
            "Original source JSON used as template to preserve fields like "
            "input.path/input.url. Default: "
            f"{DEFAULT_SOURCE_JSON_PATH}"
        ),
    )
    parser.add_argument(
        "--base-image-url",
        default="",
        help="Optional prefix for input.url when source json has no url.",
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
    base_image_url = args.base_image_url

    df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)
    source_items = _load_source_items(source_json_path)

    by_id: Dict[str, Dict[str, Any]] = {}
    by_filename: Dict[str, Dict[str, Any]] = {}
    for item in source_items:
        input_obj = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
        sid = _safe_text(input_obj.get("id", "")).strip()
        if sid and sid not in by_id:
            by_id[sid] = item
        fname = _filename_from_item(item)
        if fname and fname not in by_filename:
            by_filename[fname] = item

    result: List[Dict[str, Any]] = []
    abandoned_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        abandoned = str(rec.get("abandon", "")).strip().lower() in {"true", "1", "yes"}
        skipped = str(rec.get("skipped", "")).strip().lower() in {"true", "1", "yes"}
        filename = _safe_text(rec.get("filename", "")).strip()
        sid = _safe_text(rec.get("id", "")).strip()
        source_item = by_id.get(sid) or by_filename.get(filename)
        item = _copy_source_item(source_item)
        if not isinstance(item.get("input"), dict):
            item["input"] = {}
        if not isinstance(item.get("output"), dict):
            item["output"] = {}

        input_obj = item["input"]
        output_obj = item["output"]

        input_obj["id"] = sid or _safe_text(input_obj.get("id", "")).strip()
        input_obj["text"] = _safe_text(rec.get("input_text", "")).strip() or _safe_text(input_obj.get("text", "")).strip()

        if not _safe_text(input_obj.get("url", "")).strip():
            fallback_url = _build_image_url(filename, base_image_url)
            if fallback_url:
                input_obj["url"] = fallback_url

        output_obj["subject"] = _safe_text(rec.get("subject", "")).strip()
        output_obj["target"] = _safe_text(rec.get("target", "")).strip()
        output_obj["situation"] = _to_situation_lower(_safe_text(rec.get("situation", "")))
        output_obj["mechanism"] = _generic_mechanism(rec).strip()
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


if __name__ == "__main__":
    main()
