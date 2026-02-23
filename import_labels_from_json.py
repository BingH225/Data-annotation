from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd


CSV_COLUMNS = [
    "filename",
    "id",
    "input_text",
    "subject",
    "target",
    "situation",
    "mechanism_Affection",
    "mechanism_Intent",
    "mechanism_Attitude",
    "mechanism",
    "domain",
    "culture",
    "label_Affection",
    "label_Intent",
    "label_Attitude",
    "rationale",
    "skipped",
]

# Edit these two paths directly when you want to change default import target.
DEFAULT_INPUT_PATH = Path(r"C:\Users\xwhhh\Desktop\data_deal\Data\prelabel_slim_mm_seed42.json")
DEFAULT_OUTPUT_PATH = Path("video_labels.csv")
DEFAULT_MEDIA_DIR = Path("images")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _norm_situation(value: str) -> str:
    v = value.strip().lower()
    if v == "affection":
        return "Affection"
    if v == "intent":
        return "Intent"
    if v == "attitude":
        return "Attitude"
    return value


def _norm_affection_label(value: str) -> str:
    v = value.strip().lower()
    mapping = {
        "null": "NULL",
        "happy": "Happy",
        "sad": "Sad",
        "disgusted": "Disgusted",
        "angry": "Angry",
        "fearful": "Fearful",
        "bad": "Bad",
    }
    return mapping.get(v, value)


def _norm_mechanism_value(value: str) -> str:
    v = value.strip()
    if not v:
        return "NULL"
    if v.lower() in {"null", "none"}:
        return "NULL"
    return v


def _media_url_from_input(input_obj: Dict[str, Any]) -> str:
    # Prefer explicit HF-style "url", then backward-compatible "image_url".
    for key in ("url", "image_url"):
        value = _safe_text(input_obj.get(key, "")).strip()
        if value:
            return value
    return ""


def _filename_from_input(input_obj: Dict[str, Any]) -> str:
    for key in ("filename", "file_name", "image_file", "video_file"):
        if input_obj.get(key):
            return _safe_text(input_obj[key]).strip()

    image_url = _media_url_from_input(input_obj)
    if image_url:
        parsed = urlparse(image_url)
        name = Path(parsed.path).name
        if name:
            return name
    return ""


def _read_json_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return [x for x in data["data"] if isinstance(x, dict)]
        return [data]

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    return []


def _read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        item = json.loads(raw)
        if isinstance(item, dict):
            out.append(item)
    return out


def _iter_input_objects(src: Path) -> Iterable[Dict[str, Any]]:
    if src.is_file():
        suffix = src.suffix.lower()
        if suffix == ".jsonl":
            yield from _read_jsonl_file(src)
            return
        if suffix == ".json":
            yield from _read_json_file(src)
            return
        raise ValueError(f"Unsupported file type: {src}")

    if src.is_dir():
        for p in sorted(src.rglob("*")):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix == ".json":
                yield from _read_json_file(p)
            elif suffix == ".jsonl":
                yield from _read_jsonl_file(p)
        return

    raise ValueError(f"Input path does not exist: {src}")


def _to_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    input_obj = obj.get("input", {}) if isinstance(obj.get("input"), dict) else {}
    output_obj = obj.get("output", {}) if isinstance(obj.get("output"), dict) else {}

    situation = _norm_situation(_safe_text(output_obj.get("situation", "")).strip())
    label_generic = _safe_text(output_obj.get("label", "")).strip()
    mechanism_generic = _safe_text(output_obj.get("mechanism", "")).strip()

    label_affection = _safe_text(output_obj.get("label_Affection", "")).strip()
    label_intent = _safe_text(output_obj.get("label_Intent", "")).strip()
    label_attitude = _safe_text(output_obj.get("label_Attitude", "")).strip()
    mechanism_affection = _safe_text(output_obj.get("mechanism_Affection", "")).strip()
    mechanism_intent = _safe_text(output_obj.get("mechanism_Intent", "")).strip()
    mechanism_attitude = _safe_text(output_obj.get("mechanism_Attitude", "")).strip()

    if label_generic:
        if situation == "Affection":
            label_affection = label_generic
        elif situation == "Intent":
            label_intent = label_generic
        elif situation == "Attitude":
            label_attitude = label_generic

    # Business rule: if situation is not Intent, force Intent label to NULL.
    if situation != "Intent":
        label_intent = "NULL"
    elif not label_intent:
        label_intent = "NULL"

    label_affection = _norm_affection_label(label_affection)
    mechanism_affection = _norm_mechanism_value(mechanism_affection)
    mechanism_intent = _norm_mechanism_value(mechanism_intent)
    mechanism_attitude = _norm_mechanism_value(mechanism_attitude)
    mechanism_generic = _norm_mechanism_value(mechanism_generic)

    # Keep mechanism split fields consistent with situation, like label selection logic.
    if situation == "Affection":
        mechanism_affection = mechanism_generic if mechanism_affection == "NULL" else mechanism_affection
        mechanism_intent = "NULL"
        mechanism_attitude = "NULL"
    elif situation == "Intent":
        mechanism_intent = mechanism_generic if mechanism_intent == "NULL" else mechanism_intent
        mechanism_affection = "NULL"
        mechanism_attitude = "NULL"
    elif situation == "Attitude":
        mechanism_attitude = mechanism_generic if mechanism_attitude == "NULL" else mechanism_attitude
        mechanism_affection = "NULL"
        mechanism_intent = "NULL"

    if situation == "Affection":
        mechanism_generic = mechanism_affection
    elif situation == "Intent":
        mechanism_generic = mechanism_intent
    elif situation == "Attitude":
        mechanism_generic = mechanism_attitude

    record = {
        "filename": _filename_from_input(input_obj),
        "id": _safe_text(input_obj.get("id", "")).strip(),
        "input_text": _safe_text(input_obj.get("text", "")).strip(),
        "subject": _safe_text(output_obj.get("subject", "")).strip(),
        "target": _safe_text(output_obj.get("target", "")).strip(),
        "situation": situation,
        "mechanism_Affection": mechanism_affection,
        "mechanism_Intent": mechanism_intent,
        "mechanism_Attitude": mechanism_attitude,
        "mechanism": mechanism_generic,
        "domain": _safe_text(output_obj.get("domain", "")).strip(),
        "culture": _safe_text(output_obj.get("culture", "")).strip(),
        "label_Affection": label_affection,
        "label_Intent": label_intent,
        "label_Attitude": label_attitude,
        "rationale": _safe_text(output_obj.get("rationale", "")).strip(),
        "skipped": False,
    }
    return record


def _image_url_from_obj(obj: Dict[str, Any]) -> str:
    input_obj = obj.get("input", {}) if isinstance(obj.get("input"), dict) else {}
    return _media_url_from_input(input_obj)


def _download_media(image_url: str, filename: str, media_dir: Path) -> bool:
    if not image_url or not filename:
        return False

    media_dir.mkdir(parents=True, exist_ok=True)
    out_path = media_dir / filename
    if out_path.exists():
        return True

    try:
        with urlopen(image_url, timeout=20) as resp:
            data = resp.read()
        out_path.write_bytes(data)
        return True
    except Exception:
        return False


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = "" if col != "skipped" else False
        return df[CSV_COLUMNS].copy()
    return pd.DataFrame(columns=CSV_COLUMNS)


def _upsert(base_df: pd.DataFrame, new_records: List[Dict[str, Any]]) -> pd.DataFrame:
    if base_df.empty:
        return pd.DataFrame(new_records, columns=CSV_COLUMNS)

    df = base_df.copy()
    for rec in new_records:
        filename = str(rec["filename"])
        mask = df["filename"].astype(str) == filename
        if mask.any():
            idx = df.index[mask][0]
            for col in CSV_COLUMNS:
                df.at[idx, col] = rec.get(col, "" if col != "skipped" else False)
        else:
            df = pd.concat([df, pd.DataFrame([rec], columns=CSV_COLUMNS)], ignore_index=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import JSON/JSONL annotations into video_labels.csv."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help=(
            "Path to one .json/.jsonl file or a directory containing them. "
            f"Default: {DEFAULT_INPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--media-dir",
        default=str(DEFAULT_MEDIA_DIR),
        help=f"Directory to save downloaded media (default: {DEFAULT_MEDIA_DIR}).",
    )
    parser.add_argument(
        "--no-download-media",
        action="store_true",
        help="Do not download image_url files into media-dir.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    media_dir = Path(args.media_dir)
    download_media = not bool(args.no_download_media)

    raw_items = list(_iter_input_objects(input_path))
    records: List[Dict[str, Any]] = []
    skipped_no_filename = 0
    download_ok = 0
    download_fail = 0
    for item in raw_items:
        rec = _to_record(item)
        if not rec["filename"]:
            skipped_no_filename += 1
            continue
        records.append(rec)
        if download_media:
            image_url = _image_url_from_obj(item)
            if image_url:
                if _download_media(image_url, rec["filename"], media_dir):
                    download_ok += 1
                else:
                    download_fail += 1

    existing = _load_existing(output_path)
    merged = _upsert(existing, records)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Read objects: {len(raw_items)}")
    print(f"Imported records: {len(records)}")
    print(f"Skipped (missing filename): {skipped_no_filename}")
    if download_media:
        print(f"Media downloaded/already-exists: {download_ok}")
        print(f"Media download failed: {download_fail}")
        print(f"Media dir: {media_dir}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
