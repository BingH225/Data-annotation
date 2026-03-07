from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


CSV_COLUMNS = [
    "filename",
    "id",
    "input_text",
    "subject",
    "target",
    "subject1",
    "subject2",
    "subject3",
    "target1",
    "target2",
    "target3",
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

# Edit these paths directly when you want to change default import target.
DEFAULT_INPUT_PATH = Path(
    r"D:\NUS\ACMm\Data-annotation\Task\02_primary_assignments\day08\Yu_primary_day08.json"
)
DEFAULT_OUTPUT_PATH = Path(
    r"D:\NUS\ACMm\Data-annotation\Task\02_primary_assignments\day08\Yu_primary_day08.csv"
)
DEFAULT_MEDIA_DIR = Path(r"D:\NUS\ACMm\Data-annotation\images")


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
    # For mixed image/video payloads:
    # - video: prefer media_path
    # - image: fallback to url
    media_path = _safe_text(input_obj.get("media_path", "")).strip()
    if media_path:
        return media_path

    for key in ("url", "image_url", "video_url", "media_url"):
        value = _safe_text(input_obj.get(key, "")).strip()
        if value:
            return value
    return ""


def _normalize_download_url(image_url: str) -> str:
    url = _safe_text(image_url).strip()
    if not url:
        return ""
    # Hugging Face page URLs (/blob/) are not raw file links; convert to /resolve/.
    if "huggingface.co" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    return url


def _is_html_payload(data: bytes) -> bool:
    probe = data[:4096].lower()
    return b"<html" in probe or b"<!doctype html" in probe


def _is_valid_media_bytes(filename: str, data: bytes) -> Tuple[bool, str]:
    if not data:
        return False, "empty file content"
    if _is_html_payload(data):
        return False, "html content instead of media"

    ext = Path(filename).suffix.lower()
    if ext == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n"), "invalid png signature"
    if ext in {".jpg", ".jpeg"}:
        return data.startswith(b"\xff\xd8"), "invalid jpg signature"
    if ext == ".gif":
        return (
            data.startswith(b"GIF87a") or data.startswith(b"GIF89a"),
            "invalid gif signature",
        )
    if ext == ".webp":
        return (
            len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP",
            "invalid webp signature",
        )
    if ext == ".bmp":
        return data.startswith(b"BM"), "invalid bmp signature"
    if ext in {".mp4", ".mov"}:
        return len(data) >= 12 and data[4:8] == b"ftyp", "invalid mp4/mov signature"
    if ext == ".avi":
        return (
            len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ",
            "invalid avi signature",
        )
    if ext in {".mkv", ".webm"}:
        return data.startswith(bytes.fromhex("1A45DFA3")), "invalid mkv/webm signature"
    return True, ""


def _is_valid_media_file(path: Path) -> Tuple[bool, str]:
    if not path.exists() or not path.is_file():
        return False, "file does not exist"
    try:
        data = path.read_bytes()
    except Exception as exc:
        return False, f"read failed: {exc}"
    return _is_valid_media_bytes(path.name, data)


def _filename_from_input(input_obj: Dict[str, Any]) -> str:
    for key in ("filename", "file_name", "image_file", "video_file", "media_file"):
        if input_obj.get(key):
            return _safe_text(input_obj[key]).strip()

    # Prefer URL basename because local paths may be non-unique across samples.
    image_url = _media_url_from_input(input_obj)
    if image_url:
        parsed = urlparse(image_url)
        name = Path(parsed.path).name
        if name:
            return name

    # Fallback for local path-only payloads.
    for key in ("media_path_local", "path", "image_path", "video_path"):
        value = _safe_text(input_obj.get(key, "")).strip()
        if value:
            name = Path(value).name
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
        "id": _safe_text(input_obj.get("id", "") or input_obj.get("samples_id", "")).strip(),
        "input_text": _safe_text(input_obj.get("text", "")).strip(),
        "subject": _safe_text(output_obj.get("subject", "")).strip(),
        "target": _safe_text(output_obj.get("target", "")).strip(),
        "subject1": _safe_text(output_obj.get("subject1", "")).strip(),
        "subject2": _safe_text(output_obj.get("subject2", "")).strip(),
        "subject3": _safe_text(output_obj.get("subject3", "")).strip(),
        "target1": _safe_text(output_obj.get("target1", "")).strip(),
        "target2": _safe_text(output_obj.get("target2", "")).strip(),
        "target3": _safe_text(output_obj.get("target3", "")).strip(),
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


def _download_media_once(
    image_url: str,
    filename: str,
    media_dir: Path,
    timeout_seconds: int,
) -> Tuple[bool, str]:
    image_url = _normalize_download_url(image_url)
    if not image_url or not filename:
        return False, "missing url or filename"

    media_dir.mkdir(parents=True, exist_ok=True)
    out_path = media_dir / filename
    if out_path.exists():
        ok_existing, err_existing = _is_valid_media_file(out_path)
        if ok_existing:
            return True, ""
        # Existing file is corrupted (commonly an HTML page saved as media); replace it.
        try:
            out_path.unlink()
        except Exception as exc:
            return False, f"cannot remove invalid existing file: {exc}; reason={err_existing}"

    try:
        request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=timeout_seconds) as resp:
            data = resp.read()
        ok_data, err_data = _is_valid_media_bytes(filename, data)
        if not ok_data:
            return False, err_data
        out_path.write_bytes(data)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _download_media_with_retries(
    image_url: str,
    filename: str,
    media_dir: Path,
    max_attempts: int,
    retry_delay_seconds: float,
    timeout_seconds: int,
) -> Tuple[bool, int, str]:
    if not image_url or not filename:
        return False, 0, "missing url or filename"

    out_path = media_dir / filename
    if out_path.exists():
        return True, 0, ""

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        ok, err = _download_media_once(
            image_url=image_url,
            filename=filename,
            media_dir=media_dir,
            timeout_seconds=timeout_seconds,
        )
        if ok:
            return True, attempt, ""
        last_error = err
        if retry_delay_seconds > 0 and attempt < max_attempts:
            time.sleep(retry_delay_seconds)

    return False, max_attempts, last_error


def _download_until_limit(
    filename_to_url: Dict[str, str],
    media_dir: Path,
    max_file_attempts: int,
    max_rounds: int,
    max_total_attempts: int,
    retry_delay_seconds: float,
    timeout_seconds: int,
) -> Dict[str, Any]:
    pending: Set[str] = set(filename_to_url.keys())
    attempts_by_file: Dict[str, int] = {name: 0 for name in filename_to_url}
    last_error_by_file: Dict[str, str] = {}
    total_attempts = 0
    downloaded_this_run = 0
    already_exists = 0
    rounds_run = 0

    for name in list(pending):
        out_path = media_dir / name
        if out_path.exists():
            ok_existing, _ = _is_valid_media_file(out_path)
            if ok_existing:
                pending.remove(name)
                already_exists += 1
            else:
                try:
                    out_path.unlink()
                except Exception:
                    pass

    while pending and rounds_run < max_rounds and total_attempts < max_total_attempts:
        rounds_run += 1
        round_progress = False

        for name in list(pending):
            out_path = media_dir / name
            if out_path.exists():
                ok_existing, _ = _is_valid_media_file(out_path)
                if ok_existing:
                    pending.remove(name)
                    round_progress = True
                    continue
                try:
                    out_path.unlink()
                except Exception:
                    pass

            per_file_remaining = max_file_attempts - attempts_by_file[name]
            total_remaining = max_total_attempts - total_attempts
            tries = min(per_file_remaining, total_remaining)
            if tries <= 0:
                continue

            ok, consumed, err = _download_media_with_retries(
                image_url=filename_to_url[name],
                filename=name,
                media_dir=media_dir,
                max_attempts=tries,
                retry_delay_seconds=retry_delay_seconds,
                timeout_seconds=timeout_seconds,
            )
            attempts_by_file[name] += consumed
            total_attempts += consumed

            if ok:
                pending.remove(name)
                downloaded_this_run += 1
                round_progress = True
            elif err:
                last_error_by_file[name] = err

            if total_attempts >= max_total_attempts:
                break

        if not round_progress:
            break

    exhausted_file_cap = sorted(
        [name for name in pending if attempts_by_file.get(name, 0) >= max_file_attempts]
    )
    unresolved = sorted(pending)

    return {
        "target_files": len(filename_to_url),
        "already_exists": already_exists,
        "downloaded_this_run": downloaded_this_run,
        "resolved_files": len(filename_to_url) - len(unresolved),
        "unresolved_files": len(unresolved),
        "unresolved_filenames": unresolved,
        "attempts_total": total_attempts,
        "attempts_by_file": attempts_by_file,
        "rounds_run": rounds_run,
        "max_file_attempts_hit": exhausted_file_cap,
        "last_error_by_file": last_error_by_file,
    }


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        for col in CSV_COLUMNS:
            if col not in df.columns:
                if col == "skipped":
                    df[col] = False
                else:
                    df[col] = ""
        return df[CSV_COLUMNS].copy()
    return pd.DataFrame(columns=CSV_COLUMNS)


def _upsert(base_df: pd.DataFrame, new_records: List[Dict[str, Any]]) -> pd.DataFrame:
    if base_df.empty:
        return pd.DataFrame(new_records, columns=CSV_COLUMNS)

    df = base_df.copy()
    if "id" not in df.columns:
        df["id"] = ""
    if "input_text" not in df.columns:
        df["input_text"] = ""
    for rec in new_records:
        filename = _safe_text(rec.get("filename", "")).strip()
        rec_id = _safe_text(rec.get("id", "")).strip()
        rec_text = _safe_text(rec.get("input_text", "")).strip()
        mask = (
            (df["filename"].astype(str).str.strip() == filename)
            & (df["id"].astype(str).str.strip() == rec_id)
            & (df["input_text"].astype(str).str.strip() == rec_text)
        )
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
        help="Do not download media files into media-dir.",
    )
    parser.add_argument(
        "--max-file-attempts",
        type=int,
        default=8,
        help="Max attempts per file before giving up (default: 8).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Max full retry rounds over pending files (default: 10).",
    )
    parser.add_argument(
        "--max-total-attempts",
        type=int,
        default=3000,
        help="Global max download attempts across all files (default: 3000).",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between retries for the same file in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--download-timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout per download attempt in seconds (default: 60).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    media_dir = Path(args.media_dir)
    download_media = not bool(args.no_download_media)

    raw_items = list(_iter_input_objects(input_path))
    records: List[Dict[str, Any]] = []
    skipped_no_filename = 0
    first_url_by_filename: Dict[str, str] = {}
    for item in raw_items:
        rec = _to_record(item)
        if not rec["filename"]:
            skipped_no_filename += 1
            continue
        records.append(rec)
        image_url = _image_url_from_obj(item).strip()
        if image_url and rec["filename"] not in first_url_by_filename:
            first_url_by_filename[rec["filename"]] = image_url

    existing = _load_existing(output_path)
    merged = _upsert(existing, records)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    download_stats: Optional[Dict[str, Any]] = None
    no_url_filenames: Set[str] = set()
    if download_media:
        csv_filenames = {
            str(x).strip()
            for x in merged.get("filename", pd.Series(dtype=str)).tolist()
            if str(x).strip()
        }
        no_url_filenames = csv_filenames - set(first_url_by_filename.keys())
        download_targets = {
            name: first_url_by_filename[name]
            for name in csv_filenames
            if name in first_url_by_filename
        }
        download_stats = _download_until_limit(
            filename_to_url=download_targets,
            media_dir=media_dir,
            max_file_attempts=max(1, int(args.max_file_attempts)),
            max_rounds=max(1, int(args.max_rounds)),
            max_total_attempts=max(1, int(args.max_total_attempts)),
            retry_delay_seconds=max(0.0, float(args.retry_delay_seconds)),
            timeout_seconds=max(1, int(args.download_timeout_seconds)),
        )

    print(f"Read objects: {len(raw_items)}")
    print(f"Imported records: {len(records)}")
    print(f"Skipped (missing filename): {skipped_no_filename}")
    if download_stats is not None:
        print(f"Media target files (from CSV with URL): {download_stats['target_files']}")
        print(f"Media already exists: {download_stats['already_exists']}")
        print(f"Media downloaded this run: {download_stats['downloaded_this_run']}")
        print(f"Media resolved total: {download_stats['resolved_files']}")
        print(f"Media unresolved: {download_stats['unresolved_files']}")
        print(f"Retry rounds run: {download_stats['rounds_run']}")
        print(f"Total download attempts: {download_stats['attempts_total']}")
        print(f"No URL for CSV filenames: {len(no_url_filenames)}")
        if download_stats["max_file_attempts_hit"]:
            print(
                "Files hitting per-file max attempts: "
                f"{len(download_stats['max_file_attempts_hit'])}"
            )
        if download_stats["unresolved_filenames"]:
            print("Unresolved sample files:")
            for name in download_stats["unresolved_filenames"][:20]:
                err = download_stats["last_error_by_file"].get(name, "")
                suffix = f" | last_error={err}" if err else ""
                print(f"- {name}{suffix}")
            if len(download_stats["unresolved_filenames"]) > 20:
                print(
                    f"... and {len(download_stats['unresolved_filenames']) - 20} more."
                )
        print(f"Media dir: {media_dir}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
