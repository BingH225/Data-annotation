from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

pd = None  # Legacy CSV code below is preserved but no longer used in JSON evaluation mode.
import streamlit as st
from PIL import Image


# =========================
# Configuration
# =========================
MEDIA_DIR = Path("images")  # Directory containing images and videos
LABELS_CSV = Path(r"Data\review_assign_all_reindexed_scenario_sampled_300.csv")
INPUT_JSON_DEFAULT = Path(r"Data\review_assign_all.json")
PREVIEW_WIDTH = 280
PREVIEW_HEIGHT = 220

SCENARIO_OPTIONS = ["Affection", "Intent", "Attitude"]
MECHANISM_AFFECTION_OPTIONS = [
    "NULL",
    "multimodal_incongruity",
    "figurative_semantics",
    "affective_deception",
    "socio_cultural_dependency",
]
MECHANISM_INTENT_OPTIONS = [
    "NULL",
    "prosocial_deception",
    "malicious_manipulation",
    "expressive_aggression",
    "benevolent_provocation",
]
MECHANISM_ATTITUDE_OPTIONS = [
    "NULL",
    "dominant_affiliation",
    "dominant_detachment",
    "protective_distancing",
    "submissive_alignment",
]
DOMAIN_OPTIONS = ["NULL", "NULL", "NULL"]
CULTURE_OPTIONS = ["NULL", "NULL", "NULL"]
Affection_OPTIONS = ["NULL", "Happy", "Sad", "Disgusted", "Angry", "Fearful", "Surprised", "Bad"]
INTENT_OPTIONS = [
    "NULL",
    "alienate",
    "condemn",
    "denounce",
    "dominate",
    "intimidate",
    "mitigate",
    "mock",
    "provoke",
]
ATTITUDE_OPTIONS = [
    "NULL",
    "Supportive", "Appreciative", "Sympathetic", "Neutral", "Indifferent",
    "Disapproving", "Skeptical", "Concerned", "Dismissive", "Contemptuous", "Hostile"
]

EVAL_FIELDS = [
    "subject",
    "target",
    "subject1",
    "subject2",
    "subject3",
    "target1",
    "target2",
    "target3",
    "scenario",
    "mechanism_Affection",
    "mechanism_Intent",
    "mechanism_Attitude",
    "label_Affection",
    "label_Intent",
    "label_Attitude",
]
EVAL_COLUMN_1_FIELDS = [
    "scenario",
    "mechanism_Affection",
    "mechanism_Intent",
    "mechanism_Attitude",
]
EVAL_COLUMN_2_FIELDS = [
    "label_Affection",
    "label_Intent",
    "label_Attitude",
]
EVAL_COLUMN_3_FIELDS = [
    "subject",
    "subject1",
    "subject2",
    "subject3",
]
EVAL_COLUMN_4_FIELDS = [
    "target",
    "target1",
    "target2",
    "target3",
]
HIDDEN_FIELDS = {"domain", "culture", "rationale"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# =========================
# CSV Column Definitions
# =========================
CSV_COLUMNS = [
    "filename",
    "id",
    "input_text",  # 鉁?NEW
    "subject",
    "target",
    "subject1",
    "subject2",
    "subject3",
    "target1",
    "target2",
    "target3",
    "scenario",
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
    "abandon",
]


# =========================
# Helpers
# =========================
def _safe_choice(value: Any, options: List[str], allow_empty: bool = False) -> Any:
    """
    Ensure default value for selectbox / radio / pills is valid:
    - If value is in options, return value
    - Otherwise:
        - If allow_empty is True, return "" (or None)
        - Otherwise return options[0] (or "" if options is empty)
    """
    if value in options:
        return value
    if allow_empty:
        return "NULL" if "NULL" in options else ""
    return options[0] if options else ""


def _normalize_choice_in_state(key: str, options: List[str], allow_empty: bool = False) -> None:
    """Normalize st.session_state[key] to a valid value in options."""
    current = st.session_state.get(key, "")
    st.session_state[key] = _safe_choice(current, options, allow_empty=allow_empty)


def _safe_text(v: Any) -> str:
    """Convert any value into a safe string for Streamlit text_input/text_area."""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _ensure_text_state(keys: List[str]) -> None:
    """Force session_state keys to valid strings (prevents Streamlit widget crashes)."""
    for k in keys:
        st.session_state[k] = _safe_text(st.session_state.get(k, ""))


def _clear_bad_widget_state(keys: List[str]) -> None:
    """
    Streamlit can persist a corrupted widget state (non-str) across reruns.
    Remove those keys before widgets are created to avoid crashes.
    """
    for k in keys:
        if k in st.session_state and not isinstance(st.session_state.get(k), str):
            del st.session_state[k]


def _supported_media_files(allowed_filenames: Optional[Set[str]] = None) -> List[Path]:
    """Load supported image and video files."""
    if not MEDIA_DIR.exists():
        return []
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    video_ext = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    supported_ext = image_ext | video_ext
    files = [p for p in MEDIA_DIR.iterdir() if p.is_file() and p.suffix.lower() in supported_ext]
    if allowed_filenames:
        files = [p for p in files if p.name in allowed_filenames]
    return sorted(files, key=lambda p: p.name.lower())


def _is_image(file_path: Path) -> bool:
    """Check if file is an image format."""
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return file_path.suffix.lower() in image_ext


def _is_video(file_path: Path) -> bool:
    """Check if file is a video format."""
    video_ext = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    return file_path.suffix.lower() in video_ext


def _default_output_json_path(input_json_path: Path) -> Path:
    if input_json_path.is_file():
        return input_json_path.with_name(f"{input_json_path.stem}_evaluation.json")
    if input_json_path.is_dir():
        return input_json_path / f"{input_json_path.name}_evaluation.json"
    return INPUT_JSON_DEFAULT.with_name(f"{INPUT_JSON_DEFAULT.stem}_evaluation.json")


def _process_json_path(output_json_path: Path) -> Path:
    return output_json_path.with_name(f"{output_json_path.stem}_process.json")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
        if v in {"", "none", "null"}:
            return None
    return None


def _default_evaluation() -> Dict[str, Any]:
    return {
        "field_pass": {field: None for field in EVAL_FIELDS},
        "failed_fields": [],
        "overall_pass": None,
        "evaluated": False,
        "evaluated_at": None,
    }


def _normalize_evaluation(evaluation: Any) -> Dict[str, Any]:
    if not isinstance(evaluation, dict):
        return _default_evaluation()

    out = _default_evaluation()
    raw_field_pass = evaluation.get("field_pass")
    if isinstance(raw_field_pass, dict):
        for field in EVAL_FIELDS:
            out["field_pass"][field] = _parse_bool(raw_field_pass.get(field))

    raw_failed = evaluation.get("failed_fields")
    failed: List[str] = []
    if isinstance(raw_failed, list):
        for field in raw_failed:
            if isinstance(field, str) and field in EVAL_FIELDS and field not in failed:
                failed.append(field)
    if not failed:
        failed = [field for field, passed in out["field_pass"].items() if passed is False]
    for field in failed:
        out["field_pass"][field] = False

    evaluated = _parse_bool(evaluation.get("evaluated"))
    out["evaluated"] = bool(evaluated)

    overall_pass = _parse_bool(evaluation.get("overall_pass"))
    if overall_pass is None and out["evaluated"]:
        overall_pass = len(failed) == 0
    out["overall_pass"] = overall_pass
    out["failed_fields"] = [field for field in EVAL_FIELDS if field in set(failed)]
    evaluated_at = _safe_text(evaluation.get("evaluated_at", "")).strip()
    out["evaluated_at"] = evaluated_at or None
    return out


def _norm_scenario(value: str) -> str:
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
        "surprised": "Surprised",
        "bad": "Bad",
    }
    return mapping.get(v, value)


def _norm_intent_label(value: str) -> str:
    v = value.strip().lower()
    mapping = {
        "null": "NULL",
        "alienate": "alienate",
        "condemn": "condemn",
        "denounce": "denounce",
        "dominate": "dominate",
        "intimidate": "intimidate",
        "mitigate": "mitigate",
        "mock": "mock",
        "provoke": "provoke",
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
    media_path = _safe_text(input_obj.get("media_path", "")).strip()
    if media_path:
        return media_path
    for key in ("url", "image_url", "video_url", "media_url"):
        value = _safe_text(input_obj.get(key, "")).strip()
        if value:
            return value
    return ""


def _normalize_download_url(url: str) -> str:
    value = _safe_text(url).strip()
    if not value:
        return ""
    if "huggingface.co" in value and "/blob/" in value:
        return value.replace("/blob/", "/resolve/")
    return value


def _filename_from_input(input_obj: Dict[str, Any]) -> str:
    for key in ("filename", "file_name", "image_file", "video_file", "media_file"):
        value = _safe_text(input_obj.get(key, "")).strip()
        if value:
            return value

    media_url = _media_url_from_input(input_obj)
    if media_url:
        parsed = urlparse(media_url)
        name = Path(parsed.path).name
        if name:
            return name

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


def _iter_input_objects(src: Path) -> List[Dict[str, Any]]:
    if src.is_file():
        suffix = src.suffix.lower()
        if suffix == ".jsonl":
            return _read_jsonl_file(src)
        if suffix == ".json":
            return _read_json_file(src)
        raise ValueError(f"Unsupported file type: {src}")

    if src.is_dir():
        out: List[Dict[str, Any]] = []
        for p in sorted(src.rglob("*")):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix == ".json":
                out.extend(_read_json_file(p))
            elif suffix == ".jsonl":
                out.extend(_read_jsonl_file(p))
        return out
    raise ValueError(f"Input path does not exist: {src}")


def _normalize_output(output_obj: Dict[str, Any]) -> Dict[str, Any]:
    scenario = _norm_scenario(
        _safe_text(output_obj.get("scenario", output_obj.get("situation", ""))).strip()
    )
    label_generic = _safe_text(output_obj.get("label", "")).strip()
    mechanism_generic = _safe_text(output_obj.get("mechanism", "")).strip()

    label_affection = _safe_text(output_obj.get("label_Affection", "")).strip()
    label_intent = _safe_text(output_obj.get("label_Intent", "")).strip()
    label_attitude = _safe_text(output_obj.get("label_Attitude", "")).strip()
    mechanism_affection = _safe_text(output_obj.get("mechanism_Affection", "")).strip()
    mechanism_intent = _safe_text(output_obj.get("mechanism_Intent", "")).strip()
    mechanism_attitude = _safe_text(output_obj.get("mechanism_Attitude", "")).strip()

    if label_generic:
        if scenario == "Affection":
            label_affection = label_generic
        elif scenario == "Intent":
            label_intent = label_generic
        elif scenario == "Attitude":
            label_attitude = label_generic

    if scenario != "Intent":
        label_intent = "NULL"
    elif not label_intent:
        label_intent = "NULL"

    label_affection = _norm_affection_label(label_affection)
    label_intent = _norm_intent_label(label_intent)
    mechanism_affection = _norm_mechanism_value(mechanism_affection)
    mechanism_intent = _norm_mechanism_value(mechanism_intent)
    mechanism_attitude = _norm_mechanism_value(mechanism_attitude)
    mechanism_generic = _norm_mechanism_value(mechanism_generic)

    if scenario == "Affection":
        mechanism_affection = mechanism_generic if mechanism_affection == "NULL" else mechanism_affection
        mechanism_intent = "NULL"
        mechanism_attitude = "NULL"
    elif scenario == "Intent":
        mechanism_intent = mechanism_generic if mechanism_intent == "NULL" else mechanism_intent
        mechanism_affection = "NULL"
        mechanism_attitude = "NULL"
    elif scenario == "Attitude":
        mechanism_attitude = mechanism_generic if mechanism_attitude == "NULL" else mechanism_attitude
        mechanism_affection = "NULL"
        mechanism_intent = "NULL"

    if scenario == "Affection":
        mechanism_generic = mechanism_affection
    elif scenario == "Intent":
        mechanism_generic = mechanism_intent
    elif scenario == "Attitude":
        mechanism_generic = mechanism_attitude

    abandon_raw = _parse_bool(output_obj.get("abandon"))
    normalized = {
        "subject": _safe_text(output_obj.get("subject", "")).strip(),
        "target": _safe_text(output_obj.get("target", "")).strip(),
        "subject1": _safe_text(output_obj.get("subject1", "")).strip(),
        "subject2": _safe_text(output_obj.get("subject2", "")).strip(),
        "subject3": _safe_text(output_obj.get("subject3", "")).strip(),
        "target1": _safe_text(output_obj.get("target1", "")).strip(),
        "target2": _safe_text(output_obj.get("target2", "")).strip(),
        "target3": _safe_text(output_obj.get("target3", "")).strip(),
        "scenario": scenario,
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
        "abandon": bool(abandon_raw) if abandon_raw is not None else False,
    }
    normalized["evaluation"] = _normalize_evaluation(output_obj.get("evaluation"))
    return normalized


def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    input_obj = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
    output_obj = item.get("output", {}) if isinstance(item.get("output"), dict) else {}
    return {"input": input_obj, "output": _normalize_output(output_obj)}


def _item_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    input_obj = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
    filename = _filename_from_input(input_obj).strip()
    item_id = _safe_text(input_obj.get("id", "") or input_obj.get("samples_id", "")).strip()
    input_text = _safe_text(input_obj.get("text", "")).strip()
    return filename, item_id, input_text


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


def _download_media_once(
    media_url: str,
    filename: str,
    media_dir: Path,
    timeout_seconds: int,
) -> Tuple[bool, str]:
    media_url = _normalize_download_url(media_url)
    if not media_url or not filename:
        return False, "missing url or filename"

    media_dir.mkdir(parents=True, exist_ok=True)
    out_path = media_dir / filename
    if out_path.exists():
        ok_existing, err_existing = _is_valid_media_file(out_path)
        if ok_existing:
            return True, ""
        try:
            out_path.unlink()
        except Exception as exc:
            return False, f"cannot remove invalid existing file: {exc}; reason={err_existing}"

    try:
        request = Request(media_url, headers={"User-Agent": "Mozilla/5.0"})
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
    media_url: str,
    filename: str,
    media_dir: Path,
    max_attempts: int,
    retry_delay_seconds: float,
    timeout_seconds: int,
) -> Tuple[bool, int, str]:
    if not media_url or not filename:
        return False, 0, "missing url or filename"

    out_path = media_dir / filename
    if out_path.exists():
        ok_existing, _ = _is_valid_media_file(out_path)
        if ok_existing:
            return True, 0, ""

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        ok, err = _download_media_once(
            media_url=media_url,
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
            per_file_remaining = max_file_attempts - attempts_by_file[name]
            total_remaining = max_total_attempts - total_attempts
            tries = min(per_file_remaining, total_remaining)
            if tries <= 0:
                continue

            ok, consumed, err = _download_media_with_retries(
                media_url=filename_to_url[name],
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


def _load_json_items(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_items: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            raw_items = [x for x in data["data"] if isinstance(x, dict)]
        else:
            raw_items = [data]
    elif isinstance(data, list):
        raw_items = [x for x in data if isinstance(x, dict)]
    return [_normalize_item(item) for item in raw_items]


def _save_json_items(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_items_keep_evaluation(
    new_items: List[Dict[str, Any]],
    existing_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    existing_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for item in existing_items:
        key = _item_key(item)
        if key not in existing_map:
            existing_map[key] = item

    merged: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for item in new_items:
        key = _item_key(item)
        seen.add(key)
        old = existing_map.get(key)
        if old is not None:
            item["output"]["evaluation"] = _normalize_evaluation(
                old.get("output", {}).get("evaluation")
            )
            old_abandon = _parse_bool(old.get("output", {}).get("abandon"))
            item["output"]["abandon"] = bool(old_abandon) if old_abandon is not None else False
        else:
            item["output"]["evaluation"] = _normalize_evaluation(item["output"].get("evaluation"))
            new_abandon = _parse_bool(item["output"].get("abandon"))
            item["output"]["abandon"] = bool(new_abandon) if new_abandon is not None else False
        merged.append(item)

    for item in existing_items:
        key = _item_key(item)
        if key not in seen:
            merged.append(item)

    return merged


def _count_completed(items: List[Dict[str, Any]]) -> int:
    count = 0
    for item in items:
        evaluation = _normalize_evaluation(item.get("output", {}).get("evaluation"))
        abandon_raw = _parse_bool(item.get("output", {}).get("abandon"))
        is_abandon = bool(abandon_raw) if abandon_raw is not None else False
        if is_abandon or bool(evaluation.get("evaluated")):
            count += 1
    return count


def _save_process_json(
    process_path: Path,
    input_json_path: Path,
    output_json_path: Path,
    items: List[Dict[str, Any]],
    current_index: int,
) -> None:
    total = len(items)
    if total > 0:
        current_index = max(0, min(current_index, total - 1))
    else:
        current_index = 0
    completed = _count_completed(items)
    data = {
        "input_json": str(input_json_path),
        "output_json": str(output_json_path),
        "total": total,
        "completed": completed,
        "remaining": max(0, total - completed),
        "current_index": current_index,
        "updated_at": _now_iso(),
    }
    process_path.parent.mkdir(parents=True, exist_ok=True)
    process_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_process_index(process_path: Path, total: int) -> int:
    if not process_path.exists():
        return 0
    try:
        data = json.loads(process_path.read_text(encoding="utf-8"))
        idx = int(data.get("current_index", 0))
        if total <= 0:
            return 0
        return max(0, min(idx, total - 1))
    except Exception:
        return 0


def _apply_evaluation_result(item: Dict[str, Any], failed_fields: List[str]) -> None:
    failed_set = set(failed_fields)
    evaluation = {
        "field_pass": {field: (field not in failed_set) for field in EVAL_FIELDS},
        "failed_fields": [field for field in EVAL_FIELDS if field in failed_set],
        "overall_pass": len(failed_set) == 0,
        "evaluated": True,
        "evaluated_at": _now_iso(),
    }
    output = item.setdefault("output", {})
    output["evaluation"] = evaluation
    output["abandon"] = False


def _apply_abandon_result(item: Dict[str, Any]) -> None:
    output = item.setdefault("output", {})
    output["abandon"] = True
    output["evaluation"] = _default_evaluation()


def _start_import_from_json(input_json_path: Path) -> Dict[str, Any]:
    raw_items = _iter_input_objects(input_json_path)
    normalized_items: List[Dict[str, Any]] = []
    skipped_no_filename = 0
    for raw in raw_items:
        item = _normalize_item(raw)
        filename, _, _ = _item_key(item)
        if not filename:
            skipped_no_filename += 1
            continue
        normalized_items.append(item)

    output_json_path = _default_output_json_path(input_json_path)
    process_json_path = _process_json_path(output_json_path)
    existing_items = _load_json_items(output_json_path)
    merged_items = _merge_items_keep_evaluation(normalized_items, existing_items)
    _save_json_items(output_json_path, merged_items)

    filename_to_url: Dict[str, str] = {}
    for item in merged_items:
        input_obj = item.get("input", {})
        if not isinstance(input_obj, dict):
            continue
        filename = _filename_from_input(input_obj).strip()
        media_url = _media_url_from_input(input_obj).strip()
        if filename and media_url and filename not in filename_to_url:
            filename_to_url[filename] = media_url

    download_stats = _download_until_limit(
        filename_to_url=filename_to_url,
        media_dir=MEDIA_DIR,
        max_file_attempts=8,
        max_rounds=10,
        max_total_attempts=3000,
        retry_delay_seconds=1.0,
        timeout_seconds=60,
    )

    current_index = _load_process_index(process_json_path, len(merged_items))
    _save_process_json(
        process_path=process_json_path,
        input_json_path=input_json_path,
        output_json_path=output_json_path,
        items=merged_items,
        current_index=current_index,
    )

    return {
        "input_json_path": input_json_path,
        "output_json_path": output_json_path,
        "process_json_path": process_json_path,
        "raw_count": len(raw_items),
        "imported_count": len(normalized_items),
        "merged_count": len(merged_items),
        "skipped_no_filename": skipped_no_filename,
        "download_stats": download_stats,
        "current_index": current_index,
    }


def _load_labels_df() -> pd.DataFrame:
    if LABELS_CSV.exists():
        try:
            # 鉁?IMPORTANT: prevent empty cells from becoming NaN(float)
            df = pd.read_csv(LABELS_CSV, encoding="utf-8-sig", keep_default_na=False)
            for col in CSV_COLUMNS:
                if col not in df.columns:
                    if col in ("skipped", "abandon"):
                        df[col] = False
                    else:
                        df[col] = ""
            return df[CSV_COLUMNS].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=CSV_COLUMNS)


def _labels_index(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """filename -> row dict (for easy loading)"""
    if df.empty:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        record: Dict[str, Any] = {}
        for k in CSV_COLUMNS:
            if k in ("skipped", "abandon"):
                s = str(row.get(k, "")).strip().lower()
                record[k] = s in ("true", "1", "yes")
            else:
                record[k] = _safe_text(row.get(k, ""))
        out[str(row["filename"])] = record
    return out


def _row_to_record(row: pd.Series) -> Dict[str, Any]:
    """Convert one CSV row into app record dict."""
    record: Dict[str, Any] = {}
    for k in CSV_COLUMNS:
        if k in ("skipped", "abandon"):
            s = str(row.get(k, "")).strip().lower()
            record[k] = s in ("true", "1", "yes")
        else:
            record[k] = _safe_text(row.get(k, ""))
    return record


def _labels_media_items(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Build display items using CSV rows as the primary unit.
    This preserves multiple samples that share the same filename.
    """
    media_by_name = {p.name: p for p in _supported_media_files()}
    items: List[Dict[str, Any]] = []

    if not df.empty and "filename" in df.columns:
        for idx, row in df.iterrows():
            filename = _safe_text(row.get("filename", "")).strip()
            if not filename:
                continue
            media_path = media_by_name.get(filename)
            if media_path is None:
                continue
            items.append(
                {
                    "row_index": int(idx),
                    "path": media_path,
                    "record": _row_to_record(row),
                }
            )
        return items

    # Fallback mode: no labels CSV rows yet, use one item per media file.
    for path in sorted(media_by_name.values(), key=lambda p: p.name.lower()):
        items.append({"row_index": None, "path": path, "record": None})
    return items


def _upsert_label(df: pd.DataFrame, record: Dict[str, Any]) -> pd.DataFrame:
    filename = _safe_text(record.get("filename", "")).strip()
    rec_id = _safe_text(record.get("id", "")).strip()
    rec_text = _safe_text(record.get("input_text", "")).strip()
    if df.empty or "filename" not in df.columns:
        return pd.DataFrame([record], columns=CSV_COLUMNS)

    if "id" not in df.columns:
        df["id"] = ""
    if "input_text" not in df.columns:
        df["input_text"] = ""

    mask = (
        (df["filename"].astype(str).str.strip() == filename)
        & (df["id"].astype(str).str.strip() == rec_id)
        & (df["input_text"].astype(str).str.strip() == rec_text)
    )
    if mask.any():
        idx = df.index[mask][0]
        for k in CSV_COLUMNS:
            df.at[idx, k] = record.get(k, "" if k not in ("skipped", "abandon") else False)
        return df

    return pd.concat([df, pd.DataFrame([record], columns=CSV_COLUMNS)], ignore_index=True)


def _save_record_to_row(df: pd.DataFrame, row_index: Optional[int], record: Dict[str, Any]) -> pd.DataFrame:
    if row_index is not None and row_index in df.index:
        for k in CSV_COLUMNS:
            df.at[row_index, k] = record.get(k, "" if k not in ("skipped", "abandon") else False)
        return df
    return _upsert_label(df, record)


def _save_labels_df(df: pd.DataFrame) -> None:
    df.to_csv(LABELS_CSV, index=False, encoding="utf-8-sig")


def _rerun() -> None:
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def _init_session_state() -> None:
    default_output_path = _default_output_json_path(INPUT_JSON_DEFAULT)
    default_process_path = _process_json_path(default_output_path)
    defaults = {
        "current_index": 0,
        "is_locked": False,
        "abandon_selected": False,
        "input_json_path": str(INPUT_JSON_DEFAULT),
        "active_input_json_path": str(INPUT_JSON_DEFAULT),
        "output_json_path": str(default_output_path),
        "process_json_path": str(default_process_path),
        "last_import_summary": "",
        "last_loaded_output_path": "",
        "abandon_marked": False,
        "id": "",
        "input_text": "",  # 鉁?NEW
        "subject": "",
        "target": "",
        "subject1": "",
        "subject2": "",
        "subject3": "",
        "target1": "",
        "target2": "",
        "target3": "",
        "scenario": SCENARIO_OPTIONS[0] if SCENARIO_OPTIONS else "",
        "mechanism_Affection": "NULL",
        "mechanism_Intent": "NULL",
        "mechanism_Attitude": "NULL",
        "domain": "",
        "culture": "",
        "label_Affection": "",
        "label_Intent": "",
        "label_Attitude": "",
        "rationale": "",
        "last_loaded_sample_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    for field in EVAL_FIELDS:
        fail_key = f"failed__{field}"
        if fail_key not in st.session_state:
            st.session_state[fail_key] = False

    # Force text keys to safe strings
    _ensure_text_state(
        [
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
            "label_Intent",
            "rationale",
        ]
    )

    # Normalize choice keys
    _normalize_choice_in_state("scenario", SCENARIO_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("mechanism_Affection", MECHANISM_AFFECTION_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("mechanism_Intent", MECHANISM_INTENT_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("mechanism_Attitude", MECHANISM_ATTITUDE_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("label_Affection", Affection_OPTIONS, allow_empty=True)
    _normalize_choice_in_state("label_Intent", INTENT_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("label_Attitude", ATTITUDE_OPTIONS, allow_empty=True)


def _load_record_into_inputs(record: Optional[Dict[str, Any]]) -> None:
    """Load saved record into input fields (or clear them)."""
    if not record:
        st.session_state.abandon_selected = False
        st.session_state.id = ""
        st.session_state.input_text = ""
        st.session_state.subject = ""
        st.session_state.target = ""
        st.session_state.subject1 = ""
        st.session_state.subject2 = ""
        st.session_state.subject3 = ""
        st.session_state.target1 = ""
        st.session_state.target2 = ""
        st.session_state.target3 = ""
        st.session_state.scenario = SCENARIO_OPTIONS[0] if SCENARIO_OPTIONS else ""
        st.session_state.mechanism_Affection = "NULL"
        st.session_state.mechanism_Intent = "NULL"
        st.session_state.mechanism_Attitude = "NULL"
        st.session_state.domain = ""
        st.session_state.culture = ""
        st.session_state.label_Affection = ""
        st.session_state.label_Intent = ""
        st.session_state.label_Attitude = ""
        st.session_state.rationale = ""
        return

    # Prefer new dedicated abandon flag; fallback to old skipped for compatibility.
    st.session_state.abandon_selected = bool(record.get("abandon", record.get("skipped", False)))
    st.session_state.id = _safe_text(record.get("id", ""))
    st.session_state.input_text = _safe_text(record.get("input_text", ""))
    st.session_state.subject = _safe_text(record.get("subject", ""))
    st.session_state.target = _safe_text(record.get("target", ""))
    st.session_state.subject1 = _safe_text(record.get("subject1", ""))
    st.session_state.subject2 = _safe_text(record.get("subject2", ""))
    st.session_state.subject3 = _safe_text(record.get("subject3", ""))
    st.session_state.target1 = _safe_text(record.get("target1", ""))
    st.session_state.target2 = _safe_text(record.get("target2", ""))
    st.session_state.target3 = _safe_text(record.get("target3", ""))
    st.session_state.scenario = _safe_text(record.get("scenario", ""))
    st.session_state.mechanism_Affection = _safe_text(record.get("mechanism_Affection", ""))
    st.session_state.mechanism_Intent = _safe_text(record.get("mechanism_Intent", ""))
    st.session_state.mechanism_Attitude = _safe_text(record.get("mechanism_Attitude", ""))
    st.session_state.domain = _safe_text(record.get("domain", ""))
    st.session_state.culture = _safe_text(record.get("culture", ""))
    st.session_state.label_Affection = _safe_text(record.get("label_Affection", ""))
    st.session_state.label_Intent = _safe_text(record.get("label_Intent", ""))
    st.session_state.label_Attitude = _safe_text(record.get("label_Attitude", ""))
    st.session_state.rationale = _safe_text(record.get("rationale", ""))

    legacy_mechanism = _safe_text(record.get("mechanism", "")).strip()
    scenario_norm = _safe_text(st.session_state.scenario).strip().lower()
    if legacy_mechanism:
        if not st.session_state.mechanism_Affection and scenario_norm == "affection":
            st.session_state.mechanism_Affection = legacy_mechanism
        if not st.session_state.mechanism_Intent and scenario_norm == "intent":
            st.session_state.mechanism_Intent = legacy_mechanism
        if not st.session_state.mechanism_Attitude and scenario_norm == "attitude":
            st.session_state.mechanism_Attitude = legacy_mechanism

    _normalize_choice_in_state("scenario", SCENARIO_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("mechanism_Affection", MECHANISM_AFFECTION_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("mechanism_Intent", MECHANISM_INTENT_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("mechanism_Attitude", MECHANISM_ATTITUDE_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("label_Affection", Affection_OPTIONS, allow_empty=True)
    _normalize_choice_in_state("label_Intent", INTENT_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("label_Attitude", ATTITUDE_OPTIONS, allow_empty=True)


def _get_image_meta(image_path: Path) -> Tuple[int, int]:
    """Get image width and height (only valid for image files)."""
    try:
        with Image.open(image_path) as im:
            w, h = im.size
        return w, h
    except Exception:
        return 0, 0


def _render_media_preview(file_path: Path, frame_width: int = PREVIEW_WIDTH, frame_height: int = PREVIEW_HEIGHT) -> None:
    """Render image/video in one shared fixed-size preview frame."""
    try:
        raw = file_path.read_bytes()
    except Exception:
        st.warning(f"Failed to read media: {file_path.name}")
        return

    mime = mimetypes.guess_type(file_path.name)[0]
    if _is_image(file_path):
        mime = mime or "image/jpeg"
        media_html = (
            f'<img class="media-preview-element" src="data:{mime};base64,'
            f'{base64.b64encode(raw).decode("ascii")}" alt="{file_path.name}" />'
        )
        zoom_media_html = (
            f'<img class="media-zoom-element" src="data:{mime};base64,'
            f'{base64.b64encode(raw).decode("ascii")}" alt="{file_path.name}" />'
        )
    elif _is_video(file_path):
        # Use Streamlit native player to avoid stale DOM reuse where the video
        # frame can stay on an old source while page index changes.
        mime = mime or "video/mp4"
        st.video(raw, format=mime)
        return
    else:
        st.warning(f"Unsupported media type: {file_path.name}")
        return

    zoom_id = f"zoom_{hashlib.md5(str(file_path).encode('utf-8')).hexdigest()[:12]}"
    st.markdown(
        f"""
        <div class="media-preview-wrap">
            <input type="checkbox" id="{zoom_id}" class="media-zoom-toggle" />
            <div class="media-preview-frame" style="width:{frame_width}px;max-width:{frame_width}px;height:{frame_height}px;flex:0 0 {frame_width}px;">
                {media_html}
                <label for="{zoom_id}" class="media-zoom-btn" title="Zoom">Zoom</label>
            </div>
            <div class="media-zoom-modal">
                <label for="{zoom_id}" class="media-zoom-backdrop" aria-label="Close"></label>
                <div class="media-zoom-content">
                    {zoom_media_html}
                    <label for="{zoom_id}" class="media-zoom-close" title="Close">x</label>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _failed_key(field: str) -> str:
    return f"failed__{field}"


def _selected_failed_fields() -> List[str]:
    out: List[str] = []
    for field in EVAL_FIELDS:
        if bool(st.session_state.get(_failed_key(field), False)):
            out.append(field)
    return out


def _load_failed_checkbox_state(item: Dict[str, Any], sample_key: str) -> None:
    if st.session_state.get("last_loaded_sample_key", "") == sample_key:
        return

    evaluation = _normalize_evaluation(item.get("output", {}).get("evaluation"))
    failed_fields = set(evaluation.get("failed_fields", []))
    for field in EVAL_FIELDS:
        st.session_state[_failed_key(field)] = field in failed_fields
    abandon_raw = _parse_bool(item.get("output", {}).get("abandon"))
    st.session_state.abandon_marked = bool(abandon_raw) if abandon_raw is not None else False
    st.session_state.last_loaded_sample_key = sample_key


def _render_eval_field_box(field: str, value: Any) -> None:
    fail_key = _failed_key(field)
    failed = bool(st.session_state.get(fail_key, False))
    box_class = "eval-field-box failed" if failed else "eval-field-box"
    value_text = _safe_text(value).strip()
    if not value_text:
        value_text = "(empty)"
    value_html = html.escape(value_text).replace("\n", "<br/>")

    st.markdown(
        (
            f'<div class="{box_class}">'
            f'<div class="eval-field-label">{html.escape(field)}</div>'
            f'<div class="eval-field-value">{value_html}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_eval_field_checkbox(field: str) -> None:
    fail_key = _failed_key(field)
    st.checkbox("F", key=fail_key)


def _render_readonly_field_box(label: str, value: Any, long_text: bool = False) -> None:
    value_text = _safe_text(value).strip()
    if not value_text:
        value_text = "(empty)"
    value_html = html.escape(value_text).replace("\n", "<br/>")
    value_class = "readonly-field-value long" if long_text else "readonly-field-value"
    st.markdown(
        (
            '<div class="readonly-field-box">'
            f'<div class="readonly-field-label">{html.escape(label)}</div>'
            f'<div class="{value_class}">{value_html}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _run_json_evaluation_ui() -> None:
    import_cols = st.columns([0.75, 0.25], gap="small")
    with import_cols[0]:
        st.text_input("Input JSON/JSONL path", key="input_json_path")
    with import_cols[1]:
        start_import_clicked = st.button("start_import", use_container_width=True, type="primary")

    if start_import_clicked:
        try:
            input_json_path = Path(_safe_text(st.session_state.input_json_path)).expanduser()
            with st.spinner("Importing labels and downloading media..."):
                result = _start_import_from_json(input_json_path)

            st.session_state.active_input_json_path = str(result["input_json_path"])
            st.session_state.output_json_path = str(result["output_json_path"])
            st.session_state.process_json_path = str(result["process_json_path"])
            st.session_state.current_index = int(result["current_index"])
            st.session_state.last_loaded_output_path = ""
            st.session_state.last_loaded_sample_key = ""

            stats = result.get("download_stats", {})
            st.session_state.last_import_summary = (
                f"Imported: {result.get('imported_count', 0)} | "
                f"Merged: {result.get('merged_count', 0)} | "
                f"Skipped(no filename): {result.get('skipped_no_filename', 0)} | "
                f"Media target: {stats.get('target_files', 0)} | "
                f"Downloaded: {stats.get('downloaded_this_run', 0)} | "
                f"Existing: {stats.get('already_exists', 0)} | "
                f"Unresolved: {stats.get('unresolved_files', 0)}"
            )
            st.success("Import completed.")
        except Exception as exc:
            st.error(f"Import failed: {exc}")

    if st.session_state.get("last_import_summary"):
        st.caption(st.session_state.last_import_summary)

    output_json_path = Path(_safe_text(st.session_state.output_json_path))
    process_json_path = Path(_safe_text(st.session_state.process_json_path))
    input_json_path = Path(_safe_text(st.session_state.active_input_json_path))

    items = _load_json_items(output_json_path)
    total = len(items)
    if total == 0:
        st.warning("No records loaded. Please set input JSON and click `start_import`.")
        st.stop()

    if st.session_state.get("last_loaded_output_path", "") != str(output_json_path):
        st.session_state.current_index = _load_process_index(process_json_path, total)
        st.session_state.last_loaded_output_path = str(output_json_path)
        st.session_state.last_loaded_sample_key = ""

    current_index = int(st.session_state.current_index)
    current_index = max(0, min(current_index, total - 1))
    st.session_state.current_index = current_index
    current_item = items[current_index]
    filename, item_id, input_text = _item_key(current_item)

    sample_key = f"{current_index}|{filename}|{item_id}|{input_text}"
    _load_failed_checkbox_state(current_item, sample_key)

    completed = _count_completed(items)
    current_abandon_raw = _parse_bool(current_item.get("output", {}).get("abandon"))
    current_abandon = bool(current_abandon_raw) if current_abandon_raw is not None else False

    left, right = st.columns([0.6, 0.4], gap="small")
    with left:
        title_cols = st.columns([0.48, 0.52], gap="small")
        with title_cols[0]:
            st.markdown("**Image/Video**")
        with title_cols[1]:
            st.markdown("**ID / Input**")

        media_col, input_col = st.columns([0.48, 0.52], gap="small")
        with media_col:
            if filename:
                media_path = MEDIA_DIR / filename
                if media_path.exists():
                    _render_media_preview(
                        media_path,
                        frame_width=PREVIEW_WIDTH,
                        frame_height=PREVIEW_HEIGHT,
                    )
                else:
                    st.warning(f"Media not found locally: {filename}")
            else:
                st.warning("Missing filename for this item.")
            st.caption(f"Current file: `{filename or '(none)'}`")

        with input_col:
            _render_readonly_field_box("ID", item_id)
            _render_readonly_field_box("Input", input_text, long_text=True)

        st.divider()
        st.caption("Field columns 1-2")
        left_field_cols = st.columns([4.4, 1.0, 4.4, 1.0], gap="small")
        with left_field_cols[0]:
            for field in EVAL_COLUMN_1_FIELDS:
                _render_eval_field_box(field, current_item.get("output", {}).get(field, ""))
        with left_field_cols[1]:
            for field in EVAL_COLUMN_1_FIELDS:
                _render_eval_field_checkbox(field)
        with left_field_cols[2]:
            for field in EVAL_COLUMN_2_FIELDS:
                _render_eval_field_box(field, current_item.get("output", {}).get(field, ""))
        with left_field_cols[3]:
            for field in EVAL_COLUMN_2_FIELDS:
                _render_eval_field_checkbox(field)

    with right:
        pos_prog = (current_index + 1) / total if total else 0.0
        st.progress(pos_prog)
        st.caption(
            f"Position: {current_index + 1}/{total} | Completed(evaluated): {completed}/{total}"
        )

        jump_cols = st.columns([0.6, 0.4], gap="small")
        with jump_cols[0]:
            jump_page = st.number_input(
                "Page",
                min_value=1,
                max_value=total,
                value=current_index + 1,
                step=1,
                key="jump_page_input",
            )
        with jump_cols[1]:
            jump_clicked = st.button("Go", use_container_width=True)

        nav_cols = st.columns([1, 1, 1], gap="small")
        with nav_cols[0]:
            prev_clicked = st.button("Previous", use_container_width=True)
        with nav_cols[1]:
            next_clicked = st.button("Next", use_container_width=True)
        with nav_cols[2]:
            st.checkbox("Abandon", key="abandon_marked")

        st.caption(f"Output: `{output_json_path}`")
        st.caption("Field columns 3-4")
        right_field_cols = st.columns([4.4, 1.0, 4.4, 1.0], gap="small")
        with right_field_cols[0]:
            for field in EVAL_COLUMN_3_FIELDS:
                _render_eval_field_box(field, current_item.get("output", {}).get(field, ""))
        with right_field_cols[1]:
            for field in EVAL_COLUMN_3_FIELDS:
                _render_eval_field_checkbox(field)
        with right_field_cols[2]:
            for field in EVAL_COLUMN_4_FIELDS:
                _render_eval_field_box(field, current_item.get("output", {}).get(field, ""))
        with right_field_cols[3]:
            for field in EVAL_COLUMN_4_FIELDS:
                _render_eval_field_checkbox(field)

        failed_fields = _selected_failed_fields()
        if bool(st.session_state.abandon_marked):
            st.warning("Current item is marked as abandon.")
        if failed_fields:
            st.warning("Failed fields selected. Click Next to submit this item as failed.")

        accept_clicked = st.button(
            "Accept (All Fields Pass)",
            use_container_width=True,
            type="primary",
            disabled=bool(failed_fields) or bool(st.session_state.abandon_marked),
        )

    def _go(index: int) -> None:
        st.session_state.current_index = max(0, min(index, total - 1))
        _save_process_json(
            process_path=process_json_path,
            input_json_path=input_json_path,
            output_json_path=output_json_path,
            items=items,
            current_index=st.session_state.current_index,
        )
        _rerun()

    if jump_clicked:
        _go(int(jump_page) - 1)

    if prev_clicked:
        _go(current_index - 1)

    if next_clicked:
        failed_fields_next = _selected_failed_fields()
        abandon_selected = bool(st.session_state.abandon_marked)
        if abandon_selected:
            _apply_abandon_result(current_item)
            next_index = min(current_index + 1, total - 1)
            st.session_state.current_index = next_index
            st.session_state.last_loaded_sample_key = ""
            _save_json_items(output_json_path, items)
            _save_process_json(
                process_path=process_json_path,
                input_json_path=input_json_path,
                output_json_path=output_json_path,
                items=items,
                current_index=next_index,
            )
            _rerun()
        elif failed_fields_next:
            _apply_evaluation_result(current_item, failed_fields_next)
            next_index = min(current_index + 1, total - 1)
            st.session_state.current_index = next_index
            st.session_state.last_loaded_sample_key = ""
            _save_json_items(output_json_path, items)
            _save_process_json(
                process_path=process_json_path,
                input_json_path=input_json_path,
                output_json_path=output_json_path,
                items=items,
                current_index=next_index,
            )
            _rerun()
        else:
            if current_abandon:
                current_item.setdefault("output", {})["abandon"] = False
                current_item.setdefault("output", {})["evaluation"] = _default_evaluation()
                next_index = min(current_index + 1, total - 1)
                st.session_state.current_index = next_index
                st.session_state.last_loaded_sample_key = ""
                _save_json_items(output_json_path, items)
                _save_process_json(
                    process_path=process_json_path,
                    input_json_path=input_json_path,
                    output_json_path=output_json_path,
                    items=items,
                    current_index=next_index,
                )
                _rerun()
            else:
                _go(current_index + 1)

    if accept_clicked:
        if bool(st.session_state.abandon_marked):
            st.warning("Uncheck Abandon before accepting.")
            return
        failed_fields_accept = _selected_failed_fields()
        if failed_fields_accept:
            st.warning("Accept only supports full pass without any failed fields.")
        else:
            _apply_evaluation_result(current_item, [])
            next_index = min(current_index + 1, total - 1)
            st.session_state.current_index = next_index
            st.session_state.last_loaded_sample_key = ""
            _save_json_items(output_json_path, items)
            _save_process_json(
                process_path=process_json_path,
                input_json_path=input_json_path,
                output_json_path=output_json_path,
                items=items,
                current_index=next_index,
            )
            _rerun()


def main() -> None:
    st.set_page_config(page_title="Media Annotation Tool", layout="wide")

    st.markdown(
        """
<style>
:root {
    --preview-w: 280px;
    --preview-h: 220px;
}
/* Hide Streamlit top toolbar/header (Deploy/menu row) */
header[data-testid="stHeader"] { display: none; }

/* Adjust top padding to prevent header overlap */
.block-container { padding-top: 0.2rem; padding-bottom: 0.5rem; }
div[data-testid="stVerticalBlock"] { gap: 0.2rem; }
/* Active danger-style action button (Abandon ON). */
div[data-testid="stButton"] button[kind="primary"] {
    font-weight: 700;
    background-color: #c62828;
    border-color: #c62828;
    color: #ffffff;
}
/* Compact metadata styling */
.meta { color: rgba(49, 51, 63, 0.7); font-size: 0.85rem; margin-top: 0.2rem; }
/* Reduce spacing in form elements */
.stTextInput, .stSelectbox, .stTextArea { margin-bottom: 0.15rem; }
/* Compact subheader */
h3 { margin-top: 0.3rem; margin-bottom: 0.3rem; }
/* Shared fixed preview frame for image/video */
.media-preview-frame {
    position: relative;
    width: var(--preview-w);
    max-width: var(--preview-w);
    min-width: var(--preview-w);
    height: var(--preview-h);
    max-height: var(--preview-h);
    flex: 0 0 var(--preview-w);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0f172a;
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 8px;
    box-sizing: border-box;
}
.media-zoom-toggle {
    display: none;
}
.media-zoom-btn {
    position: absolute;
    right: 8px;
    bottom: 8px;
    z-index: 3;
    padding: 2px 8px;
    border-radius: 6px;
    background: rgba(15, 23, 42, 0.85);
    color: #ffffff;
    border: 1px solid rgba(148, 163, 184, 0.55);
    font-size: 12px;
    cursor: pointer;
    user-select: none;
}
.media-zoom-modal {
    display: none;
}
.media-zoom-toggle:checked ~ .media-zoom-modal {
    display: block;
}
.media-zoom-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.78);
    z-index: 9998;
    cursor: zoom-out;
}
.media-zoom-content {
    position: fixed;
    inset: 4vh 4vw;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
}
.media-zoom-element {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    object-fit: contain;
    background: #0f172a;
    border-radius: 8px;
    border: 1px solid rgba(148, 163, 184, 0.45);
}
.media-zoom-close {
    position: absolute;
    top: 12px;
    right: 12px;
    width: 30px;
    height: 30px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.9);
    color: #ffffff;
    border: 1px solid rgba(148, 163, 184, 0.55);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-weight: 700;
    user-select: none;
}
.media-preview-element {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    background: #0f172a;
}
/* Keep preview and ID/Input on the same row with equal height */
div[data-testid="stHorizontalBlock"]:has(.preview-col-anchor):has(.input-panel-anchor) {
    align-items: stretch;
}
div[data-testid="column"]:has(.input-panel-anchor) > div[data-testid="stVerticalBlock"] {
    height: 220px;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    min-width: 0;
}
div[data-testid="column"]:has(.input-panel-anchor) div[data-testid="stTextInput"] {
    flex: 0 0 auto;
}
div[data-testid="column"]:has(.input-panel-anchor) div[data-testid="stTextArea"] {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
}
div[data-testid="column"]:has(.input-panel-anchor) div[data-testid="stTextArea"] > div {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
}
div[data-testid="column"]:has(.input-panel-anchor) div[data-testid="stTextArea"] textarea {
    flex: 1 1 auto;
    min-height: 0 !important;
    height: 100% !important;
    overflow: auto !important;
    resize: none;
}
/* Compact right panel elements - maximize space efficiency */
div[data-testid="column"]:last-child .stSubheader { font-size: 1rem; margin-bottom: 0.2rem; margin-top: 0.2rem; }
div[data-testid="column"]:last-child .stButton button {
    padding: 0.05rem 0.2rem !important;
    font-size: 0.52rem !important;
    min-height: 20px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    width: 100%;
}
div[data-testid="column"]:last-child .stButton button * {
    font-size: 0.52rem !important;
    line-height: 1 !important;
    white-space: nowrap !important;
}
div[data-testid="column"]:last-child .stTextInput,
div[data-testid="column"]:last-child .stSelectbox,
div[data-testid="column"]:last-child .stTextArea { margin-bottom: 0.15rem; }
div[data-testid="column"]:last-child .stTextInput input,
div[data-testid="column"]:last-child .stSelectbox select,
div[data-testid="column"]:last-child .stTextArea textarea { font-size: 0.85rem; padding: 0.3rem 0.4rem; min-height: 32px; }
div[data-testid="column"]:last-child label { font-size: 0.8rem; margin-bottom: 0.1rem; }
div[data-testid="column"]:last-child [data-testid="stProgress"] { margin-bottom: 0.15rem; height: 0.5rem; display: block !important; visibility: visible !important; }
div[data-testid="column"]:last-child [data-testid="stProgress"] > div { height: 0.5rem !important; }
div[data-testid="column"]:last-child .stCaption { font-size: 0.75rem; margin-top: 0.1rem; }
div[data-testid="column"]:last-child .stMarkdown { margin-bottom: 0.2rem; }
div[data-testid="column"]:last-child .stDivider { margin: 0.3rem 0; }
div[data-testid="column"]:last-child [data-testid="stVerticalBlock"] { gap: 0.2rem; }
.eval-field-box {
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: #f8fafc;
    padding: 0.24rem 0.34rem;
    min-height: 52px;
}
.eval-field-box.failed {
    border-color: #dc2626;
    background: #fee2e2;
}
.eval-field-label {
    font-size: 0.64rem;
    font-weight: 700;
    color: #374151;
    margin-bottom: 0.05rem;
}
.eval-field-value {
    font-size: 0.66rem;
    color: #111827;
    line-height: 1.05;
    max-height: 1.65rem;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}
.readonly-field-box {
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: #f8fafc;
    padding: 0.36rem 0.5rem;
    min-height: 56px;
    margin-bottom: 0.18rem;
}
.readonly-field-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #374151;
    margin-bottom: 0.1rem;
}
.readonly-field-value {
    font-size: 0.78rem;
    color: #111827;
    line-height: 1.15;
    max-height: 2.4rem;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}
.readonly-field-value.long {
    max-height: 7.2rem;
}
div[data-testid="stButton"] button[kind="primary"] {
    min-height: 46px !important;
    font-size: 0.92rem !important;
}
div[data-testid="stButton"] button {
    min-height: 28px !important;
    font-size: 0.72rem !important;
}
div[data-testid="stCheckbox"] label p {
    font-size: 0.64rem !important;
}
div[data-testid="stCheckbox"] {
    margin-top: 0.25rem;
    margin-bottom: 0.1rem;
}
div[data-testid="stCheckbox"] label {
    min-height: 1.2rem !important;
}
div[data-testid="column"] [data-testid="stNumberInput"] input {
    min-height: 30px !important;
    font-size: 0.78rem !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    # 鉁?AUTO-CLEAR bad persisted widget state (must be before widgets are created)
    _clear_bad_widget_state(
        [
            "input_text",
            "subject",
            "target",
            "subject1",
            "subject2",
            "subject3",
            "target1",
            "target2",
            "target3",
            "label_Intent",
            "rationale",
        ]
    )

    _init_session_state()
    _run_json_evaluation_ui()
    return
    is_locked = bool(st.session_state.is_locked)

    labels_df = _load_labels_df()
    sample_items = _labels_media_items(labels_df)
    total = len(sample_items)
    if total == 0:
        st.warning(
            "No media files found for current labels. Please run import to download files into `images/`."
        )
        st.stop()

    # ====== Two-column layout: Left 70% / Right 30% ======
    left, right = st.columns([0.6, 0.4], gap="small")

    current_index = int(st.session_state.current_index)
    current_index = max(0, min(current_index, total - 1))
    st.session_state.current_index = current_index
    current_item = sample_items[current_index]
    current_row_index = current_item.get("row_index")
    current_path = current_item["path"]
    current_record = current_item.get("record")

    # Auto-load previous annotations when switching sample item.
    current_sample_key = (
        f"{current_row_index}|{current_path.name}|"
        f"{_safe_text((current_record or {}).get('id', ''))}|"
        f"{_safe_text((current_record or {}).get('input_text', ''))}"
    )
    if st.session_state.last_loaded_sample_key != current_sample_key:
        _load_record_into_inputs(current_record)
        st.session_state.last_loaded_sample_key = current_sample_key

    # 鉁?Final safety: make sure text keys are strings right before rendering widgets
    _ensure_text_state(
        [
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
            "label_Intent",
            "rationale",
        ]
    )
    # Ensure selectbox state values exist in options (avoids ValueError in Streamlit)
    _normalize_choice_in_state("label_Affection", Affection_OPTIONS, allow_empty=True)
    _normalize_choice_in_state("label_Intent", INTENT_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("label_Attitude", ATTITUDE_OPTIONS, allow_empty=True)

    # =========================
    # Left Column: Media Display + (RED BOX AREA) Input + Mechanism/Domain/Culture/Rationale
    # =========================
    with left:
        # Add top spacer so media/input block sits lower on the page.
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

        title_col_left, title_col_right = st.columns([0.48, 0.52], gap="medium")
        with title_col_left:
            st.markdown("**Image/Video**")
        with title_col_right:
            st.markdown("**ID**")

        # This creates the red-box area at the right of the media
        media_col, input_col = st.columns([0.48, 0.52], gap="medium")

        with media_col:
            st.markdown('<div class="preview-col-anchor"></div>', unsafe_allow_html=True)
            # Display media based on file type
            if _is_image(current_path):
                _render_media_preview(current_path, frame_width=PREVIEW_WIDTH, frame_height=PREVIEW_HEIGHT)
                w, h = _get_image_meta(current_path)
                if w > 0 and h > 0:
                    st.markdown(
                        f'<div class="meta">File: <b>{current_path.name}</b> | Size: <b>{w}脳{h}</b> | '
                        f'Index: <b>{current_index + 1}/{total}</b></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="meta">File: <b>{current_path.name}</b> | Index: <b>{current_index + 1}/{total}</b></div>',
                        unsafe_allow_html=True,
                    )
            elif _is_video(current_path):
                _render_media_preview(current_path, frame_width=PREVIEW_WIDTH, frame_height=PREVIEW_HEIGHT)
                st.markdown(
                    f'<div class="meta">File: <b>{current_path.name}</b> | Type: <b>Video</b> | '
                    f'Index: <b>{current_index + 1}/{total}</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="meta">File: <b>{current_path.name}</b> | Index: <b>{current_index + 1}/{total}</b></div>',
                    unsafe_allow_html=True,
                )

        with input_col:
            st.markdown('<div class="input-panel-anchor"></div>', unsafe_allow_html=True)
            # 鉁?Requirement: place Input in the red-box area (title + box like Subject)
            st.text_input("ID", key="id", label_visibility="collapsed", disabled=is_locked)
            st.text_area("Input", key="input_text", disabled=is_locked)

        st.divider()
        lower_left_col, lower_right_col = st.columns([0.42, 0.58], gap="medium")
        with lower_left_col:
            st.selectbox("Mechanism: Affection", MECHANISM_AFFECTION_OPTIONS, key="mechanism_Affection", disabled=is_locked)
            st.selectbox("Mechanism: Intent", MECHANISM_INTENT_OPTIONS, key="mechanism_Intent", disabled=is_locked)
            st.selectbox("Mechanism: Attitude", MECHANISM_ATTITUDE_OPTIONS, key="mechanism_Attitude", disabled=is_locked)
            st.text_input("Domain", key="domain", disabled=is_locked)
            st.text_input("Culture", key="culture", disabled=is_locked)
        with lower_right_col:
            st.text_area("Rationale", key="rationale", height=120, disabled=is_locked)
            st.text_input("Subject1", key="subject1", disabled=is_locked)
            st.text_input("Subject2", key="subject2", disabled=is_locked)
            st.text_input("Subject3", key="subject3", disabled=is_locked)
            st.text_input("Target1", key="target1", disabled=is_locked)
            st.text_input("Target2", key="target2", disabled=is_locked)
            st.text_input("Target3", key="target3", disabled=is_locked)

    # =========================
    # Right Column: Progress + Navigation + Form
    # =========================
    with right:
        with st.container():
            # Position progress: follows page navigation (Previous/Pending/Accept/Abandon).
            pos_prog = (current_index + 1) / total if total else 0.0
            st.progress(pos_prog)
            st.caption(f"Progress: {current_index + 1}/{total}")

            jump_cols = st.columns([0.6, 0.4], gap="small")
            with jump_cols[0]:
                jump_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total,
                    value=current_index + 1,
                    step=1,
                    key="jump_page_input",
                )
            with jump_cols[1]:
                jump_clicked = st.button("Go", use_container_width=True)

            nav_cols = st.columns([1, 1, 1, 1], gap="small")
            with nav_cols[0]:
                prev_clicked = st.button("Previous", use_container_width=True)
            with nav_cols[1]:
                accept_clicked = st.button("Accept", use_container_width=True)
            with nav_cols[2]:
                pending_clicked = st.button("Pending", use_container_width=True)
            with nav_cols[3]:
                abandon_clicked = st.button(
                    "Abandon",
                    use_container_width=True,
                    type="primary" if st.session_state.abandon_selected else "secondary",
                )
            lock_toggle_clicked = st.button(
                "Unlock Edit" if st.session_state.is_locked else "Lock Edit",
                use_container_width=True,
                type="primary" if not st.session_state.is_locked else "secondary",
            )
        st.caption(
            f"Edit: {'Locked' if st.session_state.is_locked else 'Unlocked'} | "
            f"Abandon: {'ON' if st.session_state.abandon_selected else 'OFF'}"
        )

        st.divider()

        with st.container():
            st.markdown("**Annotation Form**")

            st.text_input("Subject", key="subject", disabled=is_locked)
            st.text_input("Target", key="target", disabled=is_locked)
            st.selectbox("Scenario", SCENARIO_OPTIONS, key="scenario", disabled=is_locked)

            st.selectbox("Label: Affection", Affection_OPTIONS, key="label_Affection", disabled=is_locked)
            st.selectbox("Label: Intent", INTENT_OPTIONS, key="label_Intent", disabled=is_locked)
            st.selectbox("Label: Attitude", ATTITUDE_OPTIONS, key="label_Attitude", disabled=is_locked)

        st.caption(f"Current: `{current_path.name}`")

    # =========================
    # Event handling: Previous / Save & Next / Skip
    # =========================
    def _go(index: int) -> None:
        st.session_state.current_index = max(0, min(index, total - 1))
        _rerun()

    def _next_index(from_idx: int) -> int:
        return min(from_idx + 1, total - 1)

    if jump_clicked:
        _go(int(jump_page) - 1)

    if lock_toggle_clicked:
        st.session_state.is_locked = not bool(st.session_state.is_locked)
        _rerun()

    if prev_clicked:
        _go(current_index - 1)

    if abandon_clicked:
        st.session_state.abandon_selected = not bool(st.session_state.abandon_selected)
        scenario_norm = _safe_text(st.session_state.scenario).strip().lower()
        mechanism_affection = _safe_text(st.session_state.mechanism_Affection).strip() or "NULL"
        mechanism_intent = _safe_text(st.session_state.mechanism_Intent).strip() or "NULL"
        mechanism_attitude = _safe_text(st.session_state.mechanism_Attitude).strip() or "NULL"
        mechanism_generic = "NULL"
        if scenario_norm == "affection":
            mechanism_generic = mechanism_affection
            mechanism_intent = "NULL"
            mechanism_attitude = "NULL"
        elif scenario_norm == "intent":
            mechanism_generic = mechanism_intent
            mechanism_affection = "NULL"
            mechanism_attitude = "NULL"
        elif scenario_norm == "attitude":
            mechanism_generic = mechanism_attitude
            mechanism_affection = "NULL"
            mechanism_intent = "NULL"
        record = {
            "filename": current_path.name,
            "id": st.session_state.id,
            "input_text": st.session_state.input_text,
            "subject": st.session_state.subject,
            "target": st.session_state.target,
            "subject1": st.session_state.subject1,
            "subject2": st.session_state.subject2,
            "subject3": st.session_state.subject3,
            "target1": st.session_state.target1,
            "target2": st.session_state.target2,
            "target3": st.session_state.target3,
            "scenario": st.session_state.scenario,
            "mechanism_Affection": mechanism_affection,
            "mechanism_Intent": mechanism_intent,
            "mechanism_Attitude": mechanism_attitude,
            # Backward-compatible generic field derived from scenario-specific mechanism.
            "mechanism": mechanism_generic or "NULL",
            "domain": st.session_state.domain,
            "culture": st.session_state.culture,
            "label_Affection": st.session_state.label_Affection,
            "label_Intent": st.session_state.label_Intent,
            "label_Attitude": st.session_state.label_Attitude,
            "rationale": st.session_state.rationale,
            # Keep old skipped field untouched to avoid destructive overwrite semantics.
            "skipped": bool((current_record or {}).get("skipped", False)),
            # Dedicated abandon state for UI/export filtering.
            "abandon": bool(st.session_state.abandon_selected),
        }
        labels_df = _save_record_to_row(labels_df, current_row_index, record)
        _save_labels_df(labels_df)
        _rerun()

    if pending_clicked:
        _go(_next_index(current_index))

    if accept_clicked:
        scenario_norm = _safe_text(st.session_state.scenario).strip().lower()
        mechanism_affection = _safe_text(st.session_state.mechanism_Affection).strip() or "NULL"
        mechanism_intent = _safe_text(st.session_state.mechanism_Intent).strip() or "NULL"
        mechanism_attitude = _safe_text(st.session_state.mechanism_Attitude).strip() or "NULL"
        mechanism_generic = "NULL"
        if scenario_norm == "affection":
            mechanism_generic = mechanism_affection
            mechanism_intent = "NULL"
            mechanism_attitude = "NULL"
        elif scenario_norm == "intent":
            mechanism_generic = mechanism_intent
            mechanism_affection = "NULL"
            mechanism_attitude = "NULL"
        elif scenario_norm == "attitude":
            mechanism_generic = mechanism_attitude
            mechanism_affection = "NULL"
            mechanism_intent = "NULL"
        record = {
            "filename": current_path.name,
            "id": st.session_state.id,
            "input_text": st.session_state.input_text,
            "subject": st.session_state.subject,
            "target": st.session_state.target,
            "subject1": st.session_state.subject1,
            "subject2": st.session_state.subject2,
            "subject3": st.session_state.subject3,
            "target1": st.session_state.target1,
            "target2": st.session_state.target2,
            "target3": st.session_state.target3,
            "scenario": st.session_state.scenario,
            "mechanism_Affection": mechanism_affection,
            "mechanism_Intent": mechanism_intent,
            "mechanism_Attitude": mechanism_attitude,
            # Backward-compatible generic field derived from scenario-specific mechanism.
            "mechanism": mechanism_generic or "NULL",
            "domain": st.session_state.domain,
            "culture": st.session_state.culture,
            "label_Affection": st.session_state.label_Affection,
            "label_Intent": st.session_state.label_Intent,
            "label_Attitude": st.session_state.label_Attitude,
            "rationale": st.session_state.rationale,
            # Keep compatibility column stable; abandon state is stored independently.
            "skipped": bool((current_record or {}).get("skipped", False)),
            "abandon": bool(st.session_state.abandon_selected),
        }
        labels_df = _save_record_to_row(labels_df, current_row_index, record)
        _save_labels_df(labels_df)
        st.session_state.last_loaded_sample_key = ""  # Force reload on next item
        _go(_next_index(current_index))


if __name__ == "__main__":
    main()
