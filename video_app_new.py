from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from PIL import Image


# =========================
# Configuration
# =========================
MEDIA_DIR = Path("images")  # Directory containing images and videos
LABELS_CSV = Path(r"D:\NUS\ACMm\Data-annotation\Task\02_primary_assignments\day07\Yu_primary_day07.csv")
PREVIEW_WIDTH = 280
PREVIEW_HEIGHT = 220

SITUATION_OPTIONS = ["Affection", "Intent", "Attitude"]
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
    "Conflict Mitigation",
    "Intimidation",
    "Hate Humor",
    "Humiliation for Amusement",
    "Public Humiliation",
    "Meme-based Mockery",
    "Dominance Assertion",
    "Moral Condemnation",
]
ATTITUDE_OPTIONS = [
    "NULL",
    "Supportive", "Appreciative", "Sympathetic", "Neutral", "Indifferent",
    "Disapproving", "Skeptical", "Concerned", "Dismissive", "Contemptuous", "Hostile"
]


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
    if isinstance(v, float) and pd.isna(v):
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
    defaults = {
        "current_index": 0,
        "is_locked": False,
        "abandon_selected": False,
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
        "situation": SITUATION_OPTIONS[0] if SITUATION_OPTIONS else "",
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
    _normalize_choice_in_state("situation", SITUATION_OPTIONS, allow_empty=False)
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
        st.session_state.situation = SITUATION_OPTIONS[0] if SITUATION_OPTIONS else ""
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
    st.session_state.situation = _safe_text(record.get("situation", ""))
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
    situation_norm = _safe_text(st.session_state.situation).strip().lower()
    if legacy_mechanism:
        if not st.session_state.mechanism_Affection and situation_norm == "affection":
            st.session_state.mechanism_Affection = legacy_mechanism
        if not st.session_state.mechanism_Intent and situation_norm == "intent":
            st.session_state.mechanism_Intent = legacy_mechanism
        if not st.session_state.mechanism_Attitude and situation_norm == "attitude":
            st.session_state.mechanism_Attitude = legacy_mechanism

    _normalize_choice_in_state("situation", SITUATION_OPTIONS, allow_empty=False)
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
            st.selectbox("Situation", SITUATION_OPTIONS, key="situation", disabled=is_locked)

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
        situation_norm = _safe_text(st.session_state.situation).strip().lower()
        mechanism_affection = _safe_text(st.session_state.mechanism_Affection).strip() or "NULL"
        mechanism_intent = _safe_text(st.session_state.mechanism_Intent).strip() or "NULL"
        mechanism_attitude = _safe_text(st.session_state.mechanism_Attitude).strip() or "NULL"
        mechanism_generic = "NULL"
        if situation_norm == "affection":
            mechanism_generic = mechanism_affection
            mechanism_intent = "NULL"
            mechanism_attitude = "NULL"
        elif situation_norm == "intent":
            mechanism_generic = mechanism_intent
            mechanism_affection = "NULL"
            mechanism_attitude = "NULL"
        elif situation_norm == "attitude":
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
            "situation": st.session_state.situation,
            "mechanism_Affection": mechanism_affection,
            "mechanism_Intent": mechanism_intent,
            "mechanism_Attitude": mechanism_attitude,
            # Backward-compatible generic field derived from situation-specific mechanism.
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
        situation_norm = _safe_text(st.session_state.situation).strip().lower()
        mechanism_affection = _safe_text(st.session_state.mechanism_Affection).strip() or "NULL"
        mechanism_intent = _safe_text(st.session_state.mechanism_Intent).strip() or "NULL"
        mechanism_attitude = _safe_text(st.session_state.mechanism_Attitude).strip() or "NULL"
        mechanism_generic = "NULL"
        if situation_norm == "affection":
            mechanism_generic = mechanism_affection
            mechanism_intent = "NULL"
            mechanism_attitude = "NULL"
        elif situation_norm == "intent":
            mechanism_generic = mechanism_intent
            mechanism_affection = "NULL"
            mechanism_attitude = "NULL"
        elif situation_norm == "attitude":
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
            "situation": st.session_state.situation,
            "mechanism_Affection": mechanism_affection,
            "mechanism_Intent": mechanism_intent,
            "mechanism_Attitude": mechanism_attitude,
            # Backward-compatible generic field derived from situation-specific mechanism.
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
