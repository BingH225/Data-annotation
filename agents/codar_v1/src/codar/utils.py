from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def clip_text(text: str, max_len: int = 2000) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if math.isnan(v):
        return lo
    return max(lo, min(hi, v))


def ensure_list_of_str(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    return [str(value)]


def normalize_token(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def map_to_closed_set(candidate: str, allowed: Iterable[str], default: str) -> str:
    allowed_list = list(allowed)
    norm_allowed = {normalize_token(x): x for x in allowed_list}
    key = normalize_token(candidate)
    if key in norm_allowed:
        return norm_allowed[key]
    for k, v in norm_allowed.items():
        if key in k or k in key:
            return v
    return default


def parse_first_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty model output")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in model output")
    depth = 0
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise ValueError("unclosed JSON object in model output")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON is not an object")
    return obj

