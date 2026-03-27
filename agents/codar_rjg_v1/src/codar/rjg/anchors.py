from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List

from ..types import SampleInput

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")

_AGGRESSION_CUES = [
    "fuck",
    "wtf",
    "idiot",
    "moron",
    "bitch",
    "shut up",
    "go to hell",
    "faggot",
    "kill you",
]
_SOFTEN_CUES = [
    "just kidding",
    "no offense",
    "don't worry",
    "all good",
    "sorry",
    "i guess",
    "maybe",
    "not sure",
]
_CONTRAST_CUES = ["but", "yet", "however", "in real life", "on social media", "while", "vs"]


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _contains_any(text: str, cues: Iterable[str]) -> int:
    lowered = (text or "").lower()
    return sum(1 for cue in cues if cue in lowered)


def _best_option_hit(text: str, options: List[str]) -> str:
    lowered = (text or "").lower()
    for opt in options:
        key = str(opt or "").strip().lower()
        if key and key in lowered:
            return str(opt)
    return str(options[0]) if options else ""


def build_anchor_payload(sample: SampleInput, media_manifest: Dict[str, Any]) -> Dict[str, Any]:
    text = str(sample.text or "")
    audio_caption = str((media_manifest or {}).get("audio_caption", "") or "")
    merged = f"{text}\n{audio_caption}".strip()
    tokens = _tokenize(merged)
    token_freq = Counter(tokens)
    keyword_tokens = [t for t, _ in token_freq.most_common(32) if len(t) > 2]

    aggression_hits = _contains_any(merged, _AGGRESSION_CUES)
    soften_hits = _contains_any(merged, _SOFTEN_CUES)
    contrast_hits = _contains_any(merged, _CONTRAST_CUES)
    subject_anchor = _best_option_hit(merged, sample.subject_options)
    target_anchor = _best_option_hit(merged, sample.target_options)

    text_components = {
        "subject": subject_anchor or "speaker",
        "object": target_anchor or "listener",
        "predicate": keyword_tokens[0] if keyword_tokens else "statement",
        "attribute": keyword_tokens[1] if len(keyword_tokens) > 1 else "tone_unspecified",
        "adverbial": "directly" if aggression_hits > 0 else ("indirectly" if contrast_hits > 0 else "plainly"),
    }
    audio_components = {
        "subject": "speaker",
        "object": "utterance",
        "predicate": "describes",
        "attribute": "aggressive" if aggression_hits else ("softening" if soften_hits else "neutral"),
        "adverbial": "emotionally",
    }

    return {
        "scenario": sample.scenario,
        "subject_anchor": subject_anchor,
        "target_anchor": target_anchor,
        "aggression_hits": aggression_hits,
        "soften_hits": soften_hits,
        "contrast_hits": contrast_hits,
        "keyword_tokens": keyword_tokens,
        "token_freq": dict(token_freq),
        "text_components": text_components,
        "audio_components": audio_components,
        "parser_non_empty": all(bool(str(v).strip()) for v in text_components.values())
        and all(bool(str(v).strip()) for v in audio_components.values()),
    }
