from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from ..backends.base import BaseBackend
from ..media import MediaResolver
from ..prompting import PromptStore
from ..types import SampleInput, StageArtifact
from .common import build_media_items, run_prompt_json


class ExplicitPerceptionAgent:
    def __init__(
        self,
        backend: BaseBackend,
        prompt_store: PromptStore,
        media_resolver: MediaResolver,
        max_retries: int,
        max_frames: int,
    ):
        self.backend = backend
        self.prompt_store = prompt_store
        self.media_resolver = media_resolver
        self.max_retries = max_retries
        self.max_frames = max_frames

    @staticmethod
    def _normalize_non_empty(value: Any, fallback: str) -> str:
        text = str(value or "").strip()
        return text if text else fallback

    @staticmethod
    def _speaker_hint(text: str) -> str:
        lowered = text.lower()
        if re.search(r"\b(i|i'm|ive|i've|me|my|mine|we|our|us)\b", lowered):
            return "speaker"
        return "unknown_subject"

    @staticmethod
    def _is_placeholder(value: Any) -> bool:
        text = str(value or "").strip().lower()
        if not text:
            return True
        return (
            text.startswith("unspecified")
            or text.startswith("unknown_")
            or text in {"unknown", "n/a", "none", "null"}
        )

    @staticmethod
    def _simple_text_parser(text: str) -> Dict[str, str]:
        raw = str(text or "").strip()
        lowered = raw.lower()
        cleaned = re.sub(r"\s+", " ", raw).strip()
        clean_no_punct = re.sub(r"[^\w\s']", " ", lowered)
        tokens = [t for t in clean_no_punct.split() if t]

        if not tokens:
            return {
                "subject": "speaker",
                "object": "mentioned_entity",
                "predicate": "states",
                "attribute": "literal_text",
                "adverbial": "in_context",
            }

        subject = "speaker"
        if re.search(r"\byour boss\b", lowered):
            subject = "your boss"
        elif re.search(r"\b(i|i'm|ive|i've|me|my|mine|we|our|us)\b", lowered):
            subject = "speaker"
        else:
            m = re.search(r"^(?:when|if|as|while)\s+([^,?.!]+?)\s+(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should|calls?|asks?|wants?|tries?|tried|found|goes?|went|walks?)\b", lowered)
            if m:
                subject = m.group(1).strip()
            else:
                m2 = re.search(r"^([^,?.!]{1,36}?)\s+(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", lowered)
                subject = m2.group(1).strip() if m2 else " ".join(tokens[:2])
        if subject in {"it", "this", "that"}:
            subject = "speaker"

        predicate = "states"
        verb_match = re.search(
            r"\b(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should|calls?|asks?|wants?|tries?|tried|found|goes?|went|walks?|say|says|said|show|shows|shown|make|makes|made|feel|feels|felt)\b",
            lowered,
        )
        if verb_match:
            predicate = verb_match.group(1)
        elif len(tokens) >= 2:
            predicate = tokens[1]

        object_ = "mentioned_entity"
        obj_match = re.search(rf"\b{re.escape(predicate)}\b\s+([^,?.!]+)", lowered)
        if obj_match:
            object_ = obj_match.group(1).strip()
        elif len(tokens) >= 3:
            object_ = " ".join(tokens[2:5])

        adverbial = "in_context"
        adv_match = re.search(r"\b(when|while|because|if|before|after|on|in|at)\b\s+([^,?.!]+)", lowered)
        if adv_match:
            adverbial = f"{adv_match.group(1)} {adv_match.group(2).strip()}"

        attribute = "literal_text"
        attr_match = re.search(r"(?:\"|')([^\"']+)(?:\"|')", cleaned)
        if attr_match:
            attribute = attr_match.group(1).strip()
        elif re.search(r"\b(fuck|wtf|idiot|stupid|gross|hate|angry|sad|afraid|scared)\b", lowered):
            attribute = "strong_lexical_marker"

        return {
            "subject": subject[:80] or "speaker",
            "object": object_[:120] or "mentioned_entity",
            "predicate": predicate[:60] or "states",
            "attribute": attribute[:120] or "literal_text",
            "adverbial": adverbial[:120] or "in_context",
        }

    def _fill_defaults(self, out: Dict[str, Any], sample_text: str) -> Dict[str, Any]:
        out = out or {}
        out.setdefault("text_components", {})
        out.setdefault("image_action", {})
        out.setdefault("audio_caption", {})
        parsed = self._simple_text_parser(sample_text)
        speaker_hint = self._speaker_hint(sample_text)
        text_fallbacks = {
            "subject": parsed["subject"] if parsed["subject"] != "unknown_subject" else speaker_hint,
            "object": parsed["object"],
            "predicate": parsed["predicate"],
            "attribute": parsed["attribute"],
            "adverbial": parsed["adverbial"],
        }
        audio_fallbacks = {
            "subject": text_fallbacks["subject"],
            "object": text_fallbacks["object"],
            "predicate": text_fallbacks["predicate"],
            "attribute": text_fallbacks["attribute"],
            "adverbial": text_fallbacks["adverbial"],
        }
        image_fallbacks = {
            "subject": "visible_subject",
            "background": "visible_background",
            "behavior": "visible_behavior",
            "action": "visible_action",
        }
        for k in ("subject", "object", "predicate", "attribute", "adverbial"):
            cur_text = out["text_components"].get(k, "")
            cur_audio = out["audio_caption"].get(k, "")
            if self._is_placeholder(cur_text):
                cur_text = text_fallbacks[k]
            if self._is_placeholder(cur_audio):
                cur_audio = audio_fallbacks[k]
            out["text_components"][k] = self._normalize_non_empty(cur_text, text_fallbacks[k])
            out["audio_caption"][k] = self._normalize_non_empty(cur_audio, audio_fallbacks[k])
        for k in ("subject", "background", "behavior", "action"):
            out["image_action"][k] = self._normalize_non_empty(
                out["image_action"].get(k, ""), image_fallbacks[k]
            )
        return out

    def run(self, sample: SampleInput, scenario: str) -> Tuple[Dict[str, Any], Dict[str, Any], StageArtifact]:
        media_manifest = self.media_resolver.resolve(sample.sample_id, scenario, sample.media)
        vars_ = {
            "scenario": scenario,
            "text": sample.text,
            "media_manifest": media_manifest,
            "audio_caption_raw": media_manifest.get("audio_caption", ""),
        }
        media_items = build_media_items(media_manifest, max_frames=self.max_frames)
        out, artifact = run_prompt_json(
            backend=self.backend,
            prompt_store=self.prompt_store,
            stage_id="S1",
            prompt_id="P1_explicit_perception",
            prompt_vars=vars_,
            max_retries=self.max_retries,
            media_items=media_items,
            temperature_override=0.0,
        )
        out = self._fill_defaults(out, sample.text)
        artifact.output = out
        return out, media_manifest, artifact
