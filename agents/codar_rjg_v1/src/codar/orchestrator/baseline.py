from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from ..agents.common import build_media_items, run_prompt_json
from ..constants import VALID_LABELS, VALID_MECHANISMS
from ..media import MediaResolver
from ..prompting import PromptStore
from ..types import SampleInput
from ..utils import clamp, clip_text, map_to_closed_set
from ..backends.base import BaseBackend


class DirectBaselineRunner:
    def __init__(
        self,
        backend: BaseBackend,
        prompt_store: PromptStore,
        media_resolver: MediaResolver,
        max_stage_retries: int,
        max_video_frames: int = 4,
        temperature: float = 0.0,
    ):
        self.backend = backend
        self.prompt_store = prompt_store
        self.media_resolver = media_resolver
        self.max_stage_retries = int(max_stage_retries)
        self.max_video_frames = int(max_video_frames)
        self.temperature = float(temperature)

    @staticmethod
    def _coerce_entity(raw_value: Any, options: List[str]) -> str:
        text = str(raw_value or "").strip()
        if not options:
            return text
        fallback = options[0]
        if not text:
            return fallback
        return map_to_closed_set(text, options, fallback)

    @staticmethod
    def _coerce_closed(raw_value: Any, allowed: List[str]) -> str:
        if not allowed:
            return str(raw_value or "").strip()
        fallback = allowed[0]
        text = str(raw_value or "").strip()
        if not text:
            return fallback
        return map_to_closed_set(text, allowed, fallback)

    @staticmethod
    def _coerce_confidence(raw_value: Any) -> float:
        try:
            return clamp(float(raw_value), 0.0, 1.0)
        except Exception:
            return 0.0

    def run_sample(self, sample: SampleInput, backend_meta: Dict[str, Any]) -> Dict[str, Any]:
        stage_artifacts: List[Dict[str, Any]] = []
        media_manifest = self.media_resolver.resolve(sample_id=sample.sample_id, scenario=sample.scenario, media=sample.media)
        media_items = build_media_items(media_manifest, max_frames=self.max_video_frames)
        prompt_vars = {
            "scenario": sample.scenario,
            "text": sample.text,
            "subject_options": sample.subject_options,
            "target_options": sample.target_options,
            "valid_mechanisms": VALID_MECHANISMS.get(sample.scenario, []),
            "valid_labels": VALID_LABELS.get(sample.scenario, []),
            "audio_caption": media_manifest.get("audio_caption", ""),
        }
        parsed: Dict[str, Any] = {}
        try:
            parsed, artifact = run_prompt_json(
                backend=self.backend,
                prompt_store=self.prompt_store,
                stage_id="B0",
                prompt_id="PB0_baseline_direct",
                prompt_vars=prompt_vars,
                max_retries=self.max_stage_retries,
                media_items=media_items,
                temperature_override=self.temperature,
            )
            stage_artifacts.append(asdict(artifact))
            if artifact.status != "ok":
                raise RuntimeError(artifact.notes or "baseline stage failed")

            final_prediction = {
                "subject": self._coerce_entity(parsed.get("subject"), sample.subject_options),
                "target": self._coerce_entity(parsed.get("target"), sample.target_options),
                "mechanism": self._coerce_closed(parsed.get("mechanism"), VALID_MECHANISMS.get(sample.scenario, [])),
                "label": self._coerce_closed(parsed.get("label"), VALID_LABELS.get(sample.scenario, [])),
                "confidence": self._coerce_confidence(parsed.get("confidence")),
                "decision_rationale_short": clip_text(
                    str(parsed.get("decision_rationale_short") or parsed.get("rationale") or ""),
                    300,
                ),
            }
            return {
                "sample_id": sample.sample_id,
                "scenario": sample.scenario,
                "final_prediction": final_prediction,
                "stage_artifacts": stage_artifacts,
                "backend_meta": backend_meta,
                "trace": {"mode": "direct_baseline", "media_manifest": media_manifest},
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - integration-path failure handling
            return {
                "sample_id": sample.sample_id,
                "scenario": sample.scenario,
                "final_prediction": {
                    "subject": sample.subject_options[0] if sample.subject_options else "",
                    "target": sample.target_options[0] if sample.target_options else "",
                    "mechanism": (VALID_MECHANISMS.get(sample.scenario) or [""])[0],
                    "label": (VALID_LABELS.get(sample.scenario) or [""])[0],
                    "confidence": 0.0,
                    "decision_rationale_short": "",
                },
                "stage_artifacts": stage_artifacts,
                "backend_meta": backend_meta,
                "trace": {"mode": "direct_baseline", "media_manifest": media_manifest, "raw_output": parsed},
                "error": clip_text(str(exc), 600),
            }

