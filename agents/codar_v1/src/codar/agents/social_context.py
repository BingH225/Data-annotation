from __future__ import annotations

from typing import Any, Dict, Tuple

from ..backends.base import BaseBackend
from ..prompting import PromptStore
from ..types import SampleInput, StageArtifact
from .common import run_prompt_json


class SocialContextAgent:
    def __init__(self, backend: BaseBackend, prompt_store: PromptStore, max_retries: int):
        self.backend = backend
        self.prompt_store = prompt_store
        self.max_retries = max_retries

    @staticmethod
    def _fill_defaults(out: Dict[str, Any]) -> Dict[str, Any]:
        out = out or {}
        out.setdefault("entities", [])
        out.setdefault("relations", [])
        out.setdefault("culture_clues", [])
        out.setdefault("domain_notes", "")
        return out

    def run(self, sample: SampleInput, scenario: str, perception_json: Dict[str, Any]) -> Tuple[Dict[str, Any], StageArtifact]:
        vars_ = {
            "scenario": scenario,
            "text": sample.text,
            "perception_json": perception_json,
            "subject_options": sample.subject_options,
            "target_options": sample.target_options,
            "diversity": sample.diversity,
        }
        out, artifact = run_prompt_json(
            backend=self.backend,
            prompt_store=self.prompt_store,
            stage_id="S2",
            prompt_id="P2_social_context",
            prompt_vars=vars_,
            max_retries=self.max_retries,
            media_items=None,
        )
        out = self._fill_defaults(out)
        artifact.output = out
        return out, artifact

