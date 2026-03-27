from __future__ import annotations

from typing import Any, Dict, Tuple

from ..backends.base import BaseBackend
from ..prompting import PromptStore
from ..types import SampleInput, StageArtifact
from .common import run_prompt_json


class ExpectationAgent:
    def __init__(self, backend: BaseBackend, prompt_store: PromptStore, max_retries: int):
        self.backend = backend
        self.prompt_store = prompt_store
        self.max_retries = max_retries

    def run(
        self,
        sample: SampleInput,
        scenario: str,
        perception_json: Dict[str, Any],
        context_graph: Dict[str, Any],
        critic_feedback: str = "",
    ) -> Tuple[Dict[str, Any], StageArtifact]:
        vars_ = {
            "scenario": scenario,
            "text": sample.text,
            "perception_json": perception_json,
            "context_graph": context_graph,
            "critic_feedback": critic_feedback,
        }
        out, artifact = run_prompt_json(
            backend=self.backend,
            prompt_store=self.prompt_store,
            stage_id="S3",
            prompt_id="P3_expected_norm",
            prompt_vars=vars_,
            max_retries=self.max_retries,
            media_items=None,
        )
        out = out or {}
        out.setdefault("expected_behavior", "")
        out.setdefault("norm_assumptions", [])
        artifact.output = out
        return out, artifact

