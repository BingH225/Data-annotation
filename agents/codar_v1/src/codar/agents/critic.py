from __future__ import annotations

from typing import Any, Dict, Tuple

from ..backends.base import BaseBackend
from ..prompting import PromptStore
from ..types import StageArtifact
from .common import run_prompt_json


class CriticAgent:
    def __init__(self, backend: BaseBackend, prompt_store: PromptStore, max_retries: int):
        self.backend = backend
        self.prompt_store = prompt_store
        self.max_retries = max_retries

    def run(
        self,
        scenario: str,
        perception_json: Dict[str, Any],
        context_graph: Dict[str, Any],
        conflict_report: Dict[str, Any],
        gate_output: Dict[str, Any],
        abductive_output: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], StageArtifact]:
        vars_ = {
            "scenario": scenario,
            "perception_json": perception_json,
            "context_graph": context_graph,
            "conflict_report": conflict_report,
            "gate_output": gate_output,
            "abductive_output": abductive_output,
        }
        out, artifact = run_prompt_json(
            backend=self.backend,
            prompt_store=self.prompt_store,
            stage_id="S5",
            prompt_id="P5_critic",
            prompt_vars=vars_,
            max_retries=self.max_retries,
            media_items=None,
        )
        final = {
            "pass": bool(out.get("pass", True)),
            "issues": [str(x) for x in (out.get("issues", []) or [])],
            "revision_instructions": str(out.get("revision_instructions", "")),
            "backtrack_to": str(out.get("backtrack_to", "NONE")).upper(),
        }
        artifact.output = final
        return final, artifact

