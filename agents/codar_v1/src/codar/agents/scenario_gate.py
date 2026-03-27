from __future__ import annotations

from typing import Dict, Tuple

from ..backends.base import BaseBackend
from ..constants import SCENARIOS
from ..prompting import PromptStore
from ..types import SampleInput, StageArtifact
from .common import run_prompt_json


class ScenarioGateAgent:
    def __init__(self, backend: BaseBackend, prompt_store: PromptStore, max_retries: int):
        self.backend = backend
        self.prompt_store = prompt_store
        self.max_retries = max_retries

    def run(self, sample: SampleInput) -> Tuple[str, StageArtifact]:
        scenario = sample.scenario.strip().lower()
        if scenario in SCENARIOS:
            artifact = StageArtifact(
                stage_id="S0",
                status="ok",
                output={
                    "locked_scenario": scenario,
                    "validity_flag": True,
                    "reason_short": "trusted input scenario",
                },
                prompt_meta=None,
                retries=0,
            )
            return scenario, artifact
        vars_: Dict[str, str] = {
            "sample_id": sample.sample_id,
            "input_scenario": sample.scenario,
            "text": sample.text,
        }
        out, artifact = run_prompt_json(
            backend=self.backend,
            prompt_store=self.prompt_store,
            stage_id="S0",
            prompt_id="P0_scenario_gate",
            prompt_vars=vars_,
            max_retries=self.max_retries,
            media_items=None,
        )
        locked = str(out.get("locked_scenario", "")).strip().lower()
        if locked not in SCENARIOS:
            locked = "affection"
            artifact.status = "fallback"
            artifact.output = {
                "locked_scenario": locked,
                "validity_flag": False,
                "reason_short": "invalid scenario output, fallback to affection",
            }
        return locked, artifact

