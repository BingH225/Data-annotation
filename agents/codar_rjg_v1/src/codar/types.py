from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SampleInput:
    sample_id: str
    scenario: str
    text: str
    media: Dict[str, Any]
    subject_options: List[str]
    target_options: List[str]
    diversity: Dict[str, Any]
    ground_truth: Dict[str, Any]


@dataclass
class PromptMeta:
    prompt_id: str
    prompt_vars: Dict[str, Any]
    prompt_hash: str


@dataclass
class StageArtifact:
    stage_id: str
    status: str
    output: Dict[str, Any]
    prompt_meta: Optional[PromptMeta] = None
    retries: int = 0
    notes: str = ""


@dataclass
class FinalPrediction:
    subject: str
    target: str
    mechanism: str
    label: str
    confidence: float
    decision_rationale_short: str


@dataclass
class SampleResult:
    sample_id: str
    scenario: str
    final_prediction: FinalPrediction
    stage_artifacts: List[StageArtifact]
    backend_meta: Dict[str, Any]
    trace: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class RunContext:
    run_id: str
    output_dir: Path
    backend_name: str
    config_path: Path
    media_mode: str
    scenario_filter: Optional[str] = None


@dataclass
class LLMResponse:
    parsed_json: Dict[str, Any]
    raw_text: str
    usage: Dict[str, Any] = field(default_factory=dict)

