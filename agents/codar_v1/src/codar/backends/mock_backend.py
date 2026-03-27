from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..constants import HYPOTHESES
from ..types import LLMResponse
from .base import BaseBackend


class MockBackend(BaseBackend):
    name = "mock"

    def complete_json(
        self,
        prompt_text: str,
        prompt_id: str,
        media_items: Optional[List[Dict[str, Any]]] = None,
        temperature_override: Optional[float] = None,
    ) -> LLMResponse:
        _ = (media_items, temperature_override)
        if prompt_id == "P0_scenario_gate":
            out = {"locked_scenario": "affection", "validity_flag": True, "reason_short": "mock validated"}
        elif prompt_id == "P1_explicit_perception":
            out = {
                "text_components": {
                    "subject": "speaker",
                    "object": "target",
                    "predicate": "states message",
                    "attribute": "explicit wording",
                    "adverbial": "directly",
                },
                "image_action": {
                    "subject": "person",
                    "background": "indoor",
                    "behavior": "standing",
                    "action": "looking forward",
                },
                "audio_caption": {
                    "subject": "speaker",
                    "object": "utterance",
                    "predicate": "spoken",
                    "attribute": "neutral",
                    "adverbial": "steady",
                },
            }
        elif prompt_id == "P2_social_context":
            out = {
                "entities": [{"name": "speaker", "role": "subject"}, {"name": "target", "role": "target"}],
                "relations": [
                    {
                        "from": "speaker",
                        "to": "target",
                        "power": "equal",
                        "intimacy": "medium",
                        "history": "unknown",
                    }
                ],
                "culture_clues": [],
                "domain_notes": "mock",
            }
        elif prompt_id == "P3_expected_norm":
            out = {"expected_behavior": "calm direct expression", "norm_assumptions": ["cooperative tone"]}
        elif prompt_id == "P3b_conflict_judge":
            out = {
                "mechanism_scores": {
                    "multimodal incongruity": 0.7,
                    "figurative semantics": 0.2,
                    "affective deception": 0.1,
                    "socio_cultural dependency": 0.15,
                },
                "conflicts": [
                    {
                        "conflict_type": "multimodal incongruity",
                        "trigger_evidence": ["text says positive, context implies negative"],
                        "deviation_object": "stance polarity",
                        "deviation_direction": "positive_to_negative",
                        "confidence": 0.7,
                    }
                ],
                "summary": "mock conflict",
            }
        elif prompt_id == "P3c_null_hypothesis_gate":
            out = {
                "hypotheses": [
                    {"id": hid, "evidence_fit": 0.2, "context_fit": 0.2, "parsimony": 0.2, "total_score": 0.2, "note": ""}
                    for hid in HYPOTHESES
                ],
                "selected_hypothesis": "H3",
                "need_abduction": True,
                "reason": "mock strategic interpretation",
            }
            out["hypotheses"][3]["total_score"] = 0.8
        elif prompt_id == "P4_abductive_tot":
            out = {
                "candidates": [
                    {
                        "id": f"A{i}",
                        "cost_analysis": "social cost exists",
                        "motive_inference": "protect image",
                        "strategy_reconstruction": "indirect framing",
                        "evidence_fit": 0.6 + i * 0.05,
                        "context_fit": 0.6 + i * 0.04,
                        "parsimony": 0.5 + i * 0.03,
                        "total_score": 0.6 + i * 0.05,
                    }
                    for i in range(1, 5)
                ],
                "selected_id": "A4",
                "best_hypothesis": {
                    "id": "A4",
                    "cost_analysis": "social cost exists",
                    "motive_inference": "protect image",
                    "strategy_reconstruction": "indirect framing",
                    "evidence_fit": 0.8,
                    "context_fit": 0.76,
                    "parsimony": 0.62,
                    "total_score": 0.8,
                },
            }
        elif prompt_id == "P5_critic":
            out = {"pass": True, "issues": [], "revision_instructions": "", "backtrack_to": "NONE"}
        else:
            out = {
                "subject": "speaker",
                "target": "target",
                "mechanism": "multimodal incongruity",
                "label": "bad",
                "confidence": 0.66,
                "decision_rationale_short": "mock final decision",
            }
        return LLMResponse(parsed_json=out, raw_text=str(out), usage={})

    def metadata(self) -> Dict[str, Any]:
        return {"provider": "mock", "model": "mock-model", "base_url": "mock://"}
