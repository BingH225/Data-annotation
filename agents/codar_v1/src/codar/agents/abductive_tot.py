from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..backends.base import BaseBackend
from ..prompting import PromptStore
from ..types import StageArtifact
from ..utils import clamp
from .common import run_prompt_json


class AbductiveToTAgent:
    def __init__(self, backend: BaseBackend, prompt_store: PromptStore, max_retries: int):
        self.backend = backend
        self.prompt_store = prompt_store
        self.max_retries = max_retries

    @staticmethod
    def _normalize_candidates(rows: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                cid = str(row.get("id", "")).strip() or f"A{len(out) + 1}"
                ef = clamp(float(row.get("evidence_fit", 0.0) or 0.0))
                cf = clamp(float(row.get("context_fit", 0.0) or 0.0))
                pa = clamp(float(row.get("parsimony", 0.0) or 0.0))
                ts = clamp(float(row.get("total_score", (ef + cf + pa) / 3.0) or 0.0))
                out.append(
                    {
                        "id": cid,
                        "cost_analysis": str(row.get("cost_analysis", "")),
                        "motive_inference": str(row.get("motive_inference", "")),
                        "strategy_reconstruction": str(row.get("strategy_reconstruction", "")),
                        "evidence_fit": ef,
                        "context_fit": cf,
                        "parsimony": pa,
                        "total_score": ts,
                    }
                )
        while len(out) < 4:
            idx = len(out) + 1
            out.append(
                {
                    "id": f"A{idx}",
                    "cost_analysis": "auto-filled",
                    "motive_inference": "auto-filled",
                    "strategy_reconstruction": "auto-filled",
                    "evidence_fit": 0.0,
                    "context_fit": 0.0,
                    "parsimony": 0.0,
                    "total_score": 0.0,
                }
            )
        out = out[:4]
        out.sort(key=lambda x: x["total_score"], reverse=True)
        return out

    def run(
        self,
        scenario: str,
        selected_gate_hypothesis: str,
        conflict_report: Dict[str, Any],
        context_graph: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], StageArtifact]:
        vars_ = {
            "scenario": scenario,
            "selected_gate_hypothesis": selected_gate_hypothesis,
            "conflict_report": conflict_report,
            "context_graph": context_graph,
        }
        out, artifact = run_prompt_json(
            backend=self.backend,
            prompt_store=self.prompt_store,
            stage_id="S4",
            prompt_id="P4_abductive_tot",
            prompt_vars=vars_,
            max_retries=self.max_retries,
            media_items=None,
        )
        candidates = self._normalize_candidates(out.get("candidates", []))
        selected_id = str(out.get("selected_id", candidates[0]["id"])).strip()
        chosen = next((c for c in candidates if c["id"] == selected_id), candidates[0])
        final = {
            "candidates": candidates,
            "selected_id": selected_id,
            "best_hypothesis": chosen,
        }
        artifact.output = final
        return final, artifact

