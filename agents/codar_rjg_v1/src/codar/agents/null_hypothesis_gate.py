from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..backends.base import BaseBackend
from ..constants import HYPOTHESES
from ..prompting import PromptStore
from ..types import StageArtifact
from ..utils import clamp
from .common import run_prompt_json


class NullHypothesisGateAgent:
    def __init__(self, backend: BaseBackend, prompt_store: PromptStore, max_retries: int):
        self.backend = backend
        self.prompt_store = prompt_store
        self.max_retries = max_retries

    @staticmethod
    def _normalize(hypotheses: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if isinstance(hypotheses, list):
            for row in hypotheses:
                if not isinstance(row, dict):
                    continue
                hid = str(row.get("id", "")).strip()
                if hid not in HYPOTHESES:
                    continue
                ef = clamp(float(row.get("evidence_fit", 0.0) or 0.0))
                cf = clamp(float(row.get("context_fit", 0.0) or 0.0))
                pa = clamp(float(row.get("parsimony", 0.0) or 0.0))
                ts = clamp(float(row.get("total_score", (ef + cf + pa) / 3.0) or 0.0))
                out.append(
                    {
                        "id": hid,
                        "evidence_fit": ef,
                        "context_fit": cf,
                        "parsimony": pa,
                        "total_score": ts,
                        "note": str(row.get("note", "")),
                    }
                )
        ids = {x["id"] for x in out}
        for hid in HYPOTHESES:
            if hid not in ids:
                out.append(
                    {
                        "id": hid,
                        "evidence_fit": 0.0,
                        "context_fit": 0.0,
                        "parsimony": 0.0,
                        "total_score": 0.0,
                        "note": "auto-filled",
                    }
                )
        out.sort(key=lambda x: x["total_score"], reverse=True)
        return out

    def run(self, scenario: str, conflict_report: Dict[str, Any], context_graph: Dict[str, Any]) -> Tuple[Dict[str, Any], StageArtifact]:
        vars_ = {
            "scenario": scenario,
            "conflict_report": conflict_report,
            "context_graph": context_graph,
        }
        out, artifact = run_prompt_json(
            backend=self.backend,
            prompt_store=self.prompt_store,
            stage_id="S3.5",
            prompt_id="P3c_null_hypothesis_gate",
            prompt_vars=vars_,
            max_retries=self.max_retries,
            media_items=None,
        )
        hypotheses = self._normalize(out.get("hypotheses", []))
        selected = str(out.get("selected_hypothesis", hypotheses[0]["id"])).strip()
        if selected not in HYPOTHESES:
            selected = hypotheses[0]["id"]
        need_abduction = bool(out.get("need_abduction", selected in {"H3", "H4", "H5"}))
        final = {
            "hypotheses": hypotheses,
            "selected_hypothesis": selected,
            "need_abduction": need_abduction,
            "reason": str(out.get("reason", "")),
        }
        artifact.output = final
        return final, artifact

