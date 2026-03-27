from __future__ import annotations

from typing import Any, Dict, List

from ..agents.abductive_tot import AbductiveToTAgent
from ..agents.conflict_engine import ConflictEngine
from ..agents.critic import CriticAgent
from ..agents.explicit_perception import ExplicitPerceptionAgent
from ..agents.expectation import ExpectationAgent
from ..agents.final_decision import FinalDecisionAgent
from ..agents.null_hypothesis_gate import NullHypothesisGateAgent
from ..agents.scenario_gate import ScenarioGateAgent
from ..agents.social_context import SocialContextAgent
from ..types import FinalPrediction, SampleInput, SampleResult, StageArtifact


class CoDARPipeline:
    def __init__(
        self,
        scenario_gate: ScenarioGateAgent,
        explicit_perception: ExplicitPerceptionAgent,
        social_context: SocialContextAgent,
        expectation: ExpectationAgent,
        conflict_engine: ConflictEngine,
        null_gate: NullHypothesisGateAgent,
        abductive_tot: AbductiveToTAgent,
        critic: CriticAgent,
        final_decision: FinalDecisionAgent,
        max_backtrack_rounds: int = 2,
    ):
        self.scenario_gate = scenario_gate
        self.explicit_perception = explicit_perception
        self.social_context = social_context
        self.expectation = expectation
        self.conflict_engine = conflict_engine
        self.null_gate = null_gate
        self.abductive_tot = abductive_tot
        self.critic = critic
        self.final_decision = final_decision
        self.max_backtrack_rounds = max_backtrack_rounds

    @staticmethod
    def _artifact(stage_artifacts: List[StageArtifact], x: StageArtifact) -> None:
        stage_artifacts.append(x)

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _consistency_score(
        self,
        conflict_report: Dict[str, Any],
        gate_output: Dict[str, Any],
        abductive_output: Dict[str, Any],
        critic_output: Dict[str, Any],
    ) -> float:
        critic_pass = bool(critic_output.get("pass", False))
        issues = critic_output.get("issues", []) or []
        issue_penalty = 0.25 * len(issues)
        conflict_conf = self._to_float(conflict_report.get("top_confidence", 0.0), 0.0)

        gate_hyps = gate_output.get("hypotheses", []) or []
        gate_best = 0.0
        for h in gate_hyps:
            if not isinstance(h, dict):
                continue
            gate_best = max(gate_best, self._to_float(h.get("total_score", 0.0), 0.0))

        abductive_best = 0.0
        if isinstance(abductive_output, dict):
            best_h = abductive_output.get("best_hypothesis", {}) or {}
            abductive_best = self._to_float(best_h.get("total_score", 0.0), 0.0)

        pass_bonus = 2.0 if critic_pass else 0.0
        return pass_bonus + conflict_conf + 0.5 * gate_best + 0.25 * abductive_best - issue_penalty

    def run_sample(self, sample: SampleInput, backend_meta: Dict[str, Any]) -> SampleResult:
        artifacts: List[StageArtifact] = []
        trace: Dict[str, Any] = {"backtrack_rounds": 0}
        try:
            scenario, s0 = self.scenario_gate.run(sample)
            self._artifact(artifacts, s0)

            perception, media_manifest, s1 = self.explicit_perception.run(sample=sample, scenario=scenario)
            self._artifact(artifacts, s1)

            context_graph, s2 = self.social_context.run(sample=sample, scenario=scenario, perception_json=perception)
            self._artifact(artifacts, s2)

            critic_feedback = ""
            expectation_out: Dict[str, Any] = {}
            conflict_report: Dict[str, Any] = {}
            gate_output: Dict[str, Any] = {}
            abductive_output: Dict[str, Any] = {"skipped": True, "candidates": []}
            critic_output: Dict[str, Any] = {"pass": True, "issues": []}
            branch_records: List[Dict[str, Any]] = []

            for round_idx in range(self.max_backtrack_rounds + 1):
                trace["backtrack_rounds"] = round_idx
                expectation_out, s3 = self.expectation.run(
                    sample=sample,
                    scenario=scenario,
                    perception_json=perception,
                    context_graph=context_graph,
                    critic_feedback=critic_feedback,
                )
                self._artifact(artifacts, s3)

                conflict_report, s31 = self.conflict_engine.run(
                    sample=sample,
                    scenario=scenario,
                    perception_json=perception,
                    expected_norm=expectation_out,
                    critic_feedback=critic_feedback,
                )
                self._artifact(artifacts, s31)

                gate_output, s35 = self.null_gate.run(
                    scenario=scenario,
                    conflict_report=conflict_report,
                    context_graph=context_graph,
                )
                self._artifact(artifacts, s35)

                if gate_output.get("need_abduction", False):
                    abductive_output, s4 = self.abductive_tot.run(
                        scenario=scenario,
                        selected_gate_hypothesis=gate_output.get("selected_hypothesis", "H0"),
                        conflict_report=conflict_report,
                        context_graph=context_graph,
                    )
                else:
                    s4 = StageArtifact(
                        stage_id="S4",
                        status="skipped",
                        output={
                            "candidates": [],
                            "selected_id": "",
                            "best_hypothesis": {},
                            "reason": "null-hypothesis gate returned need_abduction=false",
                        },
                        prompt_meta=None,
                        retries=0,
                    )
                    abductive_output = s4.output
                self._artifact(artifacts, s4)

                critic_output, s5 = self.critic.run(
                    scenario=scenario,
                    perception_json=perception,
                    context_graph=context_graph,
                    conflict_report=conflict_report,
                    gate_output=gate_output,
                    abductive_output=abductive_output,
                )
                self._artifact(artifacts, s5)
                branch_records.append(
                    {
                        "round_idx": round_idx,
                        "expectation_out": expectation_out,
                        "conflict_report": conflict_report,
                        "gate_output": gate_output,
                        "abductive_output": abductive_output,
                        "critic_output": critic_output,
                        "consistency_score": self._consistency_score(
                            conflict_report=conflict_report,
                            gate_output=gate_output,
                            abductive_output=abductive_output,
                            critic_output=critic_output,
                        ),
                    }
                )

                if critic_output.get("pass", True):
                    break
                critic_feedback = critic_output.get("revision_instructions", "")

            if branch_records:
                selected_branch = max(
                    branch_records,
                    key=lambda x: (
                        bool((x.get("critic_output", {}) or {}).get("pass", False)),
                        self._to_float(x.get("consistency_score", 0.0), 0.0),
                        -len((x.get("critic_output", {}) or {}).get("issues", []) or []),
                    ),
                )
                expectation_out = selected_branch.get("expectation_out", expectation_out)
                conflict_report = selected_branch.get("conflict_report", conflict_report)
                gate_output = selected_branch.get("gate_output", gate_output)
                abductive_output = selected_branch.get("abductive_output", abductive_output)
                critic_output = selected_branch.get("critic_output", critic_output)
                trace["branch_scores"] = [
                    {
                        "round_idx": int(x.get("round_idx", 0)),
                        "consistency_score": self._to_float(x.get("consistency_score", 0.0), 0.0),
                        "critic_pass": bool((x.get("critic_output", {}) or {}).get("pass", False)),
                        "issue_count": len((x.get("critic_output", {}) or {}).get("issues", []) or []),
                    }
                    for x in branch_records
                ]
                trace["selected_round_idx"] = int(selected_branch.get("round_idx", 0))

            pred, s6 = self.final_decision.run(
                sample=sample,
                scenario=scenario,
                conflict_report=conflict_report,
                gate_output=gate_output,
                abductive_output=abductive_output,
                critic_output=critic_output,
            )
            self._artifact(artifacts, s6)
            trace["media_manifest"] = media_manifest
            return SampleResult(
                sample_id=sample.sample_id,
                scenario=scenario,
                final_prediction=pred,
                stage_artifacts=artifacts,
                backend_meta=backend_meta,
                trace=trace,
                error=None,
            )
        except Exception as exc:  # pragma: no cover
            return SampleResult(
                sample_id=sample.sample_id,
                scenario=sample.scenario,
                final_prediction=FinalPrediction(
                    subject="",
                    target="",
                    mechanism="",
                    label="",
                    confidence=0.0,
                    decision_rationale_short="",
                ),
                stage_artifacts=artifacts,
                backend_meta=backend_meta,
                trace=trace,
                error=str(exc),
            )
