from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..backends.base import BaseBackend
from ..constants import VALID_MECHANISMS
from ..prompting import PromptStore
from ..semantic_matcher import SemanticClosedSetMatcher
from ..types import SampleInput, StageArtifact
from ..utils import clamp
from .common import run_prompt_json


class ConflictEngine:
    def __init__(
        self,
        backend: BaseBackend,
        prompt_store: PromptStore,
        max_retries: int,
        alpha_rule: float,
        alpha_llm: float,
        scenario_policy: Dict[str, Any],
        semantic_matcher: SemanticClosedSetMatcher | None = None,
        semantic_threshold: float = 0.45,
    ):
        self.backend = backend
        self.prompt_store = prompt_store
        self.max_retries = max_retries
        self.alpha_rule = alpha_rule
        self.alpha_llm = alpha_llm
        self.scenario_policy = scenario_policy
        self.semantic_matcher = semantic_matcher
        self.semantic_threshold = float(semantic_threshold)

    def _rule_score(self, scenario: str, sample_text: str) -> Dict[str, float]:
        rules = (
            self.scenario_policy.get("scenarios", {})
            .get(scenario, {})
            .get("rule_keywords", {})
        )
        text = sample_text.lower()
        out: Dict[str, float] = {}
        for mechanism in VALID_MECHANISMS[scenario]:
            kws = [str(x).lower() for x in rules.get(mechanism, [])]
            if not kws:
                out[mechanism] = 0.0
                continue
            scored_hits: List[float] = []
            for kw in kws:
                lexical = 1.0 if kw in text else 0.0
                semantic = 0.0
                if self.semantic_matcher:
                    sim = self.semantic_matcher.similarity(sample_text, kw)
                    if sim is not None:
                        semantic = clamp((sim - self.semantic_threshold) / max(1e-6, 1.0 - self.semantic_threshold))
                scored_hits.append(max(lexical, semantic))
            top_hits = sorted(scored_hits, reverse=True)[: min(3, len(scored_hits))]
            max_hit = top_hits[0] if top_hits else 0.0
            dense_hit = sum(top_hits) / max(1.0, len(top_hits))
            lexical_presence = 1.0 if any(h >= 0.99 for h in scored_hits) else 0.0
            out[mechanism] = clamp(0.55 * max_hit + 0.35 * dense_hit + 0.10 * lexical_presence)
        cue_scores = self._cue_score(scenario=scenario, sample_text=sample_text)
        for mechanism in VALID_MECHANISMS[scenario]:
            out[mechanism] = clamp(max(out.get(mechanism, 0.0), cue_scores.get(mechanism, 0.0)))
        return out

    @staticmethod
    def _count_hits(text: str, patterns: List[str]) -> int:
        hits = 0
        for p in patterns:
            if p.startswith("re:"):
                if re.search(p[3:], text):
                    hits += 1
            elif p in text:
                hits += 1
        return hits

    def _cue_score(self, scenario: str, sample_text: str) -> Dict[str, float]:
        text = str(sample_text or "").lower()
        out = {m: 0.0 for m in VALID_MECHANISMS[scenario]}
        if scenario == "attitude":
            cues = {
                "dominant affiliation": [
                    "you got this",
                    "good job",
                    "re:for your own good",
                    "i'll handle",
                    "let me handle",
                ],
                "dominant detachment": [
                    "whatever",
                    "who cares",
                    "no thanks",
                    "re:third strike",
                    "re:\\bnot a\\b",
                    "fucker",
                    "ignorant",
                    "asked for it",
                ],
                "protective distancing": [
                    "i don't know",
                    "idk",
                    "maybe",
                    "not sure",
                    "we'll see",
                    "i guess",
                    "rather not",
                ],
                "submissive alignment": [
                    "sorry",
                    "my bad",
                    "please",
                    "you decide",
                    "whatever you want",
                    "need your help",
                ],
            }
        elif scenario == "intent":
            cues = {
                "prosocial deception": [
                    "just kidding",
                    "no offense",
                    "don't worry",
                    "all good",
                    "white lie",
                ],
                "malicious manipulation": [
                    "you owe me",
                    "if you cared",
                    "good person would",
                    "you must",
                    "moral",
                    "guilt",
                ],
                "expressive aggression": [
                    "fuck",
                    "wtf",
                    "idiot",
                    "moron",
                    "bitch",
                    "faggot",
                    "re:trying to kill",
                    "re:go to hell",
                ],
                "benevolent provocation": [
                    "what did you",
                    "prove it",
                    "dare you",
                    "come at me",
                    "hurry up",
                    "re:is it a boy or a girl",
                ],
            }
        else:
            return out
        for mechanism, patterns in cues.items():
            hits = self._count_hits(text, patterns)
            out[mechanism] = clamp(0.18 * hits)
        return out

    @staticmethod
    def _normalize_conflicts(conflicts: Any) -> List[Dict[str, Any]]:
        if not isinstance(conflicts, list):
            return []
        out: List[Dict[str, Any]] = []
        for c in conflicts:
            if not isinstance(c, dict):
                continue
            out.append(
                {
                    "conflict_type": str(c.get("conflict_type", "")),
                    "trigger_evidence": [str(x) for x in (c.get("trigger_evidence", []) or [])],
                    "deviation_object": str(c.get("deviation_object", "")),
                    "deviation_direction": str(c.get("deviation_direction", "")),
                    "confidence": clamp(float(c.get("confidence", 0.0) or 0.0)),
                }
            )
        return out

    def run(
        self,
        sample: SampleInput,
        scenario: str,
        perception_json: Dict[str, Any],
        expected_norm: Dict[str, Any],
        critic_feedback: str = "",
    ) -> Tuple[Dict[str, Any], StageArtifact]:
        mechanisms = VALID_MECHANISMS[scenario]
        vars_ = {
            "scenario": scenario,
            "mechanisms": mechanisms,
            "observed_x": perception_json,
            "expected_e": expected_norm,
            "critic_feedback": critic_feedback,
        }
        llm_out, artifact = run_prompt_json(
            backend=self.backend,
            prompt_store=self.prompt_store,
            stage_id="S3.1",
            prompt_id="P3b_conflict_judge",
            prompt_vars=vars_,
            max_retries=self.max_retries,
            media_items=None,
        )
        rule_scores = self._rule_score(scenario=scenario, sample_text=sample.text)
        llm_scores_raw = llm_out.get("mechanism_scores", {}) if isinstance(llm_out, dict) else {}
        llm_scores = {m: clamp(float(llm_scores_raw.get(m, 0.0) or 0.0)) for m in mechanisms}
        combined_scores: Dict[str, float] = {}
        for m in mechanisms:
            combined_scores[m] = clamp(self.alpha_rule * rule_scores.get(m, 0.0) + self.alpha_llm * llm_scores.get(m, 0.0))
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        conflicts = self._normalize_conflicts(llm_out.get("conflicts", []))
        top_mech, top_score = ranked[0]
        if not conflicts:
            conflicts = [
                {
                    "conflict_type": top_mech,
                    "trigger_evidence": ["score-based fallback"],
                    "deviation_object": "overall expression",
                    "deviation_direction": "non-default",
                    "confidence": top_score,
                }
            ]
        out = {
            "rule_scores": rule_scores,
            "llm_scores": llm_scores,
            "combined_scores": combined_scores,
            "top_mechanism": top_mech,
            "top_confidence": top_score,
            "conflicts": conflicts,
            "summary": str(llm_out.get("summary", "")),
        }
        artifact.output = out
        return out, artifact
