from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from ..agents.common import build_media_items, run_prompt_json
from ..backends.base import BaseBackend
from ..constants import VALID_LABELS, VALID_MECHANISMS
from ..media import MediaResolver
from ..prompting import PromptStore
from ..types import SampleInput, StageArtifact
from ..utils import clamp, clip_text, map_to_closed_set
from .anchors import build_anchor_payload
from .fusion import (
    RJGConstraintConfig,
    RJGWeights,
    compatibility_prior_score,
    compute_total_score,
    default_rjg_weights,
    heuristic_agreement_score,
    predict_heuristic_label,
    predict_heuristic_mechanism,
    repair_inconsistent_label,
    score_penalty_components,
    rule_cue_score,
)
from .memory import MemoryIndex, retrieve_similar_entries

_AFFECTION_LABEL_CUES: Dict[str, List[str]] = {
    "happy": ["love", "great", "nice", "wonderful", "proud", "cute", "good"],
    "sad": ["sad", "hurt", "cry", "lonely", "down", "depressed"],
    "disgusted": ["gross", "disgust", "nasty", "ew", "repulsive", "no place", "hate"],
    "angry": ["wtf", "fuck", "fucking", "bitch", "mad", "rage", "damn", "srsly"],
    "fearful": ["afraid", "scared", "fear", "threat", "terror", "anxious", "danger"],
    "bad": ["tired", "busy", "stressed", "bored", "meh", "whatever", "fine", "okay"],
}

_AFFECTION_ALT_LABELS: List[str] = ["angry", "bad", "happy", "sad", "fearful"]


def should_rejudge_branch(
    scored: List[Dict[str, Any]],
    accept_total_threshold: float = 0.70,
    min_margin_threshold: float = 0.06,
    min_component_threshold: float = 0.34,
    min_component_gap: float = 0.08,
) -> Tuple[bool, str, Dict[str, float]]:
    if len(scored) < 2:
        return False, "single_candidate", {"margin": 0.0}

    best = scored[0]
    second = scored[1]
    best_total = float(best.get("total_score", 0.0))
    second_total = float(second.get("total_score", 0.0))
    margin = best_total - second_total
    best_components = best.get("components", {}) or {}
    second_components = second.get("components", {}) or {}
    best_label = float(best_components.get("judge_label", 0.0))
    best_mech = float(best_components.get("judge_mech", 0.0))
    best_heur = float(best_components.get("heuristic_agreement", 0.0))
    second_label = float(second_components.get("judge_label", 0.0))
    second_mech = float(second_components.get("judge_mech", 0.0))
    second_heur = float(second_components.get("heuristic_agreement", 0.0))

    metrics = {
        "best_total": best_total,
        "second_total": second_total,
        "margin": margin,
        "best_label": best_label,
        "best_mech": best_mech,
        "best_heuristic": best_heur,
        "second_label": second_label,
        "second_mech": second_mech,
        "second_heuristic": second_heur,
    }

    if best_total >= accept_total_threshold and margin >= min_margin_threshold:
        return False, "clear_winner", metrics
    if best_total >= accept_total_threshold and min(best_label, best_mech, best_heur) >= min_component_threshold:
        return False, "high_confidence", metrics
    if best_total < accept_total_threshold and margin < min_margin_threshold:
        return True, "low_total_close_gap", metrics
    if min(best_label, best_mech) < min_component_threshold and margin < max(min_margin_threshold, min_component_gap):
        return True, "weak_judge_signals", metrics
    if best_heur < min_component_threshold and second_heur > best_heur + min_component_gap and margin < min_margin_threshold * 2:
        return True, "heuristic_disagreement", metrics
    return False, "stable", metrics


class RJGPipeline:
    def __init__(
        self,
        backend: BaseBackend,
        prompt_store: PromptStore,
        media_resolver: MediaResolver,
        memory_index: MemoryIndex,
        scenario_policy: Dict[str, Any],
        max_retries: int,
        max_video_frames: int = 4,
        weights: RJGWeights | None = None,
        tie_margin: float = 0.06,
        top_k: int = 40,
        rerank_k: int = 12,
        loo: bool = True,
        constraint_config: RJGConstraintConfig | Dict[str, float] | None = None,
        accept_total_threshold: float = 0.70,
        min_margin_threshold: float = 0.06,
        min_component_threshold: float = 0.34,
        min_component_gap: float = 0.08,
    ):
        self.backend = backend
        self.prompt_store = prompt_store
        self.media_resolver = media_resolver
        self.memory_index = memory_index
        self.scenario_policy = scenario_policy
        self.max_retries = int(max_retries)
        self.max_video_frames = int(max_video_frames)
        self.weights = weights or default_rjg_weights()
        self.tie_margin = float(tie_margin)
        self.top_k = int(top_k)
        self.rerank_k = int(rerank_k)
        self.loo = bool(loo)
        self.constraint_config = constraint_config
        self.accept_total_threshold = float(accept_total_threshold)
        self.min_margin_threshold = float(min_margin_threshold)
        self.min_component_threshold = float(min_component_threshold)
        self.min_component_gap = float(min_component_gap)

    @staticmethod
    def _artifact(stage_id: str, output: Dict[str, Any], status: str = "ok", notes: str = "") -> StageArtifact:
        return StageArtifact(stage_id=stage_id, status=status, output=output, prompt_meta=None, retries=0, notes=notes)

    @staticmethod
    def _tie_break_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
        components = candidate.get("components", {}) or {}
        return {
            "subject": candidate.get("subject", ""),
            "target": candidate.get("target", ""),
            "mechanism": candidate.get("mechanism", ""),
            "label": candidate.get("label", ""),
            "total_score": float(candidate.get("total_score", 0.0)),
            "heuristic_agreement": float(components.get("heuristic_agreement", 0.0)),
            "judge_label": float(components.get("judge_label", 0.0)),
            "judge_mech": float(components.get("judge_mech", 0.0)),
            "rule_cue": float(components.get("rule_cue", 0.0)),
        }

    @staticmethod
    def _normalize_tie_break_winner(raw_winner: Any) -> str | None:
        winner = str(raw_winner or "").strip().upper()
        if winner in {"A", "1", "FIRST", "LEFT"}:
            return "A"
        if winner in {"B", "2", "SECOND", "RIGHT"}:
            return "B"
        return None

    def _deterministic_tie_break(self, scored: List[Dict[str, Any]], branch_reason: str) -> Tuple[int, Dict[str, Any]]:
        tie_break = {
            "mode": "deterministic",
            "winner": "A",
            "reason": branch_reason,
            "candidate_a": self._tie_break_summary(scored[0]),
            "candidate_b": self._tie_break_summary(scored[1]),
        }
        chosen_idx = 0
        if (
            float(scored[1]["components"]["heuristic_agreement"]) > float(scored[0]["components"]["heuristic_agreement"])
            or (
                float(scored[1]["components"]["heuristic_agreement"]) == float(scored[0]["components"]["heuristic_agreement"])
                and float(scored[1]["components"]["judge_label"]) > float(scored[0]["components"]["judge_label"])
            )
            or (
                float(scored[1]["components"]["heuristic_agreement"]) == float(scored[0]["components"]["heuristic_agreement"])
                and float(scored[1]["components"]["judge_label"]) == float(scored[0]["components"]["judge_label"])
                and float(scored[1]["components"]["judge_mech"]) > float(scored[0]["components"]["judge_mech"])
            )
        ):
            chosen_idx = 1
            tie_break["winner"] = "B"
            tie_break["reason"] = f"{branch_reason}: deterministic consistency tie-break"
        return chosen_idx, tie_break

    def _should_attempt_llm_tie_break(self, branch_reason: str, branch_metrics: Dict[str, float]) -> bool:
        if branch_reason not in {"low_total_close_gap", "weak_judge_signals", "heuristic_disagreement"}:
            return False
        margin = float(branch_metrics.get("margin", 0.0))
        margin_limit = max(self.tie_margin, self.min_margin_threshold) * 1.5
        return margin <= margin_limit

    def _llm_tie_break(
        self,
        scenario: str,
        text: str,
        audio_caption: str,
        scored: List[Dict[str, Any]],
        media_items: List[Dict[str, Any]],
    ) -> Tuple[int | None, Dict[str, Any], StageArtifact]:
        parsed, artifact = run_prompt_json(
            backend=self.backend,
            prompt_store=self.prompt_store,
            stage_id="M5_tie_break",
            prompt_id="R5_tiebreak",
            prompt_vars={
                "scenario": scenario,
                "text": text,
                "audio_caption": audio_caption,
                "candidate_a": self._tie_break_summary(scored[0]),
                "candidate_b": self._tie_break_summary(scored[1]),
            },
            max_retries=self.max_retries,
            media_items=media_items,
            temperature_override=0.0,
        )
        winner = self._normalize_tie_break_winner(parsed.get("winner"))
        tie_break = {
            "mode": "llm",
            "prompt_status": artifact.status,
            "prompt_notes": artifact.notes,
            "parsed_winner": parsed.get("winner"),
            "reason_short": clip_text(str(parsed.get("reason_short") or parsed.get("reason") or ""), 120),
            "candidate_a": self._tie_break_summary(scored[0]),
            "candidate_b": self._tie_break_summary(scored[1]),
        }
        if winner is None:
            tie_break["mode"] = "llm_invalid"
            return None, tie_break, artifact
        tie_break["winner"] = winner
        return (0 if winner == "A" else 1), tie_break, artifact

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _pick_entity(value: str, options: List[str]) -> str:
        text = str(value or "").strip()
        if not options:
            return text
        return map_to_closed_set(text, options, options[0])

    @staticmethod
    def _pick_taxonomy(value: str, allowed: List[str]) -> str:
        text = str(value or "").strip()
        if not allowed:
            return text
        return map_to_closed_set(text, allowed, allowed[0])

    @staticmethod
    def _keyword_hits(text: str, keywords: List[str]) -> int:
        lowered = str(text or "").lower()
        return sum(1 for kw in keywords if str(kw).lower() in lowered)

    def _affection_signal_profile(self, text: str) -> Dict[str, Any]:
        profile_hits = {lb: self._keyword_hits(text, kws) for lb, kws in _AFFECTION_LABEL_CUES.items()}
        alt_max = 0
        for lb in _AFFECTION_ALT_LABELS:
            alt_max = max(alt_max, int(profile_hits.get(lb, 0)))
        return {
            "hits": profile_hits,
            "disgust_hits": int(profile_hits.get("disgusted", 0)),
            "alt_max": int(alt_max),
        }

    def _run_baseline_probe(
        self,
        sample: SampleInput,
        scenario: str,
        media_items: List[Dict[str, Any]],
        audio_caption: str,
    ) -> Tuple[Dict[str, Any], StageArtifact]:
        parsed, artifact = run_prompt_json(
            backend=self.backend,
            prompt_store=self.prompt_store,
            stage_id="M2_baseline_probe",
            prompt_id="PB0_baseline_direct",
            prompt_vars={
                "scenario": scenario,
                "text": sample.text,
                "audio_caption": audio_caption,
                "subject_options": sample.subject_options,
                "target_options": sample.target_options,
                "valid_mechanisms": VALID_MECHANISMS.get(scenario, []) or [],
                "valid_labels": VALID_LABELS.get(scenario, []) or [],
            },
            max_retries=self.max_retries,
            media_items=media_items,
            temperature_override=0.0,
        )
        rows = self._extract_candidates(parsed, sample)
        if rows:
            candidate = dict(rows[0])
        else:
            candidate = {
                "subject": sample.subject_options[0] if sample.subject_options else "",
                "target": sample.target_options[0] if sample.target_options else "",
                "mechanism": (VALID_MECHANISMS.get(scenario) or [""])[0],
                "label": (VALID_LABELS.get(scenario) or [""])[0],
                "confidence": 0.0,
                "rationale": "baseline probe fallback",
            }
        candidate["source"] = "baseline_probe"
        candidate["probe_prompt_id"] = "PB0_baseline_direct"
        return candidate, artifact

    def _affection_candidate_adjustment(
        self,
        text: str,
        candidate_label: str,
        baseline_label: str,
    ) -> Dict[str, Any]:
        profile = self._affection_signal_profile(text)
        cand = str(candidate_label or "").strip().lower()
        base = str(baseline_label or "").strip().lower()
        label_bonus = 0.0
        penalty_add = 0.0
        reason = "none"

        if base and cand == base and base != "disgusted":
            label_bonus += 0.22
            reason = "baseline_non_disgust_anchor"

        if cand == "disgusted":
            if profile["disgust_hits"] >= 1 and profile["alt_max"] <= 1:
                label_bonus += 0.08
                reason = "disgust_supported"
            else:
                penalty_add += 0.18 + 0.04 * max(0, profile["alt_max"] - 1)
                reason = "disgust_overuse_penalty"
        elif base == cand and base != "disgusted" and profile["disgust_hits"] == 0:
            label_bonus += 0.04
            reason = "non_disgust_reinforce"

        return {
            "label_bonus": float(label_bonus),
            "penalty_add": float(penalty_add),
            "reason": reason,
            "disgust_hits": profile["disgust_hits"],
            "alt_max": profile["alt_max"],
        }

    def _arbitrate_affection_label(
        self,
        text: str,
        rjg_label: str,
        baseline_label: str,
    ) -> Tuple[str, Dict[str, Any]]:
        profile = self._affection_signal_profile(text)
        rjg = str(rjg_label or "").strip()
        baseline = str(baseline_label or "").strip()
        final_label = rjg or baseline
        reason = "fallback_rjg"

        if baseline:
            final_label = baseline
            reason = "baseline_default"
            if rjg.lower() == "disgusted" and baseline.lower() != "disgusted":
                if profile["disgust_hits"] >= 1 and profile["alt_max"] <= 1:
                    final_label = "disgusted"
                    reason = "rjg_disgust_override"
            elif baseline.lower() == "disgusted" and rjg and rjg.lower() != "disgusted":
                if profile["disgust_hits"] == 0 and profile["alt_max"] >= 1:
                    final_label = rjg
                    reason = "avoid_false_disgust"
        info = {
            "reason": reason,
            "rjg_label": rjg,
            "baseline_label": baseline,
            "selected_label": final_label,
            "disgust_hits": profile["disgust_hits"],
            "alt_max": profile["alt_max"],
            "signal_hits": profile["hits"],
        }
        return final_label, info

    def _extract_candidates(self, parsed: Dict[str, Any], sample: SampleInput) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        raw_list = parsed.get("candidates")
        if isinstance(raw_list, list):
            src = [x for x in raw_list if isinstance(x, dict)]
        elif isinstance(parsed, dict):
            src = [parsed]
        else:
            src = []
        valid_mechs = VALID_MECHANISMS.get(sample.scenario, [])
        valid_labels = VALID_LABELS.get(sample.scenario, [])
        for row in src:
            cand = {
                "subject": self._pick_entity(str(row.get("subject", "")), sample.subject_options),
                "target": self._pick_entity(str(row.get("target", "")), sample.target_options),
                "mechanism": self._pick_taxonomy(str(row.get("mechanism", "")), valid_mechs),
                "label": self._pick_taxonomy(str(row.get("label", "")), valid_labels),
                "confidence": clamp(self._to_float(row.get("confidence", 0.0), 0.0), 0.0, 1.0),
                "rationale": clip_text(str(row.get("decision_rationale_short") or row.get("rationale") or ""), 300),
            }
            rows.append(cand)
        return rows

    def _dedup_candidates(self, candidates: List[Dict[str, Any]], sample: SampleInput) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for c in candidates:
            key = (c["subject"], c["target"], c["mechanism"], c["label"])
            cur = merged.get(key)
            if cur is None or float(c.get("confidence", 0.0)) > float(cur.get("confidence", 0.0)):
                merged[key] = c
        out = list(merged.values())
        if out:
            return out
        return [
            {
                "subject": sample.subject_options[0] if sample.subject_options else "",
                "target": sample.target_options[0] if sample.target_options else "",
                "mechanism": (VALID_MECHANISMS.get(sample.scenario) or [""])[0],
                "label": (VALID_LABELS.get(sample.scenario) or [""])[0],
                "confidence": 0.0,
                "rationale": "",
            }
        ]

    @staticmethod
    def _retrieve_support(candidate: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> float:
        if not retrieved:
            return 0.0
        mean_score = sum(float(x.get("retrieval_score", 0.0)) for x in retrieved) / max(len(retrieved), 1)
        sub = str(candidate.get("subject", "")).strip().lower()
        tgt = str(candidate.get("target", "")).strip().lower()
        sub_hit = 0.0
        tgt_hit = 0.0
        for row in retrieved:
            sub_opts = [str(x).strip().lower() for x in (row.get("subject_options", []) or [])]
            tgt_opts = [str(x).strip().lower() for x in (row.get("target_options", []) or [])]
            if sub and sub in sub_opts:
                sub_hit += 1.0
            if tgt and tgt in tgt_opts:
                tgt_hit += 1.0
        sub_hit /= max(len(retrieved), 1)
        tgt_hit /= max(len(retrieved), 1)
        return clamp(0.4 * mean_score + 0.3 * sub_hit + 0.3 * tgt_hit, 0.0, 1.0)

    def _build_heuristic_candidates(
        self,
        sample: SampleInput,
        scenario: str,
        anchors: Dict[str, Any],
        text: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        best_mech, mech_scores = predict_heuristic_mechanism(scenario, text, self.scenario_policy)
        mech_ranked = sorted(mech_scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
        best_label, label_scores = predict_heuristic_label(scenario, best_mech, text)
        subject = anchors.get("subject_anchor") or (sample.subject_options[0] if sample.subject_options else "")
        target = anchors.get("target_anchor") or (sample.target_options[0] if sample.target_options else "")

        primary_detail = {
            "predicted_mechanism": best_mech,
            "predicted_label": best_label,
            "mechanism_scores": mech_scores,
            "label_scores": label_scores,
            "ranked_mechanisms": mech_ranked[:4],
        }
        primary_agreement, primary_signal = heuristic_agreement_score(
            scenario=scenario,
            mechanism=best_mech,
            label=best_label,
            text=text,
            scenario_policy=self.scenario_policy,
            anchors=anchors,
        )
        primary = {
            "subject": subject,
            "target": target,
            "mechanism": best_mech,
            "label": best_label,
            "confidence": clamp(0.55 + 0.35 * primary_agreement, 0.0, 1.0),
            "rationale": "heuristic cue fusion",
            "source": "heuristic_primary",
            "heuristic_agreement": primary_agreement,
            "heuristic_detail": {**primary_detail, **primary_signal},
        }

        candidates = [primary]
        if len(mech_ranked) > 1:
            alt_mech = mech_ranked[1][0]
            if alt_mech and alt_mech != best_mech:
                alt_label, alt_label_scores = predict_heuristic_label(scenario, alt_mech, text)
                alt_agreement, alt_signal = heuristic_agreement_score(
                    scenario=scenario,
                    mechanism=alt_mech,
                    label=alt_label,
                    text=text,
                    scenario_policy=self.scenario_policy,
                    anchors=anchors,
                )
                candidates.append(
                    {
                        "subject": subject,
                        "target": target,
                        "mechanism": alt_mech,
                        "label": alt_label,
                        "confidence": clamp(0.48 + 0.30 * alt_agreement, 0.0, 1.0),
                        "rationale": "heuristic alternative cue path",
                        "source": "heuristic_alternative",
                        "heuristic_agreement": alt_agreement,
                        "heuristic_detail": {
                            "predicted_mechanism": alt_mech,
                            "predicted_label": alt_label,
                            "mechanism_scores": mech_scores,
                            "label_scores": alt_label_scores,
                            "ranked_mechanisms": mech_ranked[:4],
                            **alt_signal,
                        },
                    }
                )
        return candidates, {"primary": primary, "mechanism_ranking": mech_ranked[:4], "subject": subject, "target": target}

    def _build_llm_candidates(
        self,
        sample: SampleInput,
        scenario: str,
        anchors: Dict[str, Any],
        retrieved_context: List[Dict[str, Any]],
        media_items: List[Dict[str, Any]],
        audio_caption: str,
    ) -> Tuple[List[Dict[str, Any]], List[StageArtifact], Dict[str, Any]]:
        plans: List[Tuple[str, float]] = [("literal_pragmatic", 0.0)]
        if scenario == "affection":
            plans.append(("affection_label_diversity", 0.2))

        out: List[Dict[str, Any]] = []
        artifacts: List[StageArtifact] = []
        by_view: List[Dict[str, Any]] = []
        for idx, (view_name, temp) in enumerate(plans, start=1):
            parsed, artifact = run_prompt_json(
                backend=self.backend,
                prompt_store=self.prompt_store,
                stage_id=f"M2_candidate_generate_{idx}",
                prompt_id="R1_candidate_dualview",
                prompt_vars={
                    "scenario": scenario,
                    "view": view_name,
                    "temperature": temp,
                    "text": sample.text,
                    "audio_caption": audio_caption,
                    "subject_options": sample.subject_options,
                    "target_options": sample.target_options,
                    "valid_mechanisms": VALID_MECHANISMS.get(scenario, []) or [],
                    "valid_labels": VALID_LABELS.get(scenario, []) or [],
                    "anchors": anchors,
                    "retrieved_context": retrieved_context,
                },
                max_retries=self.max_retries,
                media_items=media_items,
                temperature_override=temp,
            )
            artifacts.append(artifact)
            rows = self._extract_candidates(parsed, sample)
            for row in rows:
                item = dict(row)
                item["source"] = "llm_candidate"
                item["generation_view"] = view_name
                item["generation_temperature"] = temp
                out.append(item)
            by_view.append(
                {
                    "view": view_name,
                    "temperature": temp,
                    "status": artifact.status,
                    "count": len(rows),
                    "notes": artifact.notes,
                }
            )
        summary = {
            "status": "ok" if any(v.get("status") == "ok" for v in by_view) else "failed",
            "count": len(out),
            "views": by_view,
        }
        return out, artifacts, summary

    def _judge_score(
        self,
        stage_id: str,
        prompt_vars: Dict[str, Any],
        media_items: List[Dict[str, Any]],
        prompt_id: str | None = None,
    ) -> Tuple[float, StageArtifact, Dict[str, Any]]:
        scenario = str(prompt_vars.get("scenario", "")).strip().lower()
        candidate = dict(prompt_vars.get("candidate", {}) or {})
        text = str(prompt_vars.get("text", ""))
        audio_caption = str(prompt_vars.get("audio_caption", ""))
        full_text = f"{text}\n{audio_caption}".strip()
        anchors = dict(prompt_vars.get("anchors", {}) or {})
        subject_options = list(prompt_vars.get("subject_options", []) or [])
        target_options = list(prompt_vars.get("target_options", []) or [])
        mechanism = str(candidate.get("mechanism", ""))
        label = str(candidate.get("label", ""))

        parsed: Dict[str, Any]
        if "mech" in stage_id:
            pred_mech, mech_scores = predict_heuristic_mechanism(scenario, full_text, self.scenario_policy)
            heuristic_score, heuristic_detail = heuristic_agreement_score(
                scenario=scenario,
                mechanism=mechanism,
                label=label,
                text=full_text,
                scenario_policy=self.scenario_policy,
                anchors=anchors,
            )
            cue_score = rule_cue_score(
                scenario=scenario,
                mechanism=mechanism,
                text=full_text,
                scenario_policy=self.scenario_policy,
            )
            compat = compatibility_prior_score(scenario, mechanism, label)
            match = 1.0 if str(pred_mech).strip().lower() == mechanism.strip().lower() else 0.0
            score = clamp(0.58 * match + 0.17 * cue_score + 0.15 * heuristic_score + 0.10 * compat, 0.0, 1.0)
            parsed = {
                "score": score,
                "predicted_mechanism": pred_mech,
                "mechanism_score_map": mech_scores,
                "heuristic_agreement": heuristic_score,
                "cue_score": cue_score,
                "compatibility_prior": compat,
                "match": match,
                "heuristic_detail": heuristic_detail,
                "mode": "deterministic_mech_judge",
            }
        elif "label" in stage_id:
            pred_label, label_scores = predict_heuristic_label(scenario, mechanism, full_text)
            heuristic_score, heuristic_detail = heuristic_agreement_score(
                scenario=scenario,
                mechanism=mechanism,
                label=label,
                text=full_text,
                scenario_policy=self.scenario_policy,
                anchors=anchors,
            )
            compat = compatibility_prior_score(scenario, mechanism, label)
            label_signal = clamp(self._to_float(label_scores.get(label, 0.0), 0.0) / 3.0, 0.0, 1.0)
            label_match = 1.0 if str(pred_label).strip().lower() == label.strip().lower() else 0.0
            if scenario == "affection":
                # Affection label space is noisy. Re-balance to avoid disgusted collapse.
                score = clamp(0.18 * label_match + 0.36 * label_signal + 0.26 * heuristic_score + 0.20 * compat, 0.0, 1.0)
                if label.strip().lower() == "disgusted":
                    score = max(0.0, score - 0.08)
                elif label.strip().lower() in {"angry", "bad", "happy", "fearful", "sad"}:
                    score = min(1.0, score + 0.04)
            elif scenario == "intent":
                score = clamp(0.28 * label_match + 0.24 * label_signal + 0.30 * heuristic_score + 0.18 * compat, 0.0, 1.0)
                l = label.strip().lower()
                if l in {"provoke", "mitigate"}:
                    score = max(0.0, score - 0.10)
                if l in {"mock", "alienate", "condemn", "intimidate", "dominate", "denounce"}:
                    score = min(1.0, score + 0.06)
            else:
                score = clamp(0.55 * label_match + 0.20 * label_signal + 0.15 * heuristic_score + 0.10 * compat, 0.0, 1.0)
            llm_label_score = None
            llm_reason_short = ""
            if scenario == "affection":
                llm_prompt_id = prompt_id or "R3_judge_label"
                llm_parsed, llm_artifact = run_prompt_json(
                    backend=self.backend,
                    prompt_store=self.prompt_store,
                    stage_id=f"{stage_id}_llm",
                    prompt_id=llm_prompt_id,
                    prompt_vars={
                        "scenario": scenario,
                        "text": text,
                        "audio_caption": audio_caption,
                        "anchors": anchors,
                        "retrieved_context": prompt_vars.get("retrieved_context", []) or [],
                        "candidate": candidate,
                    },
                    max_retries=self.max_retries,
                    media_items=media_items,
                    temperature_override=0.0,
                )
                llm_label_score = clamp(self._to_float(llm_parsed.get("score", 0.0), 0.0), 0.0, 1.0)
                llm_reason_short = clip_text(str(llm_parsed.get("reason_short") or ""), 120)
                score = clamp(0.35 * score + 0.65 * llm_label_score, 0.0, 1.0)
                parsed = {
                    "score": score,
                    "predicted_label": pred_label,
                    "label_score_map": label_scores,
                    "heuristic_agreement": heuristic_score,
                    "compatibility_prior": compat,
                    "label_signal": label_signal,
                    "match": label_match,
                    "heuristic_detail": heuristic_detail,
                    "llm_label_score": llm_label_score,
                    "llm_reason_short": llm_reason_short,
                    "llm_prompt_status": llm_artifact.status,
                    "mode": "affection_hybrid_label_judge",
                }
                artifact = self._artifact(stage_id, parsed, notes="affection deterministic+llm label fusion")
                return clamp(self._to_float(parsed.get("score", 0.0), 0.0), 0.0, 1.0), artifact, parsed
            parsed = {
                "score": score,
                "predicted_label": pred_label,
                "label_score_map": label_scores,
                "heuristic_agreement": heuristic_score,
                "compatibility_prior": compat,
                "label_signal": label_signal,
                "match": label_match,
                "heuristic_detail": heuristic_detail,
                "mode": "deterministic_label_judge",
            }
        else:
            subject = str(candidate.get("subject", ""))
            target = str(candidate.get("target", ""))
            subject_match = 1.0 if subject and subject in subject_options else 0.0
            target_match = 1.0 if target and target in target_options else 0.0
            anchor_bonus = 0.0
            if anchors.get("subject_anchor") and str(anchors.get("subject_anchor")).strip().lower() == subject.strip().lower():
                anchor_bonus += 0.25
            if anchors.get("target_anchor") and str(anchors.get("target_anchor")).strip().lower() == target.strip().lower():
                anchor_bonus += 0.25
            score = clamp(0.45 * subject_match + 0.45 * target_match + 0.10 * anchor_bonus, 0.0, 1.0)
            parsed = {
                "score": score,
                "subject_match": subject_match,
                "target_match": target_match,
                "anchor_bonus": anchor_bonus,
                "mode": "deterministic_role_judge",
            }
        artifact = self._artifact(stage_id, parsed, notes="deterministic heuristic judge")
        return clamp(self._to_float(parsed.get("score", 0.0), 0.0), 0.0, 1.0), artifact, parsed

    def _build_deterministic_view_candidates(
        self,
        sample: SampleInput,
        scenario: str,
        anchors: Dict[str, Any],
        text: str,
    ) -> List[Dict[str, Any]]:
        subject = anchors.get("subject_anchor") or (sample.subject_options[0] if sample.subject_options else "")
        target = anchors.get("target_anchor") or (sample.target_options[0] if sample.target_options else "")
        best_mech, mech_scores = predict_heuristic_mechanism(scenario, text, self.scenario_policy)
        mech_ranked = sorted(mech_scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
        best_label, label_scores = predict_heuristic_label(scenario, best_mech, text)
        label_ranked = sorted(label_scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
        allowed_mechs = list(VALID_MECHANISMS.get(scenario, []) or [])
        allowed_labels = list(VALID_LABELS.get(scenario, []) or [])

        def _normalize_candidate(mech: str, label: str, source: str, rationale: str) -> Dict[str, Any]:
            mech = self._pick_taxonomy(mech, allowed_mechs)
            label = self._pick_taxonomy(label, allowed_labels)
            heuristic_agreement, heuristic_detail = heuristic_agreement_score(
                scenario=scenario,
                mechanism=mech,
                label=label,
                text=text,
                scenario_policy=self.scenario_policy,
                anchors=anchors,
            )
            return {
                "subject": subject,
                "target": target,
                "mechanism": mech,
                "label": label,
                "confidence": clamp(0.45 + 0.40 * heuristic_agreement, 0.0, 1.0),
                "rationale": rationale,
                "source": source,
                "heuristic_agreement": heuristic_agreement,
                "heuristic_detail": {
                    "predicted_mechanism": best_mech,
                    "predicted_label": best_label,
                    "mechanism_scores": mech_scores,
                    "label_scores": label_scores,
                    "ranked_mechanisms": mech_ranked[:4],
                    "ranked_labels": label_ranked[:4],
                    **heuristic_detail,
                },
            }

        candidates: List[Dict[str, Any]] = []
        candidates.append(_normalize_candidate(best_mech, best_label, "heuristic_primary", "scenario cue fusion"))

        if len(mech_ranked) > 1:
            alt_mech = mech_ranked[1][0]
            alt_label, _ = predict_heuristic_label(scenario, alt_mech, text)
            candidates.append(_normalize_candidate(alt_mech, alt_label, "heuristic_mech_alt", "alternative mechanism cue"))

        if len(label_ranked) > 1:
            alt_label = label_ranked[1][0]
            candidates.append(_normalize_candidate(best_mech, alt_label, "heuristic_label_alt", "alternative label cue"))

        if scenario == "affection":
            # Affection often collapses to one dominant label; keep more label branches alive for reranking.
            for alt_label, _ in label_ranked[1:4]:
                candidates.append(
                    _normalize_candidate(
                        best_mech,
                        alt_label,
                        "heuristic_affection_label_diversity",
                        "affection label diversity branch",
                    )
                )
            for alt_mech, _ in mech_ranked[:2]:
                alt_best_label, alt_score_map = predict_heuristic_label(scenario, alt_mech, text)
                alt_label_ranked = sorted(alt_score_map.items(), key=lambda x: (x[1], x[0]), reverse=True)
                candidates.append(
                    _normalize_candidate(
                        alt_mech,
                        alt_best_label,
                        "heuristic_affection_cross_mech",
                        "affection cross mechanism branch",
                    )
                )
                for alt_label, _ in alt_label_ranked[:2]:
                    candidates.append(
                        _normalize_candidate(
                            alt_mech,
                            alt_label,
                            "heuristic_affection_cross_mech_label",
                            "affection cross mechanism-label branch",
                        )
                    )
            # Explicit fearful branch to reduce missing-coverage cases.
            fearful_cues = ["afraid", "fear", "scared", "anxious", "panic", "danger", "threat"]
            if any(cue in text.lower() for cue in fearful_cues):
                for mech_name in [best_mech] + [m for m, _ in mech_ranked[:2] if m != best_mech]:
                    candidates.append(
                        _normalize_candidate(
                            mech_name,
                            "fearful",
                            "heuristic_affection_fearful_recall",
                            "affection fearful recall branch",
                        )
                    )
        elif scenario == "intent":
            # Intent gets a broad label sweep to avoid generic collapse to provoke/mitigate.
            for mech_name in [m for m, _ in mech_ranked[:3]]:
                alt_best_label, alt_score_map = predict_heuristic_label(scenario, mech_name, text)
                alt_label_ranked = sorted(alt_score_map.items(), key=lambda x: (x[1], x[0]), reverse=True)
                candidates.append(
                    _normalize_candidate(
                        mech_name,
                        alt_best_label,
                        "heuristic_intent_cross_mech",
                        "intent cross mechanism branch",
                    )
                )
                for alt_label, _ in alt_label_ranked[:4]:
                    candidates.append(
                        _normalize_candidate(
                            mech_name,
                            alt_label,
                            "heuristic_intent_label_sweep",
                            "intent label sweep branch",
                        )
                    )
            for forced_label in ["mock", "alienate", "condemn", "intimidate", "dominate", "denounce", "provoke", "mitigate"]:
                candidates.append(
                    _normalize_candidate(
                        best_mech,
                        forced_label,
                        "heuristic_intent_forced_label",
                        "intent forced label anti-collapse branch",
                    )
                )

        compat_pairs: List[Tuple[float, str, str]] = []
        for mech in allowed_mechs:
            row = compatibility_prior_score(scenario, mech, "")
            for label in allowed_labels:
                compat = compatibility_prior_score(scenario, mech, label)
                compat_pairs.append((compat + 0.05 * row, mech, label))
        compat_pairs.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        if compat_pairs:
            _, mech, label = compat_pairs[0]
            candidates.append(_normalize_candidate(mech, label, "heuristic_compat_prior", "compatibility prior anchor"))
        if len(compat_pairs) > 1:
            _, mech, label = compat_pairs[1]
            candidates.append(_normalize_candidate(mech, label, "heuristic_compat_runner_up", "compatibility runner-up"))

        fallback_mech = allowed_mechs[0] if allowed_mechs else best_mech
        fallback_label = allowed_labels[0] if allowed_labels else best_label
        candidates.append(_normalize_candidate(fallback_mech, fallback_label, "heuristic_fallback", "closed-set fallback"))

        return candidates

    def _apply_affection_structural_rerank(self, scored: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not scored:
            return {"applied": False, "reason": "empty"}
        label_counts: Dict[str, int] = {}
        for row in scored:
            lb = str(row.get("label", "")).strip().lower()
            if lb:
                label_counts[lb] = label_counts.get(lb, 0) + 1
        total = max(len(scored), 1)
        disgust_share = float(label_counts.get("disgusted", 0)) / float(total)
        adjustments = 0
        for row in scored:
            comp = row.setdefault("components", {})
            label = str(row.get("label", "")).strip().lower()
            jm = float(comp.get("judge_mech", 0.0))
            jl = float(comp.get("judge_label", 0.0))
            heur = float(comp.get("heuristic_agreement", 0.0))
            penalty = 0.0
            bonus = 0.0
            # hard anti-collapse for disgusted when it dominates candidate pool
            if label == "disgusted" and disgust_share >= 0.50:
                penalty += 0.10 + 0.20 * (disgust_share - 0.50)
            # encourage alternatives when mechanism evidence is solid
            if label in {"angry", "bad", "happy", "fearful", "sad"} and jm >= 0.65:
                bonus += 0.04 + 0.05 * max(0.0, jl - 0.25)
            if label == "fearful" and (jm >= 0.62 or heur >= 0.60):
                bonus += 0.05
            if bonus or penalty:
                row["total_score"] = float(row.get("total_score", 0.0)) + bonus - penalty
                comp["affection_structural_bonus"] = bonus
                comp["affection_structural_penalty"] = penalty
                adjustments += 1
        return {
            "applied": True,
            "disgust_share": disgust_share,
            "label_counts": label_counts,
            "adjustments": adjustments,
        }

    def _apply_intent_structural_rerank(self, scored: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not scored:
            return {"applied": False, "reason": "empty"}
        label_counts: Dict[str, int] = {}
        for row in scored:
            lb = str(row.get("label", "")).strip().lower()
            if lb:
                label_counts[lb] = label_counts.get(lb, 0) + 1
        total = max(len(scored), 1)
        generic_share = float(label_counts.get("provoke", 0) + label_counts.get("mitigate", 0)) / float(total)
        adjustments = 0
        for row in scored:
            comp = row.setdefault("components", {})
            label = str(row.get("label", "")).strip().lower()
            jm = float(comp.get("judge_mech", 0.0))
            jl = float(comp.get("judge_label", 0.0))
            heur = float(comp.get("heuristic_agreement", 0.0))
            penalty = 0.0
            bonus = 0.0
            if label in {"provoke", "mitigate"} and generic_share >= 0.45:
                penalty += 0.08 + 0.18 * (generic_share - 0.45)
            if label in {"mock", "alienate", "condemn", "intimidate", "dominate", "denounce"}:
                bonus += 0.04 + 0.05 * max(0.0, heur - 0.40)
                if jm >= 0.66:
                    bonus += 0.04
            if label == "alienate" and (heur >= 0.62 or jl >= 0.52):
                bonus += 0.05
            if bonus or penalty:
                row["total_score"] = float(row.get("total_score", 0.0)) + bonus - penalty
                comp["intent_structural_bonus"] = bonus
                comp["intent_structural_penalty"] = penalty
                adjustments += 1
        return {
            "applied": True,
            "generic_share": generic_share,
            "label_counts": label_counts,
            "adjustments": adjustments,
        }

    def _decode_affection_dual_head(self, scored: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not scored:
            return (
                {
                    "subject": "",
                    "target": "",
                    "mechanism": "",
                    "label": "",
                },
                {"applied": False, "reason": "empty"},
            )

        def _mech_head_score(row: Dict[str, Any]) -> float:
            comp = row.get("components", {}) or {}
            return (
                1.0 * float(comp.get("judge_mech", 0.0))
                + 0.30 * float(comp.get("heuristic_agreement", 0.0))
                + 0.30 * float(comp.get("rule_cue", 0.0))
                - 0.40 * float(comp.get("penalty", 0.0))
            )

        mech_candidate = max(scored, key=lambda x: (_mech_head_score(x), float(x.get("total_score", 0.0))))
        label_votes: Dict[str, float] = {}
        for row in scored:
            comp = row.get("components", {}) or {}
            label = str(row.get("label", "")).strip()
            if not label:
                continue
            vote = (
                0.80 * float(comp.get("judge_label", 0.0))
                + 0.20 * float(comp.get("heuristic_agreement", 0.0))
                - 0.40 * float(comp.get("penalty", 0.0))
            )
            label_votes[label] = label_votes.get(label, 0.0) + vote

        if label_votes:
            ranked_votes = sorted(label_votes.items(), key=lambda x: (x[1], x[0]), reverse=True)
            best_label = ranked_votes[0][0]
            second_vote = ranked_votes[1][1] if len(ranked_votes) > 1 else ranked_votes[0][1]
            margin = float(ranked_votes[0][1] - second_vote)
        else:
            ranked_votes = []
            best_label = str(mech_candidate.get("label", ""))
            margin = 0.0

        return (
            {
                "subject": str(mech_candidate.get("subject", "")),
                "target": str(mech_candidate.get("target", "")),
                "mechanism": str(mech_candidate.get("mechanism", "")),
                "label": best_label,
            },
            {
                "applied": True,
                "mechanism_source_label": str(mech_candidate.get("label", "")),
                "mechanism_source_total_score": float(mech_candidate.get("total_score", 0.0)),
                "label_votes": ranked_votes[:8],
                "label_margin": margin,
            },
        )

    def run_sample(self, sample: SampleInput, backend_meta: Dict[str, Any]) -> Dict[str, Any]:
        artifacts: List[StageArtifact] = []
        trace: Dict[str, Any] = {}
        scenario = str(sample.scenario or "").strip().lower()
        if scenario not in VALID_MECHANISMS:
            scenario = "affection"
        media_manifest = self.media_resolver.resolve(sample_id=sample.sample_id, scenario=scenario, media=sample.media)
        media_items = build_media_items(media_manifest, max_frames=self.max_video_frames)
        try:
            anchors = build_anchor_payload(sample, media_manifest=media_manifest)
            artifacts.append(self._artifact("M0_anchor_parse", anchors))

            retrieved = retrieve_similar_entries(
                index=self.memory_index,
                scenario=scenario,
                query_tf=anchors.get("token_freq", {}) or {},
                query_subject_options=sample.subject_options,
                query_target_options=sample.target_options,
                query_sample_id=sample.sample_id,
                top_k=self.top_k,
                rerank_k=self.rerank_k,
                loo=self.loo,
            )
            artifacts.append(
                self._artifact(
                    "M1_retrieve",
                    {
                        "top_k": self.top_k,
                        "rerank_k": self.rerank_k,
                        "retrieved": [
                            {"sample_id": x.get("sample_id"), "retrieval_score": x.get("retrieval_score", 0.0)}
                            for x in retrieved
                        ],
                    },
                )
            )
            retrieved_context = [
                {
                    "sample_id": x.get("sample_id"),
                    "text": x.get("text", ""),
                    "audio_caption": x.get("audio_caption", ""),
                    "retrieval_score": x.get("retrieval_score", 0.0),
                }
                for x in retrieved[:3]
            ]

            full_text = f"{sample.text}\n{media_manifest.get('audio_caption', '')}".strip()
            heuristic_candidates, heuristic_trace = self._build_heuristic_candidates(
                sample=sample,
                scenario=scenario,
                anchors=anchors,
                text=full_text,
            )
            artifacts.append(self._artifact("M1.5_heuristic_candidates", heuristic_trace))
            llm_candidates, llm_artifacts, llm_summary = self._build_llm_candidates(
                sample=sample,
                scenario=scenario,
                anchors=anchors,
                retrieved_context=retrieved_context,
                media_items=media_items,
                audio_caption=media_manifest.get("audio_caption", ""),
            )
            artifacts.extend(llm_artifacts)
            artifacts.append(self._artifact("M2_candidate_generate_summary", llm_summary, notes="llm candidate branch"))

            baseline_probe: Dict[str, Any] | None = None
            if scenario == "affection":
                baseline_probe, baseline_artifact = self._run_baseline_probe(
                    sample=sample,
                    scenario=scenario,
                    media_items=media_items,
                    audio_caption=media_manifest.get("audio_caption", ""),
                )
                artifacts.append(baseline_artifact)
                artifacts.append(
                    self._artifact(
                        "M2.5_baseline_probe_summary",
                        {
                            "label": baseline_probe.get("label", ""),
                            "mechanism": baseline_probe.get("mechanism", ""),
                            "confidence": baseline_probe.get("confidence", 0.0),
                        },
                        notes="affection baseline probe",
                    )
                )

            candidates_raw: List[Dict[str, Any]] = list(heuristic_candidates)
            candidates_raw.extend(llm_candidates)
            candidates_raw.extend(
                [
                    {
                        "subject": x.get("subject", heuristic_trace.get("subject", "")),
                        "target": x.get("target", heuristic_trace.get("target", "")),
                        "mechanism": x.get("mechanism", ""),
                        "label": x.get("label", ""),
                        "confidence": x.get("confidence", 0.0),
                        "rationale": x.get("rationale", ""),
                        "source": x.get("source", "deterministic_heuristic"),
                        "heuristic_agreement": x.get("heuristic_agreement", 0.0),
                        "heuristic_detail": x.get("heuristic_detail", {}),
                    }
                    for x in self._build_deterministic_view_candidates(
                        sample=sample,
                        scenario=scenario,
                        anchors=anchors,
                        text=full_text,
                    )
                ]
            )
            if baseline_probe is not None:
                candidates_raw.append(
                    {
                        "subject": baseline_probe.get("subject", heuristic_trace.get("subject", "")),
                        "target": baseline_probe.get("target", heuristic_trace.get("target", "")),
                        "mechanism": baseline_probe.get("mechanism", ""),
                        "label": baseline_probe.get("label", ""),
                        "confidence": baseline_probe.get("confidence", 0.0),
                        "rationale": baseline_probe.get("rationale", "baseline direct probe"),
                        "source": "baseline_probe",
                        "heuristic_agreement": 0.0,
                        "heuristic_detail": {"probe_prompt_id": baseline_probe.get("probe_prompt_id", "PB0_baseline_direct")},
                    }
                )

            candidates = self._dedup_candidates(candidates_raw, sample=sample)
            if not candidates:
                raise RuntimeError("candidate generation failed")

            scored: List[Dict[str, Any]] = []
            for i, cand in enumerate(candidates, start=1):
                if str(cand.get("source", "")).strip() == "baseline_probe":
                    repaired_label = str(cand.get("label", ""))
                    repair_info = {
                        "scenario": scenario,
                        "mechanism": str(cand.get("mechanism", "")),
                        "original_label": str(cand.get("label", "")),
                        "repaired": False,
                        "reason": "skip_for_baseline_probe",
                    }
                else:
                    repaired_label, repair_info = repair_inconsistent_label(
                        scenario=scenario,
                        mechanism=str(cand.get("mechanism", "")),
                        label=str(cand.get("label", "")),
                        text=full_text,
                    )
                cand_eval = dict(cand)
                cand_eval["label"] = repaired_label
                cand_eval["repair_info"] = repair_info
                mech_score, a_mech, _ = self._judge_score(
                    stage_id=f"M3_judge_mech_{i}",
                    prompt_vars={
                        "scenario": scenario,
                        "text": sample.text,
                        "audio_caption": media_manifest.get("audio_caption", ""),
                        "anchors": anchors,
                        "retrieved_context": retrieved_context,
                        "candidate": cand_eval,
                    },
                    media_items=media_items,
                )
                artifacts.append(a_mech)
                label_score, a_label, _ = self._judge_score(
                    stage_id=f"M3_judge_label_{i}",
                    prompt_vars={
                        "scenario": scenario,
                        "text": sample.text,
                        "audio_caption": media_manifest.get("audio_caption", ""),
                        "anchors": anchors,
                        "retrieved_context": retrieved_context,
                        "candidate": cand_eval,
                    },
                    media_items=media_items,
                )
                artifacts.append(a_label)
                role_score, a_role, _ = self._judge_score(
                    stage_id=f"M3_judge_role_{i}",
                    prompt_vars={
                        "scenario": scenario,
                        "text": sample.text,
                        "subject_options": sample.subject_options,
                        "target_options": sample.target_options,
                        "candidate": cand_eval,
                    },
                    media_items=media_items,
                )
                artifacts.append(a_role)

                retrieve_support = self._retrieve_support(cand_eval, retrieved)
                cue_score = rule_cue_score(
                    scenario=scenario,
                    mechanism=str(cand_eval.get("mechanism", "")),
                    text=full_text,
                    scenario_policy=self.scenario_policy,
                )
                heuristic_score, heuristic_detail = heuristic_agreement_score(
                    scenario=scenario,
                    mechanism=str(cand_eval.get("mechanism", "")),
                    label=str(cand_eval.get("label", "")),
                    text=full_text,
                    scenario_policy=self.scenario_policy,
                    anchors=anchors,
                )
                if cand_eval.get("source", "").startswith("heuristic"):
                    heuristic_score = max(heuristic_score, self._to_float(cand_eval.get("heuristic_agreement", 0.0), 0.0))
                    heuristic_detail = {**heuristic_detail, **(cand_eval.get("heuristic_detail", {}) or {})}
                penalty, penalty_detail = score_penalty_components(
                    scenario=scenario,
                    mechanism=str(cand_eval.get("mechanism", "")),
                    label=str(cand_eval.get("label", "")),
                    subject=str(cand_eval.get("subject", "")),
                    target=str(cand_eval.get("target", "")),
                    subject_options=sample.subject_options,
                    target_options=sample.target_options,
                    parser_non_empty=bool(anchors.get("parser_non_empty", False)),
                    text=full_text,
                    anchors=anchors,
                    constraint_config=self.constraint_config,
                )
                label_policy_adjustment = {
                    "label_bonus": 0.0,
                    "penalty_add": 0.0,
                    "reason": "none",
                }
                if scenario == "affection" and baseline_probe is not None:
                    label_policy_adjustment = self._affection_candidate_adjustment(
                        text=full_text,
                        candidate_label=str(cand_eval.get("label", "")),
                        baseline_label=str(baseline_probe.get("label", "")),
                    )
                    label_score = clamp(
                        label_score + float(label_policy_adjustment.get("label_bonus", 0.0)),
                        0.0,
                        1.0,
                    )
                    policy_penalty = float(label_policy_adjustment.get("penalty_add", 0.0))
                    if policy_penalty > 0.0:
                        penalty += policy_penalty
                        penalty_detail["affection_policy_penalty"] = policy_penalty
                total = compute_total_score(
                    weights=self.weights,
                    retrieve_support=retrieve_support,
                    judge_mech=mech_score,
                    judge_label=label_score,
                    judge_role=role_score,
                    rule_cue=cue_score,
                    penalty=penalty,
                    heuristic_agreement=heuristic_score,
                )
                scored.append(
                    {
                        **cand_eval,
                        "components": {
                            "retrieve_support": retrieve_support,
                            "judge_mech": mech_score,
                            "judge_label": label_score,
                            "judge_role": role_score,
                            "rule_cue": cue_score,
                            "heuristic_agreement": heuristic_score,
                            "penalty": penalty,
                            "penalty_detail": penalty_detail,
                            "label_repair": repair_info,
                            "label_policy_adjustment": label_policy_adjustment,
                        },
                        "heuristic_detail": heuristic_detail,
                        "total_score": float(total),
                    }
                )

            if scenario == "affection" and scored:
                trace["affection_structural_rerank"] = self._apply_affection_structural_rerank(scored)
            elif scenario == "intent" and scored:
                trace["intent_structural_rerank"] = self._apply_intent_structural_rerank(scored)

            scored.sort(key=lambda x: float(x.get("total_score", 0.0)), reverse=True)
            chosen_idx = 0
            should_rejudge, branch_reason, branch_metrics = should_rejudge_branch(
                scored,
                accept_total_threshold=self.accept_total_threshold,
                min_margin_threshold=self.min_margin_threshold,
                min_component_threshold=self.min_component_threshold,
                min_component_gap=self.min_component_gap,
            )
            trace["branch_rejudge"] = {
                "triggered": should_rejudge,
                "reason": branch_reason,
                **branch_metrics,
            }
            branch_rejudge_record = trace["branch_rejudge"]
            if should_rejudge:
                if self._should_attempt_llm_tie_break(branch_reason, branch_metrics):
                    llm_idx, tie_break, artifact = self._llm_tie_break(
                        scenario=scenario,
                        text=sample.text,
                        audio_caption=media_manifest.get("audio_caption", ""),
                        scored=scored,
                        media_items=media_items,
                    )
                    branch_rejudge_record["tie_break"] = tie_break
                    artifacts.append(artifact)
                    if llm_idx is None:
                        chosen_idx, fallback_tie_break = self._deterministic_tie_break(scored, branch_reason)
                        branch_rejudge_record["tie_break_fallback"] = fallback_tie_break
                        artifacts.append(self._artifact("M5_tie_break_fallback", fallback_tie_break, notes="llm tie-break fallback"))
                    else:
                        chosen_idx = llm_idx
                else:
                    chosen_idx, tie_break = self._deterministic_tie_break(scored, branch_reason)
                    branch_rejudge_record["tie_break"] = tie_break
                    artifacts.append(self._artifact("M5_tie_break", tie_break, notes="low-confidence branch rejudge"))
            chosen = scored[chosen_idx]
            second_score = float(scored[1]["total_score"]) if len(scored) > 1 else float(chosen["total_score"])
            margin = float(chosen["total_score"]) - float(second_score)
            confidence = clamp(
                0.32
                + 0.38 * float(chosen["components"]["judge_label"])
                + 0.22 * float(chosen["components"]["judge_mech"])
                + 0.18 * float(chosen["components"]["heuristic_agreement"])
                + 0.10 * max(0.0, margin),
                0.0,
                1.0,
            )
            final_subject = str(chosen.get("subject", ""))
            final_target = str(chosen.get("target", ""))
            final_mechanism = str(chosen.get("mechanism", ""))
            final_label = str(chosen.get("label", ""))
            affection_dual_head: Dict[str, Any] | None = None
            if scenario == "affection" and False:
                dual_pred, affection_dual_head = self._decode_affection_dual_head(scored)
                final_subject = dual_pred.get("subject", final_subject)
                final_target = dual_pred.get("target", final_target)
                final_mechanism = dual_pred.get("mechanism", final_mechanism)
                final_label = dual_pred.get("label", final_label)
                artifacts.append(
                    self._artifact(
                        "M6_affection_dual_head_decode",
                        affection_dual_head,
                        notes="mechanism head + label vote head",
                    )
                )
            affection_arbitration: Dict[str, Any] | None = None
            if scenario == "affection" and baseline_probe is not None:
                if float((affection_dual_head or {}).get("label_margin", 0.0)) < 0.06:
                    final_label, affection_arbitration = self._arbitrate_affection_label(
                        text=full_text,
                        rjg_label=final_label,
                        baseline_label=str(baseline_probe.get("label", "")),
                    )
                else:
                    affection_arbitration = {
                        "reason": "skip_arbitration_dual_head_confident",
                        "selected_label": final_label,
                        "dual_head_margin": float((affection_dual_head or {}).get("label_margin", 0.0)),
                        "baseline_label": str(baseline_probe.get("label", "")),
                    }
                artifacts.append(
                    self._artifact(
                        "M6_affection_label_arbitration",
                        affection_arbitration,
                        notes="baseline+rjg label fusion",
                    )
                )
            final_prediction = {
                "subject": final_subject,
                "target": final_target,
                "mechanism": final_mechanism,
                "label": final_label,
                "confidence": confidence,
                "decision_rationale_short": clip_text(str(chosen.get("rationale", "")), 300),
            }
            trace = {
                "mode": "rjg_v1",
                "weights": asdict(self.weights),
                "retrieved_ids": [x.get("sample_id") for x in retrieved],
                "candidate_scores": scored,
                "heuristic_candidates": heuristic_candidates,
                "heuristic_trace": heuristic_trace,
                "branch_rejudge": branch_rejudge_record,
                "selected_index": int(chosen_idx),
                "tie_margin": self.tie_margin,
                "media_manifest": media_manifest,
                "baseline_probe": baseline_probe,
                "affection_dual_head_decode": affection_dual_head,
                "affection_label_arbitration": affection_arbitration,
                "full_text_for_policy": full_text if scenario == "affection" else "",
            }
            return {
                "sample_id": sample.sample_id,
                "scenario": scenario,
                "final_prediction": final_prediction,
                "stage_artifacts": [asdict(x) for x in artifacts],
                "backend_meta": backend_meta,
                "trace": trace,
                "error": None,
            }
        except Exception as exc:  # pragma: no cover
            return {
                "sample_id": sample.sample_id,
                "scenario": scenario,
                "final_prediction": {
                    "subject": sample.subject_options[0] if sample.subject_options else "",
                    "target": sample.target_options[0] if sample.target_options else "",
                    "mechanism": (VALID_MECHANISMS.get(scenario) or [""])[0],
                    "label": (VALID_LABELS.get(scenario) or [""])[0],
                    "confidence": 0.0,
                    "decision_rationale_short": "",
                },
                "stage_artifacts": [asdict(x) for x in artifacts],
                "backend_meta": backend_meta,
                "trace": trace,
                "error": clip_text(str(exc), 600),
            }
