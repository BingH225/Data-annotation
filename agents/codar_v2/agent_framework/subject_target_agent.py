from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")

_POWER_HINTS = ("boss", "manager", "leader", "teacher", "parent", "senior")
_INTIMACY_HINTS = ("friend", "family", "partner", "couple", "close", "dear")
_HISTORY_HINTS = ("again", "always", "before", "previously", "history")
_SPEAKER_HINTS = ("i", "me", "my", "we", "us", "our", "speaker", "author", "poster", "op")
_TARGET_ADDRESS_HINTS = ("you", "your", "u", "them", "they", "those", "that group")


def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


def _tokenize(text: Any) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(str(text or ""))]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _contains_any(text: str, hints: Tuple[str, ...]) -> bool:
    lowered = _norm(text)
    return any(h in lowered for h in hints)


def _slot_index(slot: str, prefix: str) -> int:
    m = re.match(rf"^{re.escape(prefix)}(\d+)$", str(slot or "").strip().lower())
    if not m:
        return 999
    try:
        return int(m.group(1))
    except Exception:
        return 999


def _overlap_score(text_tokens: List[str], option: str) -> float:
    opt_tokens = _tokenize(option)
    if not opt_tokens:
        return 0.0
    tset = set(text_tokens)
    hits = sum(1 for x in opt_tokens if x in tset)
    return hits / len(opt_tokens)


def _raw_score(text: str, option: str) -> float:
    if not option:
        return 0.0
    lowered = _norm(text)
    key = _norm(option)
    score = 0.0
    if key and key in lowered:
        score += 1.8
    score += 1.2 * _overlap_score(_tokenize(text), option)
    if any(h in lowered for h in _SPEAKER_HINTS) and key in ("speaker", "author", "poster", "op"):
        score += 0.4
    return score


def _exact_hit_slot(merged_text: str, slot_map: Dict[str, str], prefix: str) -> Optional[Tuple[str, str]]:
    lowered = _norm(merged_text)
    best_slot = None
    best_option = ""
    best_len = -1
    for slot, option in slot_map.items():
        key = _norm(option)
        if not key:
            continue
        if key in lowered:
            cur_len = len(key)
            if cur_len > best_len:
                best_len = cur_len
                best_slot = slot
                best_option = option
            elif cur_len == best_len and best_slot is not None:
                if _slot_index(slot, prefix) < _slot_index(best_slot, prefix):
                    best_slot = slot
                    best_option = option
    if best_slot is None:
        return None
    return best_slot, best_option


def _rank_slot_candidates(merged_text: str, slot_map: Dict[str, str], prefix: str) -> List[Tuple[str, str, float]]:
    ranked: List[Tuple[str, str, float]] = []
    for slot, option in slot_map.items():
        ranked.append((slot, option, _raw_score(merged_text, option)))
    ranked.sort(key=lambda x: (-float(x[2]), _slot_index(x[0], prefix)))
    return ranked


def _anchor_confidence(ranked: List[Tuple[str, str, float]], exact_hit: bool = False) -> float:
    if not ranked:
        return 0.0
    top = float(ranked[0][2])
    second = float(ranked[1][2]) if len(ranked) > 1 else 0.0
    if top <= 0.0:
        return 0.0
    margin = max(0.0, top - second)
    conf = 0.18 + 0.32 * min(1.0, top) + 0.50 * min(1.0, margin)
    if exact_hit:
        conf = max(conf, 0.78)
    return _clip01(conf)


def _coerce_to_slot(raw_value: Any, prefix: str, slot_map: Dict[str, str]) -> str | None:
    raw = _norm(raw_value)
    if not raw:
        return None
    allowed = set(slot_map.keys())
    if raw in allowed:
        return raw
    for slot, text in slot_map.items():
        if raw == _norm(text):
            return slot
    for slot in allowed:
        if slot in raw:
            return slot
    if raw.startswith(prefix):
        maybe = raw.replace(" ", "")
        if maybe in allowed:
            return maybe
    return None


@dataclass
class SubjectTargetPipelineResult:
    explicit_perception: Dict[str, Any]
    social_graph: Dict[str, Any]
    subject_anchor_slot: str
    target_anchor_slot: str
    subject_anchor_option: str
    target_anchor_option: str
    subject_confidence: float
    target_confidence: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoleProbeResult:
    subject_slot: Optional[str]
    target_slot: Optional[str]
    confidence: float
    reason_short: str
    raw_json: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SubjectTargetAgentPipeline:
    """
    Role-focused RJG-style pipeline:
    1) Explicit perception decomposition
    2) Subject-relations graph construction
    3) Role probe + deterministic role fusion for subject/target only
    """

    def __init__(
        self,
        strategy: str = "rjg_role_fusion",
        anchor_min_conf: float = 0.55,
        role_probe_min_conf: float = 0.55,
    ):
        self.strategy = str(strategy or "rjg_role_fusion").strip().lower()
        self.anchor_min_conf = float(anchor_min_conf)
        self.role_probe_min_conf = float(role_probe_min_conf)

    def run(
        self,
        *,
        text: str,
        audio_caption: str,
        subject_slots: Dict[str, str],
        target_slots: Dict[str, str],
        scenario: str = "",
        domain: str = "",
        culture: str = "",
    ) -> SubjectTargetPipelineResult:
        merged = f"{text or ''}\n{audio_caption or ''}".strip()
        subject_ranked = _rank_slot_candidates(merged, subject_slots, "subject")
        target_ranked = _rank_slot_candidates(merged, target_slots, "target")

        subject_exact = _exact_hit_slot(merged, subject_slots, "subject")
        target_exact = _exact_hit_slot(merged, target_slots, "target")

        if subject_exact is not None:
            subject_anchor_slot, subject_anchor_option = subject_exact
        elif subject_ranked:
            subject_anchor_slot, subject_anchor_option, _ = subject_ranked[0]
        else:
            subject_anchor_slot, subject_anchor_option = "subject0", ""

        if target_exact is not None:
            target_anchor_slot, target_anchor_option = target_exact
        elif target_ranked:
            target_anchor_slot, target_anchor_option, _ = target_ranked[0]
        else:
            target_anchor_slot, target_anchor_option = "target0", ""

        subject_conf = _anchor_confidence(subject_ranked, exact_hit=subject_exact is not None)
        target_conf = _anchor_confidence(target_ranked, exact_hit=target_exact is not None)

        explicit_perception = {
            "text_components": {
                "subject": subject_anchor_option,
                "object": target_anchor_option,
                "predicate": "implicit_social_expression",
                "attribute": str(scenario or ""),
                "adverbial": "multimodal",
            },
            "image_action": {
                "subject": subject_anchor_option,
                "background": str(domain or ""),
                "behavior": "context_dependent",
                "action": "address_or_evaluate_target",
            },
            "audio_caption": {
                "subject": subject_anchor_option,
                "object": target_anchor_option,
                "predicate": "tone_support",
                "attribute": "available" if _norm(audio_caption) else "",
                "adverbial": "prosody_inferred",
            },
        }

        relation_cues = {
            "power": "high" if _contains_any(merged, _POWER_HINTS) else "unknown",
            "intimacy": "high" if _contains_any(merged, _INTIMACY_HINTS) else "unknown",
            "history": "present" if _contains_any(merged, _HISTORY_HINTS) else "unknown",
            "culture": str(culture or ""),
            "domain": str(domain or ""),
        }
        social_graph = {
            "nodes": [
                {"id": "sub_anchor", "label": subject_anchor_option, "slot": subject_anchor_slot},
                {"id": "tgt_anchor", "label": target_anchor_option, "slot": target_anchor_slot},
            ],
            "edges": [
                {
                    "source": "sub_anchor",
                    "target": "tgt_anchor",
                    "relation": "addresses_or_evaluates",
                    "cues": relation_cues,
                }
            ],
            "relation_cues": relation_cues,
        }

        return SubjectTargetPipelineResult(
            explicit_perception=explicit_perception,
            social_graph=social_graph,
            subject_anchor_slot=subject_anchor_slot,
            target_anchor_slot=target_anchor_slot,
            subject_anchor_option=subject_anchor_option,
            target_anchor_option=target_anchor_option,
            subject_confidence=subject_conf,
            target_confidence=target_conf,
        )

    def build_prompt_block(
        self,
        result: SubjectTargetPipelineResult,
        *,
        include_recommendation: bool = False,
    ) -> str:
        payload: Dict[str, Any] = {
            "explicit_perception": result.explicit_perception,
            "subject_relations_graph": result.social_graph,
            "confidence_summary": {
                "subject_confidence": round(result.subject_confidence, 4),
                "target_confidence": round(result.target_confidence, 4),
            },
        }
        if include_recommendation:
            payload["recommended_slots"] = {
                "subject": result.subject_anchor_slot,
                "target": result.target_anchor_slot,
            }
        return (
            "Subject/Target Agent Context (for role understanding only):\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\nRule: Treat this as weak context evidence, never as a forced slot choice."
        )

    def build_role_probe_prompts(
        self,
        *,
        scenario: str,
        text: str,
        audio_caption: str,
        subject_slots: Dict[str, str],
        target_slots: Dict[str, str],
        result: SubjectTargetPipelineResult,
        mode: str = "balanced",
    ) -> Tuple[str, str]:
        mode_norm = str(mode or "balanced").strip().lower()
        if mode_norm not in {"balanced", "literal", "pragmatic"}:
            mode_norm = "balanced"
        system_prompt = (
            "You are a strict subject-target role resolver.\n"
            "You only output role slots and confidence.\n"
            "Do not output mechanism or label.\n"
            "Return JSON only with keys: subject, target, confidence, reason_short.\n"
            f"Role decision mode: {mode_norm}."
        )
        if mode_norm == "literal":
            mode_rules = (
                "Mode hint:\n"
                "- Prefer direct lexical and grammatical cues.\n"
                "- Prioritize explicit speaker/addressee mentions over inferred social background.\n"
            )
        elif mode_norm == "pragmatic":
            mode_rules = (
                "Mode hint:\n"
                "- Prefer pragmatic social-relation reading when literal cues are ambiguous.\n"
                "- Use role relations implied by image/audio context and social framing.\n"
            )
        else:
            mode_rules = (
                "Mode hint:\n"
                "- Balance literal lexical cues and pragmatic relation cues.\n"
                "- Avoid overfitting to either pure syntax or pure world-knowledge bias.\n"
            )
        user_prompt = (
            f"Scenario: {scenario}\n"
            f"Text: {text}\n"
            f"Audio Caption: {audio_caption}\n\n"
            "Subject slot mapping:\n"
            + "\n".join([f"{k} = {v}" for k, v in subject_slots.items()])
            + "\n\nTarget slot mapping:\n"
            + "\n".join([f"{k} = {v}" for k, v in target_slots.items()])
            + "\n\nInternal grounding context:\n"
            + self.build_prompt_block(result, include_recommendation=False)
            + "\n\nDecision rules:\n"
            + mode_rules
            + "\n"
            + "1) Keep subject on the speaker by default for speaker-centric text.\n"
            + "2) Move subject away from speaker only when evidence clearly supports another acting source.\n"
            + "3) Keep target on the most directly addressed/evaluated entity.\n"
            + "4) Prefer exact option grounding over abstract role names.\n"
            + "5) Never output names outside provided slots.\n\n"
            + "Return JSON only:\n"
            + "{\"subject\":\"subjectX\",\"target\":\"targetY\",\"confidence\":0.0,\"reason_short\":\"<=30 words\"}"
        )
        return system_prompt, user_prompt

    def parse_role_probe_prediction(
        self,
        raw_json: Dict[str, Any],
        *,
        subject_slots: Dict[str, str],
        target_slots: Dict[str, str],
    ) -> RoleProbeResult:
        raw_json = dict(raw_json or {})
        sub_slot = _coerce_to_slot(raw_json.get("subject"), "subject", subject_slots)
        tgt_slot = _coerce_to_slot(raw_json.get("target"), "target", target_slots)
        try:
            conf = float(raw_json.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        conf = _clip01(conf)
        reason_short = str(raw_json.get("reason_short", "") or "").strip()
        return RoleProbeResult(
            subject_slot=sub_slot,
            target_slot=tgt_slot,
            confidence=conf,
            reason_short=reason_short,
            raw_json=raw_json,
        )

    def _slot_evidence(self, merged_text: str, slot: Optional[str], slot_map: Dict[str, str]) -> float:
        if not slot:
            return 0.0
        option = slot_map.get(slot, "")
        raw = _raw_score(merged_text, option)
        return _clip01(raw / 2.0)

    def _pair_score(
        self,
        *,
        merged_text: str,
        subject_slot: Optional[str],
        target_slot: Optional[str],
        subject_slots: Dict[str, str],
        target_slots: Dict[str, str],
        result: SubjectTargetPipelineResult,
        source_conf: float,
    ) -> float:
        if not subject_slot or subject_slot not in subject_slots:
            return -1.0
        if not target_slot or target_slot not in target_slots:
            return -1.0

        subject_ev = self._slot_evidence(merged_text, subject_slot, subject_slots)
        target_ev = self._slot_evidence(merged_text, target_slot, target_slots)
        textual_support = 0.5 * subject_ev + 0.5 * target_ev

        anchor_reliability = 0.5 * _clip01(result.subject_confidence) + 0.5 * _clip01(result.target_confidence)
        anchor_match = 0.0
        if subject_slot == result.subject_anchor_slot:
            anchor_match += 0.5
        if target_slot == result.target_anchor_slot:
            anchor_match += 0.5
        anchor_term = anchor_reliability * anchor_match

        direction_hint = 0.0
        lowered = _norm(merged_text)
        if _contains_any(lowered, _TARGET_ADDRESS_HINTS) and target_ev >= 0.45:
            direction_hint = 0.06

        score = 0.36 * _clip01(source_conf) + 0.36 * textual_support + 0.28 * anchor_term + direction_hint
        return _clip01(score)

    def resolve(
        self,
        *,
        raw_subject: Any,
        raw_target: Any,
        subject_slots: Dict[str, str],
        target_slots: Dict[str, str],
        result: SubjectTargetPipelineResult,
        probe_result: RoleProbeResult | None = None,
        extra_candidates: Optional[List[Dict[str, Any]]] = None,
        text: str = "",
        audio_caption: str = "",
    ) -> Tuple[str, str, Dict[str, Any]]:
        model_sub = _coerce_to_slot(raw_subject, "subject", subject_slots)
        model_tgt = _coerce_to_slot(raw_target, "target", target_slots)
        merged = f"{text or ''}\n{audio_caption or ''}".strip()

        if self.strategy == "anchor_only":
            final_sub = result.subject_anchor_slot
            final_tgt = result.target_anchor_slot
            detail = {
                "strategy": self.strategy,
                "raw_subject": raw_subject,
                "raw_target": raw_target,
                "model_subject_slot": model_sub,
                "model_target_slot": model_tgt,
                "final_subject_slot": final_sub,
                "final_target_slot": final_tgt,
                "subject_source": "anchor_only",
                "target_source": "anchor_only",
            }
            return final_sub, final_tgt, detail

        probe_sub = probe_result.subject_slot if probe_result is not None else None
        probe_tgt = probe_result.target_slot if probe_result is not None else None
        probe_conf = float(probe_result.confidence) if probe_result is not None else 0.0

        candidates: List[Dict[str, Any]] = []

        if model_sub is not None and model_tgt is not None:
            candidates.append(
                {
                    "name": "model",
                    "subject": model_sub,
                    "target": model_tgt,
                    "source_conf": 0.60,
                }
            )
        if probe_sub is not None and probe_tgt is not None:
            candidates.append(
                {
                    "name": "role_probe",
                    "subject": probe_sub,
                    "target": probe_tgt,
                    "source_conf": max(0.45, probe_conf),
                }
            )
        candidates.append(
            {
                "name": "anchor_pair",
                "subject": result.subject_anchor_slot,
                "target": result.target_anchor_slot,
                "source_conf": 0.5 * (_clip01(result.subject_confidence) + _clip01(result.target_confidence)),
            }
        )
        if model_sub is not None and probe_tgt is not None:
            candidates.append(
                {
                    "name": "hybrid_model_sub_probe_tgt",
                    "subject": model_sub,
                    "target": probe_tgt,
                    "source_conf": 0.5 * (0.60 + max(0.45, probe_conf)),
                }
            )
        if probe_sub is not None and model_tgt is not None:
            candidates.append(
                {
                    "name": "hybrid_probe_sub_model_tgt",
                    "subject": probe_sub,
                    "target": model_tgt,
                    "source_conf": 0.5 * (0.60 + max(0.45, probe_conf)),
                }
            )

        for idx, cand in enumerate(list(extra_candidates or []), start=1):
            c_sub = _coerce_to_slot(cand.get("subject"), "subject", subject_slots)
            c_tgt = _coerce_to_slot(cand.get("target"), "target", target_slots)
            if c_sub is None or c_tgt is None:
                continue
            try:
                c_conf = float(cand.get("confidence", 0.0) or 0.0)
            except Exception:
                c_conf = 0.0
            name = str(cand.get("name", f"extra_{idx}") or f"extra_{idx}")
            candidates.append(
                {
                    "name": name,
                    "subject": c_sub,
                    "target": c_tgt,
                    "source_conf": max(0.40, _clip01(c_conf)),
                }
            )

        scored_candidates: List[Dict[str, Any]] = []
        for c in candidates:
            sc = self._pair_score(
                merged_text=merged,
                subject_slot=c["subject"],
                target_slot=c["target"],
                subject_slots=subject_slots,
                target_slots=target_slots,
                result=result,
                source_conf=float(c.get("source_conf", 0.0)),
            )
            item = dict(c)
            item["score"] = float(sc)
            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: float(x.get("score", -1.0)), reverse=True)
        best = scored_candidates[0] if scored_candidates else None

        if self.strategy == "model_first":
            final_sub = model_sub or result.subject_anchor_slot
            final_tgt = model_tgt or result.target_anchor_slot
            sub_source = "model_first_model" if model_sub is not None else "model_first_anchor_fallback"
            tgt_source = "model_first_model" if model_tgt is not None else "model_first_anchor_fallback"
        elif self.strategy == "anchor_bias":
            final_sub = model_sub
            final_tgt = model_tgt
            sub_source = "model"
            tgt_source = "model"
            if final_sub is None:
                final_sub = result.subject_anchor_slot
                sub_source = "anchor_fallback"
            elif result.subject_confidence >= self.anchor_min_conf and final_sub != result.subject_anchor_slot:
                final_sub = result.subject_anchor_slot
                sub_source = "anchor_override"
            if final_tgt is None:
                final_tgt = result.target_anchor_slot
                tgt_source = "anchor_fallback"
            elif result.target_confidence >= self.anchor_min_conf and final_tgt != result.target_anchor_slot:
                final_tgt = result.target_anchor_slot
                tgt_source = "anchor_override"
        else:
            final_sub = model_sub or result.subject_anchor_slot
            final_tgt = model_tgt or result.target_anchor_slot
            sub_source = "model_fallback"
            tgt_source = "model_fallback"

            if best is not None:
                best_name = str(best.get("name", "best"))
                if best_name == "role_probe" and probe_conf < self.role_probe_min_conf:
                    pass
                else:
                    final_sub = best.get("subject") or final_sub
                    final_tgt = best.get("target") or final_tgt
                    sub_source = best_name
                    tgt_source = best_name

            if final_sub is None:
                final_sub = result.subject_anchor_slot
                sub_source = "anchor_safety"
            if final_tgt is None:
                final_tgt = result.target_anchor_slot
                tgt_source = "anchor_safety"

        detail = {
            "strategy": self.strategy,
            "anchor_min_conf": self.anchor_min_conf,
            "role_probe_min_conf": self.role_probe_min_conf,
            "raw_subject": raw_subject,
            "raw_target": raw_target,
            "model_subject_slot": model_sub,
            "model_target_slot": model_tgt,
            "probe_subject_slot": probe_sub,
            "probe_target_slot": probe_tgt,
            "probe_confidence": probe_conf,
            "anchor_subject_slot": result.subject_anchor_slot,
            "anchor_target_slot": result.target_anchor_slot,
            "subject_confidence": result.subject_confidence,
            "target_confidence": result.target_confidence,
            "candidate_scores": scored_candidates[:8],
            "final_subject_slot": final_sub,
            "final_target_slot": final_tgt,
            "subject_source": sub_source,
            "target_source": tgt_source,
        }
        return final_sub, final_tgt, detail
