from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple

from ..constants import VALID_LABELS, VALID_MECHANISMS


@dataclass
class RJGWeights:
    retrieve_support: float = 0.18
    judge_mech: float = 0.30
    judge_label: float = 0.42
    judge_role: float = 0.03
    rule_cue: float = 0.02
    heuristic_agreement: float = 0.22


@dataclass
class RJGConstraintConfig:
    compat_floor: float = 0.10
    compat_hard_floor: float = 0.04
    compat_soft_penalty: float = 0.85
    compat_hard_penalty: float = 0.22
    parser_missing_penalty: float = 0.12
    anchor_missing_penalty: float = 0.14
    anchor_mismatch_penalty: float = 0.16
    role_option_penalty: float = 0.18
    invalid_mechanism_penalty: float = 0.26
    invalid_label_penalty: float = 0.26


@dataclass
class RJGRepairConfig:
    enabled: bool = True
    compat_gate: float = 0.08
    high_confidence_skip: float = 0.68
    min_score_gap: float = 0.12
    min_keyword_hits: int = 1


def default_rjg_weights() -> RJGWeights:
    return RJGWeights()


_COMPAT_PRIOR: Dict[str, Dict[str, Dict[str, float]]] = {
    "affection": {
        "multimodal incongruity": {"disgusted": 0.34, "angry": 0.22, "bad": 0.18, "sad": 0.12, "happy": 0.06, "fearful": 0.08},
        "figurative semantics": {"bad": 0.26, "happy": 0.22, "sad": 0.21, "disgusted": 0.12, "fearful": 0.10, "angry": 0.09},
        "affective deception": {"bad": 0.32, "disgusted": 0.21, "sad": 0.18, "angry": 0.14, "happy": 0.10, "fearful": 0.05},
        "socio_cultural dependency": {"bad": 0.25, "disgusted": 0.24, "angry": 0.19, "sad": 0.14, "happy": 0.12, "fearful": 0.06},
    },
    "attitude": {
        "dominant affiliation": {
            "supportive": 0.18,
            "concerned": 0.16,
            "dismissive": 0.15,
            "disapproving": 0.13,
            "skeptical": 0.10,
            "contemptuous": 0.08,
            "appreciative": 0.08,
            "indifferent": 0.07,
            "neutral": 0.05,
        },
        "dominant detachment": {
            "contemptuous": 0.24,
            "dismissive": 0.20,
            "hostile": 0.18,
            "disapproving": 0.14,
            "indifferent": 0.10,
            "skeptical": 0.06,
            "concerned": 0.05,
            "neutral": 0.04,
        },
        "protective distancing": {
            "skeptical": 0.22,
            "concerned": 0.14,
            "indifferent": 0.15,
            "dismissive": 0.10,
            "disapproving": 0.12,
            "neutral": 0.11,
            "contemptuous": 0.05,
        },
        "submissive alignment": {
            "supportive": 0.20,
            "sympathetic": 0.18,
            "appreciative": 0.15,
            "concerned": 0.13,
            "neutral": 0.10,
            "dismissive": 0.08,
            "disapproving": 0.08,
            "skeptical": 0.06,
        },
    },
    "intent": {
        "prosocial deception": {"mitigate": 0.41, "dominate": 0.16, "mock": 0.13, "provoke": 0.12},
        "malicious manipulation": {"dominate": 0.22, "intimidate": 0.19, "condemn": 0.15, "denounce": 0.12, "provoke": 0.11},
        "expressive aggression": {"mock": 0.24, "alienate": 0.21, "provoke": 0.16, "intimidate": 0.12, "condemn": 0.11},
        "benevolent provocation": {"provoke": 0.24, "mitigate": 0.19, "mock": 0.17, "dominate": 0.14, "denounce": 0.10},
    },
}


def compatibility_prior_score(scenario: str, mechanism: str, label: str) -> float:
    row = (_COMPAT_PRIOR.get(str(scenario).strip().lower(), {}) or {}).get(str(mechanism), {}) or {}
    if not row:
        return 0.05
    return float(row.get(str(label), 0.03))


def _hit_count(text: str, keywords: Iterable[str]) -> int:
    lowered = str(text or "").lower()
    return sum(1 for kw in keywords if str(kw).lower() in lowered)


def _best_scored_item(score_map: Dict[str, float], default: str) -> Tuple[str, float]:
    if not score_map:
        return default, 0.0
    best_item, best_score = max(score_map.items(), key=lambda x: (x[1], x[0]))
    return best_item, float(best_score)


_LABEL_CUES: Dict[str, Dict[str, List[str]]] = {
    "affection": {
        "happy": ["love", "great", "nice", "wonderful", "proud", "cute", "good"],
        "sad": ["sad", "hurt", "cry", "lonely", "down", "depressed"],
        "disgusted": ["gross", "disgust", "nasty", "ew", "repulsive", "no place", "hate"],
        "angry": ["wtf", "fuck", "fucking", "bitch", "mad", "rage", "damn", "srsly"],
        "fearful": ["afraid", "scared", "fear", "threat", "terror", "anxious", "danger"],
        "bad": ["tired", "busy", "stressed", "bored", "meh", "whatever", "fine", "okay"],
    },
    "attitude": {
        "supportive": ["you got this", "i'm with you", "i support", "back you", "on your side"],
        "appreciative": ["thank you", "nice work", "well done", "impressive", "good effort"],
        "sympathetic": ["sorry", "that sucks", "feel for you", "must be hard", "tough"],
        "neutral": ["according to", "reported", "stated", "it is"],
        "indifferent": ["whatever", "fine", "meh", "not my problem", "i don't care", "who cares"],
        "concerned": ["careful", "risk", "worry", "might hurt", "dangerous"],
        "skeptical": ["really", "are you sure", "doubt", "not convinced", "questionable"],
        "dismissive": ["who cares", "doesn't matter", "move on", "not worth", "yeah yeah"],
        "disapproving": ["shouldn't", "not okay", "wrong", "bad idea", "don't do that", "third strike"],
        "contemptuous": ["pathetic", "what a baby", "loser", "beneath", "idiot", "fucker", "ignorant"],
        "hostile": ["hate you", "shut up", "go to hell", "attack", "threat"],
    },
    "intent": {
        "mitigate": ["calm down", "let's settle", "no offense", "keep peace", "de-escalate"],
        "intimidate": ["or else", "watch yourself", "you'll regret", "threat", "scare"],
        "alienate": ["these people", "they don't belong", "outsider", "not one of us", "their kind", "muslims", "faggot"],
        "mock": ["lol", "haha", "joke", "funny", "clown", "ridiculous", "bitchin", "midget"],
        "denounce": ["everyone should know", "expose", "publicly", "call out", "cancel"],
        "provoke": ["trigger", "bait", "come at me", "poke", "goad", "taunt", "what did you", "show us"],
        "dominate": ["i decide", "obey", "real man", "know your place", "my rules"],
        "condemn": ["immoral", "shame on", "wrong", "disgrace", "unacceptable", "not okay"],
    },
}

_INTENT_TOXIC_KEYWORDS: List[str] = [
    "bitch",
    "whore",
    "prostitute",
    "retarded",
    "master race",
    "school shooting",
    "caliphate",
    "kill",
    "destroy",
    "mengele",
    "crack",
    "faggot",
    "muslim",
]

_AFFECTION_MM_MARKERS_STRONG: List[str] = [
    "image",
    "picture",
    "photo",
    "video",
    "caption",
    "screenshot",
    "on social media",
    "in real life",
    "viral",
    "meme",
    "versus",
    " vs ",
]

_AFFECTION_MM_MARKERS_WEAK: List[str] = [
    " but ",
    " yet ",
    " however ",
    " although ",
    " though ",
]

_AFFECTION_FIGURATIVE_MARKERS: List[str] = [
    " like ",
    " as if ",
    "metaphor",
    "hyperbole",
    "poetic",
    "literally",
    "figurative",
]

_AFFECTION_SOCIOCULTURAL_MARKERS: List[str] = [
    "meme",
    "viral",
    "trend",
    "facebook",
    "twitter",
    "social media",
    "culture",
]

_SCENARIO_CONSTRAINT_OVERRIDES: Dict[str, Dict[str, float]] = {
    "affection": {
        "compat_floor": 0.08,
        "compat_hard_floor": 0.03,
        "anchor_missing_penalty": 0.12,
        "anchor_mismatch_penalty": 0.14,
        "role_option_penalty": 0.16,
    },
    "attitude": {
        "compat_floor": 0.11,
        "compat_hard_floor": 0.04,
        "anchor_missing_penalty": 0.16,
        "anchor_mismatch_penalty": 0.20,
        "role_option_penalty": 0.20,
        "invalid_label_penalty": 0.30,
    },
    "intent": {
        "compat_floor": 0.14,
        "compat_hard_floor": 0.06,
        "anchor_missing_penalty": 0.16,
        "anchor_mismatch_penalty": 0.20,
        "role_option_penalty": 0.20,
        "invalid_label_penalty": 0.30,
    },
}


_SCENARIO_REPAIR_OVERRIDES: Dict[str, Dict[str, float]] = {
    "affection": {
        "compat_gate": 0.07,
        "high_confidence_skip": 0.70,
        "min_score_gap": 0.10,
    },
    "attitude": {
        "compat_gate": 0.10,
        "high_confidence_skip": 0.72,
        "min_score_gap": 0.11,
    },
    "intent": {
        "compat_gate": 0.10,
        "high_confidence_skip": 0.72,
        "min_score_gap": 0.11,
    },
}


def resolve_constraint_config(
    scenario: str,
    override: RJGConstraintConfig | Dict[str, float] | None = None,
) -> RJGConstraintConfig:
    payload = asdict(RJGConstraintConfig())
    payload.update(_SCENARIO_CONSTRAINT_OVERRIDES.get(str(scenario or "").strip().lower(), {}))
    if override is not None:
        if isinstance(override, RJGConstraintConfig):
            payload.update(asdict(override))
        elif isinstance(override, dict):
            payload.update({k: float(v) for k, v in override.items() if k in payload})
    return RJGConstraintConfig(**payload)


def resolve_repair_config(
    scenario: str,
    override: RJGRepairConfig | Dict[str, float] | None = None,
) -> RJGRepairConfig:
    payload = asdict(RJGRepairConfig())
    payload.update(_SCENARIO_REPAIR_OVERRIDES.get(str(scenario or "").strip().lower(), {}))
    if override is not None:
        if isinstance(override, RJGRepairConfig):
            payload.update(asdict(override))
        elif isinstance(override, dict):
            payload.update({k: float(v) for k, v in override.items() if k in payload and k != "enabled"})
            if "enabled" in override:
                payload["enabled"] = bool(override["enabled"])
    return RJGRepairConfig(**payload)


def predict_heuristic_mechanism(scenario: str, text: str, scenario_policy: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    sc = str(scenario or "").strip().lower()
    lowered = str(text or "").lower()
    sc_node = ((scenario_policy or {}).get("scenarios", {}) or {}).get(sc, {}) or {}
    cue_map = sc_node.get("rule_keywords", {}) or {}
    scores: Dict[str, float] = {}
    for mech, keywords in cue_map.items():
        hits = _hit_count(lowered, keywords)
        if hits <= 0:
            continue
        scores[str(mech)] = float(hits)

    if sc == "affection":
        mm_strong = _hit_count(lowered, _AFFECTION_MM_MARKERS_STRONG)
        mm_weak = _hit_count(lowered, _AFFECTION_MM_MARKERS_WEAK)
        if mm_strong > 0:
            scores["multimodal incongruity"] = scores.get("multimodal incongruity", 0.0) + 1.8 + 0.3 * min(2, mm_strong - 1)
        elif mm_weak > 0:
            # Contrast words alone are weak evidence and should not dominate mechanism choice.
            scores["multimodal incongruity"] = scores.get("multimodal incongruity", 0.0) + 0.6
        if any(x in lowered for x in _AFFECTION_FIGURATIVE_MARKERS):
            scores["figurative semantics"] = scores.get("figurative semantics", 0.0) + 1.5
        if any(x in lowered for x in ["i'm fine", "im fine", "yeah, yeah", "sure", "okay", "don't worry", "totally"]):
            scores["affective deception"] = scores.get("affective deception", 0.0) + 1.8
        if any(x in lowered for x in _AFFECTION_SOCIOCULTURAL_MARKERS):
            scores["socio_cultural dependency"] = scores.get("socio_cultural dependency", 0.0) + 1.6
    elif sc == "attitude":
        if any(
            x in lowered
            for x in [
                "you got this",
                "good job",
                "for your own good",
                "let me handle",
                "i'll handle",
                "i support",
                "on your side",
            ]
        ):
            scores["dominant affiliation"] = scores.get("dominant affiliation", 0.0) + 1.8
        if any(
            x in lowered
            for x in [
                "whatever",
                "not worth",
                "your fault",
                "asked for it",
                "doesn't matter",
                "don't care",
                "no thanks",
                "third strike",
                "fucker",
                "ignorant",
                "pathetic",
                "loser",
            ]
        ):
            scores["dominant detachment"] = scores.get("dominant detachment", 0.0) + 2.0
        if any(
            x in lowered
            for x in [
                "maybe",
                "we'll see",
                "not sure",
                "i guess",
                "don't know",
                "rather not",
                "idk",
                "careful",
                "risk",
                "worry",
                "dangerous",
                "not okay",
                "shouldn't",
                "bad idea",
            ]
        ):
            scores["protective distancing"] = scores.get("protective distancing", 0.0) + 1.8
        if any(
            x in lowered
            for x in [
                "sorry",
                "my bad",
                "you decide",
                "please",
                "need your help",
                "whatever you want",
                "as you wish",
                "you're right",
                "i'll do it",
                "i understand",
            ]
        ):
            scores["submissive alignment"] = scores.get("submissive alignment", 0.0) + 1.7
    elif sc == "intent":
        if any(x in lowered for x in ["just kidding", "white lie", "no offense", "don't worry", "keep peace", "all good"]):
            scores["prosocial deception"] = scores.get("prosocial deception", 0.0) + 1.6
        if any(x in lowered for x in ["you owe me", "after all i did", "guilt", "moral", "you must", "if you cared"]):
            scores["malicious manipulation"] = scores.get("malicious manipulation", 0.0) + 1.5
        if any(x in lowered for x in ["idiot", "bitch", "fool", "stupid", "threat", "humiliate", "fuck", "faggot", "kill you", "whore", "prostitute", "retarded", "master race", "school shooting", "mengele", "caliphate"]):
            scores["expressive aggression"] = scores.get("expressive aggression", 0.0) + 2.1
        if any(x in lowered for x in ["prove it", "dare you", "come on", "what did you", "show us", "reverse psychology", "hurry up", "is it a boy or a girl"]):
            scores["benevolent provocation"] = scores.get("benevolent provocation", 0.0) + 1.7
        if any(x in lowered for x in ["911", "two can play that game", "you started this", "your fault", "pay for this"]):
            scores["malicious manipulation"] = scores.get("malicious manipulation", 0.0) + 1.3

    if not scores:
        allowed = ((scenario_policy or {}).get("scenarios", {}) or {}).get(sc, {}).get("mechanisms", []) or []
        default = str(allowed[0]) if allowed else ""
        return default, {}
    best, _ = _best_scored_item(scores, "")
    return best, scores


def predict_heuristic_label(scenario: str, mechanism: str, text: str) -> Tuple[str, Dict[str, float]]:
    sc = str(scenario or "").strip().lower()
    mech = str(mechanism or "")
    lowered = str(text or "").lower()
    cue_map = _LABEL_CUES.get(sc, {})
    scores: Dict[str, float] = {}
    for label, keywords in cue_map.items():
        hits = _hit_count(lowered, keywords)
        if hits <= 0:
            continue
        scores[str(label)] = float(hits)

    if mech and sc in _COMPAT_PRIOR:
        row = _COMPAT_PRIOR[sc].get(mech, {})
        for label, prior in row.items():
            scores[label] = scores.get(label, 0.0) + 1.25 * float(prior)

    if sc == "attitude":
        if any(x in lowered for x in ["whatever", "who cares", "yeah yeah", "doesn't matter"]):
            scores["dismissive"] = scores.get("dismissive", 0.0) + 1.8
            scores["indifferent"] = scores.get("indifferent", 0.0) + 1.2
        if any(x in lowered for x in ["pathetic", "what a baby", "loser", "beneath", "idiot", "fucker", "ignorant"]):
            scores["contemptuous"] = scores.get("contemptuous", 0.0) + 1.9
        if any(
            x in lowered
            for x in [
                "shouldn't",
                "not okay",
                "wrong",
                "bad idea",
                "don't do that",
                "third strike",
                "not acceptable",
                "out of line",
                "unprofessional",
                "inappropriate",
            ]
        ):
            scores["disapproving"] = scores.get("disapproving", 0.0) + 2.0
        if any(
            x in lowered
            for x in [
                "shut up",
                "go to hell",
                "attack",
                "threat",
                "hate you",
                "screw you",
                "drop dead",
                "stfu",
            ]
        ):
            scores["hostile"] = scores.get("hostile", 0.0) + 2.2
        # Boundary correction: when explicit hostile/disapproving cues exist, avoid defaulting to contemptuous.
        boundary_hits = 0
        for kw in ["shouldn't", "not okay", "wrong", "bad idea", "shut up", "go to hell", "screw you", "stfu"]:
            if kw in lowered:
                boundary_hits += 1
        if boundary_hits >= 1:
            scores["contemptuous"] = max(0.0, scores.get("contemptuous", 0.0) - 0.8)
    elif sc == "intent":
        if any(x in lowered for x in ["or else", "watch yourself", "you'll regret", "threat", "scare"]):
            scores["intimidate"] = scores.get("intimidate", 0.0) + 1.8
        if any(x in lowered for x in ["these people", "they don't belong", "their kind", "muslims", "faggot", "two genders", "not asian", "master race", "western civilization"]):
            scores["alienate"] = scores.get("alienate", 0.0) + 1.8
        if any(x in lowered for x in ["lol", "haha", "joke", "funny", "clown", "ridiculous", "bitchin", "crack whore", "prostitute", "retarded", "mengele"]):
            scores["mock"] = scores.get("mock", 0.0) + 1.8
        if any(x in lowered for x in ["immoral", "shame on", "wrong", "disgrace", "unacceptable", "not okay", "what they call shame", "wasnt theirs", "bans abortions"]):
            scores["condemn"] = scores.get("condemn", 0.0) + 1.8
        if any(x in lowered for x in ["prove it", "dare you", "come on", "what did you", "show us", "reverse psychology"]):
            scores["provoke"] = scores.get("provoke", 0.0) + 1.5
        if any(x in lowered for x in ["master race", "know your place"]):
            scores["dominate"] = scores.get("dominate", 0.0) + 1.7
    else:
        if any(x in lowered for x in ["wtf", "fuck", "fucking", "bitch", "mad", "rage", "damn", "srsly"]):
            scores["angry"] = scores.get("angry", 0.0) + 2.0
        if any(x in lowered for x in ["gross", "disgust", "nasty", "ew", "repulsive", "no place", "hate"]):
            scores["disgusted"] = scores.get("disgusted", 0.0) + 1.8
        if any(x in lowered for x in ["sad", "hurt", "cry", "lonely", "down", "depressed"]):
            scores["sad"] = scores.get("sad", 0.0) + 1.6
        if any(x in lowered for x in ["afraid", "scared", "fear", "threat", "terror", "anxious", "danger"]):
            scores["fearful"] = scores.get("fearful", 0.0) + 1.6
        if any(x in lowered for x in ["love", "great", "nice", "wonderful", "proud", "cute", "good"]):
            scores["happy"] = scores.get("happy", 0.0) + 1.4
        if any(x in lowered for x in ["tired", "busy", "stressed", "bored", "meh", "whatever", "fine", "okay"]):
            scores["bad"] = scores.get("bad", 0.0) + 1.3

    if not scores:
        allowed = VALID_LABELS.get(sc, []) or []
        default = str(allowed[0]) if allowed else ""
        return default, {}
    best, _ = _best_scored_item(scores, "")
    return best, scores


def _label_keyword_hits(scenario: str, text: str) -> int:
    sc = str(scenario or "").strip().lower()
    lowered = str(text or "").lower()
    cue_map = _LABEL_CUES.get(sc, {}) or {}
    return sum(_hit_count(lowered, keywords) for keywords in cue_map.values())


def _label_repair_candidates(scenario: str, mechanism: str, text: str) -> Dict[str, float]:
    sc = str(scenario or "").strip().lower()
    valid_labels = list(VALID_LABELS.get(sc, []) or [])
    _, score_map = predict_heuristic_label(scenario, mechanism, text)
    if not valid_labels:
        return dict(score_map)
    out: Dict[str, float] = {}
    for label in valid_labels:
        out[label] = float(score_map.get(label, 0.0))
    return out


def repair_inconsistent_label(
    scenario: str,
    mechanism: str,
    label: str,
    text: str,
    repair_config: RJGRepairConfig | Dict[str, float] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    sc = str(scenario or "").strip().lower()
    current_label = str(label or "").strip()
    text = str(text or "")
    mechanism = str(mechanism or "").strip()
    valid_labels = list(VALID_LABELS.get(sc, []) or [])
    cfg = resolve_repair_config(sc, repair_config)

    info: Dict[str, Any] = {
        "scenario": sc,
        "mechanism": mechanism,
        "original_label": current_label,
        "repaired": False,
        "reason": "disabled",
        "compatibility_prior": compatibility_prior_score(sc, mechanism, current_label),
        "current_score": 0.0,
        "best_score": 0.0,
        "score_gap": 0.0,
        "keyword_hits": _label_keyword_hits(sc, text),
    }

    if not cfg.enabled:
        return current_label, info
    if not valid_labels:
        info["reason"] = "invalid_scenario"
        return current_label, info
    if not text.strip():
        info["reason"] = "empty_text_no_repair"
        return current_label, info
    if info["keyword_hits"] < int(cfg.min_keyword_hits):
        info["reason"] = "insufficient_text_signal"
        return current_label, info

    candidate_scores = _label_repair_candidates(sc, mechanism, text)
    if not candidate_scores:
        info["reason"] = "no_candidates"
        return current_label, info

    best_label, best_score = _best_scored_item(candidate_scores, valid_labels[0])
    current_score = float(candidate_scores.get(current_label, 0.0)) if current_label in valid_labels else -1.0
    current_compat = compatibility_prior_score(sc, mechanism, current_label)
    score_gap = float(best_score - current_score)
    info.update(
        {
            "compatibility_prior": current_compat,
            "current_score": current_score,
            "best_score": float(best_score),
            "score_gap": score_gap,
            "best_label": best_label,
        }
    )

    # Scenario-specific anti-collapse guard for generic fallbacks.
    if sc == "attitude" and current_label == "indifferent":
        specific_order = ["hostile", "contemptuous", "disapproving", "dismissive", "skeptical", "concerned", "supportive"]
        specific_scores = [(lb, float(candidate_scores.get(lb, 0.0))) for lb in specific_order if lb in candidate_scores]
        if specific_scores:
            best_specific_label, best_specific_score = max(specific_scores, key=lambda x: (x[1], x[0]))
            if best_specific_label != current_label and (best_specific_score - current_score) >= cfg.min_score_gap:
                info["repaired"] = True
                info["reason"] = "attitude_generic_collapse_repair"
                info["best_label"] = best_specific_label
                info["best_score"] = best_specific_score
                info["score_gap"] = float(best_specific_score - current_score)
                return best_specific_label, info

    if sc == "affection" and current_label == "disgusted":
        disgust_hits = _hit_count(text, _LABEL_CUES.get(sc, {}).get("disgusted", []))
        specific_order = ["angry", "bad", "sad", "happy", "fearful"]
        specific_scores = [(lb, float(candidate_scores.get(lb, 0.0))) for lb in specific_order if lb in candidate_scores]
        specific_hits = sum(_hit_count(text, _LABEL_CUES.get(sc, {}).get(lb, [])) for lb in specific_order)
        if specific_scores and specific_hits > disgust_hits:
            best_specific_label, best_specific_score = max(specific_scores, key=lambda x: (x[1], x[0]))
            if best_specific_label != current_label and (best_specific_score - current_score) >= cfg.min_score_gap:
                info["repaired"] = True
                info["reason"] = "affection_disgust_collapse_repair"
                info["best_label"] = best_specific_label
                info["best_score"] = best_specific_score
                info["score_gap"] = float(best_specific_score - current_score)
                return best_specific_label, info

    if sc == "intent" and current_label == "provoke":
        specific_order = ["mock", "alienate", "condemn", "intimidate", "dominate", "denounce", "mitigate"]
        specific_scores = [(lb, float(candidate_scores.get(lb, 0.0))) for lb in specific_order if lb in candidate_scores]
        if specific_scores:
            best_specific_label, best_specific_score = max(specific_scores, key=lambda x: (x[1], x[0]))
            if best_specific_label != current_label and (best_specific_score - current_score) >= cfg.min_score_gap:
                info["repaired"] = True
                info["reason"] = "intent_generic_collapse_repair"
                info["best_label"] = best_specific_label
                info["best_score"] = best_specific_score
                info["score_gap"] = float(best_specific_score - current_score)
                return best_specific_label, info

    if sc == "intent" and current_label == "mitigate":
        specific_order = ["mock", "alienate", "condemn", "intimidate", "dominate", "denounce", "provoke"]
        specific_scores = [(lb, float(candidate_scores.get(lb, 0.0))) for lb in specific_order if lb in candidate_scores]
        if specific_scores:
            best_specific_label, best_specific_score = max(specific_scores, key=lambda x: (x[1], x[0]))
            if best_specific_label != current_label and (best_specific_score - current_score) >= cfg.min_score_gap:
                info["repaired"] = True
                info["reason"] = "intent_mitigate_collapse_repair"
                info["best_label"] = best_specific_label
                info["best_score"] = best_specific_score
                info["score_gap"] = float(best_specific_score - current_score)
                return best_specific_label, info

    if current_label in valid_labels and current_compat >= cfg.compat_gate:
        info["reason"] = "compatibility_ok"
        return current_label, info
    if current_score >= cfg.high_confidence_skip:
        info["reason"] = "high_confidence_skip"
        return current_label, info
    if best_label == current_label:
        info["reason"] = "already_best"
        return current_label, info
    if score_gap < cfg.min_score_gap:
        info["reason"] = "insufficient_margin"
        return current_label, info

    info["repaired"] = True
    info["reason"] = "compatibility_gate_repair"
    return best_label, info


def heuristic_agreement_score(
    scenario: str,
    mechanism: str,
    label: str,
    text: str,
    scenario_policy: Dict[str, Any],
    anchors: Dict[str, Any] | None = None,
) -> Tuple[float, Dict[str, Any]]:
    pred_mech, mech_scores = predict_heuristic_mechanism(scenario, text, scenario_policy)
    pred_label, label_scores = predict_heuristic_label(scenario, mechanism or pred_mech, text)
    compat = compatibility_prior_score(scenario, mechanism, label)

    mech_score = 0.0
    if pred_mech:
        mech_score = 1.0 if str(pred_mech).strip().lower() == str(mechanism).strip().lower() else 0.0
    label_score = 0.0
    if pred_label:
        label_score = 1.0 if str(pred_label).strip().lower() == str(label).strip().lower() else 0.0

    anchor_bonus = 0.0
    if anchors:
        if anchors.get("subject_anchor") and str(anchors.get("subject_anchor")).strip().lower() in str(text).lower():
            anchor_bonus += 0.1
        if anchors.get("target_anchor") and str(anchors.get("target_anchor")).strip().lower() in str(text).lower():
            anchor_bonus += 0.1

    raw = 0.45 * mech_score + 0.35 * label_score + 0.15 * compat + 0.05 * anchor_bonus
    detail = {
        "predicted_mechanism": pred_mech,
        "predicted_label": pred_label,
        "mechanism_score_map": mech_scores,
        "label_score_map": label_scores,
        "compatibility_prior": compat,
        "anchor_bonus": anchor_bonus,
    }
    return float(max(0.0, min(1.0, raw))), detail


def rule_cue_score(scenario: str, mechanism: str, text: str, scenario_policy: Dict[str, Any]) -> float:
    sc = str(scenario or "").strip().lower()
    mech = str(mechanism or "")
    lowered = str(text or "").lower()
    sc_node = ((scenario_policy or {}).get("scenarios", {}) or {}).get(sc, {}) or {}
    cue_map = sc_node.get("rule_keywords", {}) or {}
    keywords = cue_map.get(mech, []) or []
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if str(kw).lower() in lowered)
    if hits <= 0:
        return 0.0
    return min(1.0, 0.2 * float(hits))


def score_penalty_components(
    scenario: str,
    mechanism: str,
    label: str,
    subject: str,
    target: str,
    subject_options: Iterable[str],
    target_options: Iterable[str],
    parser_non_empty: bool,
    text: str = "",
    anchors: Dict[str, Any] | None = None,
    constraint_config: RJGConstraintConfig | Dict[str, float] | None = None,
) -> Tuple[float, Dict[str, float]]:
    cfg = resolve_constraint_config(scenario, constraint_config)
    sc = str(scenario or "").strip().lower()
    anchors = anchors or {}
    subject_anchor = str(anchors.get("subject_anchor", "") or "").strip()
    target_anchor = str(anchors.get("target_anchor", "") or "").strip()
    valid_mechanisms = VALID_MECHANISMS.get(sc, []) or []
    valid_labels = VALID_LABELS.get(sc, []) or []
    compat = compatibility_prior_score(sc, mechanism, label)

    components: Dict[str, float] = {}
    if compat < cfg.compat_floor:
        components["compatibility_floor"] = cfg.compat_soft_penalty * (cfg.compat_floor - compat)
    if compat < cfg.compat_hard_floor:
        components["compatibility_hard"] = cfg.compat_hard_penalty
    if not parser_non_empty:
        components["parser_missing"] = cfg.parser_missing_penalty
    if not subject_anchor:
        components["subject_anchor_missing"] = cfg.anchor_missing_penalty
    if not target_anchor:
        components["target_anchor_missing"] = cfg.anchor_missing_penalty
    if not valid_mechanisms or str(mechanism).strip() not in valid_mechanisms:
        components["invalid_mechanism"] = cfg.invalid_mechanism_penalty
    if not valid_labels or str(label).strip() not in valid_labels:
        components["invalid_label"] = cfg.invalid_label_penalty
    if not _in_options(subject, subject_options):
        components["subject_role_mismatch"] = cfg.role_option_penalty
    if not _in_options(target, target_options):
        components["target_role_mismatch"] = cfg.role_option_penalty
    if subject_anchor and str(subject).strip().lower() != subject_anchor.lower():
        components["subject_anchor_mismatch"] = cfg.anchor_mismatch_penalty
    if target_anchor and str(target).strip().lower() != target_anchor.lower():
        components["target_anchor_mismatch"] = cfg.anchor_mismatch_penalty

    # Anti-collapse penalties: discourage over-positive defaults when text cues are clearly opposite.
    if sc == "attitude":
        negative_hits = 0
        for bucket in ["dismissive", "contemptuous", "disapproving", "hostile", "skeptical", "indifferent"]:
            negative_hits += _hit_count(text, _LABEL_CUES.get(sc, {}).get(bucket, []))
        positive_hits = 0
        for bucket in ["supportive", "appreciative", "sympathetic", "concerned"]:
            positive_hits += _hit_count(text, _LABEL_CUES.get(sc, {}).get(bucket, []))
        contempt_hits = _hit_count(text, _LABEL_CUES.get(sc, {}).get("contemptuous", []))
        dismissive_hits = _hit_count(text, _LABEL_CUES.get(sc, {}).get("dismissive", []))
        disapproving_hits = _hit_count(text, _LABEL_CUES.get(sc, {}).get("disapproving", []))
        hostile_hits = _hit_count(text, _LABEL_CUES.get(sc, {}).get("hostile", []))
        boundary_negative_hits = max(dismissive_hits, disapproving_hits, hostile_hits)
        if str(label).strip().lower() == "supportive" and negative_hits > 0:
            components["attitude_supportive_conflict"] = 0.22 + 0.03 * min(3, negative_hits - 1)
        if str(label).strip().lower() == "supportive" and positive_hits <= 0:
            components["attitude_supportive_without_positive_signal"] = 0.16
        if str(mechanism).strip().lower() == "dominant affiliation" and negative_hits >= 2:
            components["attitude_affiliation_conflict"] = 0.16
        if str(label).strip().lower() == "indifferent" and negative_hits > 0:
            components["attitude_indifferent_conflict"] = 0.20 + 0.02 * min(3, negative_hits - 1)
        if str(label).strip().lower() == "contemptuous" and boundary_negative_hits >= max(1, contempt_hits):
            components["attitude_contempt_boundary_conflict"] = 0.14 + 0.02 * min(
                3,
                boundary_negative_hits - max(1, contempt_hits),
            )
    elif sc == "affection":
        disgust_hits = _hit_count(text, _LABEL_CUES.get(sc, {}).get("disgusted", []))
        specific_hits = 0
        for bucket in ["angry", "bad", "sad", "happy", "fearful"]:
            specific_hits += _hit_count(text, _LABEL_CUES.get(sc, {}).get(bucket, []))
        if str(label).strip().lower() == "disgusted" and specific_hits > disgust_hits:
            components["affection_disgust_conflict"] = 0.20 + 0.02 * min(3, specific_hits - disgust_hits - 1)

        lowered = str(text or "").lower()
        mm_signal = _hit_count(lowered, _AFFECTION_MM_MARKERS_STRONG) + _hit_count(lowered, _AFFECTION_MM_MARKERS_WEAK)
        figurative_signal = _hit_count(lowered, _AFFECTION_FIGURATIVE_MARKERS)
        socio_signal = _hit_count(lowered, _AFFECTION_SOCIOCULTURAL_MARKERS)
        mech_l = str(mechanism).strip().lower()
        if mech_l == "multimodal incongruity" and mm_signal <= 0:
            components["affection_mm_without_signal"] = 0.18
        if mech_l == "multimodal incongruity" and figurative_signal >= 1:
            components["affection_mm_vs_figurative_conflict"] = 0.12 + 0.02 * min(2, figurative_signal - 1)
        if mech_l != "figurative semantics" and figurative_signal >= 2:
            components["affection_missing_figurative"] = 0.10
        if mech_l != "socio_cultural dependency" and socio_signal >= 2:
            components["affection_missing_sociocultural"] = 0.10
    elif sc == "intent":
        aggressive_hits = 0
        for bucket in ["mock", "alienate", "intimidate", "condemn", "dominate", "provoke", "denounce"]:
            aggressive_hits += _hit_count(text, _LABEL_CUES.get(sc, {}).get(bucket, []))
        specific_intent_hits = 0
        for bucket in ["mock", "alienate", "intimidate", "condemn", "dominate", "denounce"]:
            specific_intent_hits += _hit_count(text, _LABEL_CUES.get(sc, {}).get(bucket, []))
        toxic_hits = _hit_count(text, _INTENT_TOXIC_KEYWORDS)
        if str(label).strip().lower() == "mitigate" and aggressive_hits > 0:
            components["intent_mitigate_conflict"] = 0.24 + 0.03 * min(3, aggressive_hits - 1)
        if str(label).strip().lower() == "mitigate" and toxic_hits > 0:
            components["intent_mitigate_toxic_conflict"] = 0.34 + 0.03 * min(3, toxic_hits - 1)
        if str(mechanism).strip().lower() == "prosocial deception" and aggressive_hits >= 2:
            components["intent_prosocial_conflict"] = 0.16
        if str(mechanism).strip().lower() == "prosocial deception" and toxic_hits > 0:
            components["intent_prosocial_toxic_conflict"] = 0.24 + 0.02 * min(3, toxic_hits - 1)
        if str(label).strip().lower() == "provoke" and specific_intent_hits > 0:
            components["intent_provoke_conflict"] = 0.20 + 0.02 * min(3, specific_intent_hits - 1)

    total = sum(components.values())
    return float(total), components


def _in_options(value: str, options: Iterable[str]) -> bool:
    val = str(value or "").strip().lower()
    return any(val == str(x or "").strip().lower() for x in options)


def compute_penalty(
    scenario: str,
    mechanism: str,
    label: str,
    subject: str,
    target: str,
    subject_options: Iterable[str],
    target_options: Iterable[str],
    parser_non_empty: bool,
    text: str = "",
    anchors: Dict[str, Any] | None = None,
    constraint_config: RJGConstraintConfig | Dict[str, float] | None = None,
) -> float:
    penalty, _ = score_penalty_components(
        scenario=scenario,
        mechanism=mechanism,
        label=label,
        subject=subject,
        target=target,
        subject_options=subject_options,
        target_options=target_options,
        parser_non_empty=parser_non_empty,
        text=text,
        anchors=anchors,
        constraint_config=constraint_config,
    )
    return float(penalty)


def compute_total_score(
    weights: RJGWeights,
    retrieve_support: float,
    judge_mech: float,
    judge_label: float,
    judge_role: float,
    rule_cue: float,
    penalty: float,
    heuristic_agreement: float = 0.0,
) -> float:
    return (
        float(weights.retrieve_support) * float(retrieve_support)
        + float(weights.judge_mech) * float(judge_mech)
        + float(weights.judge_label) * float(judge_label)
        + float(weights.judge_role) * float(judge_role)
        + float(weights.rule_cue) * float(rule_cue)
        + float(weights.heuristic_agreement) * float(heuristic_agreement)
        - float(penalty)
    )


def weights_to_dict(weights: RJGWeights) -> Dict[str, float]:
    return {k: float(v) for k, v in asdict(weights).items()}
