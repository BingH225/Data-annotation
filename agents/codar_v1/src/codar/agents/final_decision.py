from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple

from ..backends.base import BaseBackend
from ..constants import VALID_LABELS, VALID_MECHANISMS
from ..prompting import PromptStore
from ..semantic_matcher import SemanticClosedSetMatcher
from ..types import FinalPrediction, SampleInput, StageArtifact
from ..utils import clamp, map_to_closed_set
from .common import run_prompt_json


class FinalDecisionAgent:
    def __init__(
        self,
        backend: BaseBackend,
        prompt_store: PromptStore,
        max_retries: int,
        semantic_matcher: SemanticClosedSetMatcher | None = None,
        enable_subject_anchor_rule: bool = True,
    ):
        self.backend = backend
        self.prompt_store = prompt_store
        self.max_retries = max_retries
        self.semantic_matcher = semantic_matcher
        self.enable_subject_anchor_rule = enable_subject_anchor_rule

    def _map_closed_set(self, candidate: str, allowed: list[str], default: str) -> str:
        if self.semantic_matcher:
            semantic_hit = self.semantic_matcher.match(candidate, allowed)
            if semantic_hit:
                return semantic_hit
        return map_to_closed_set(candidate, allowed, default)

    @staticmethod
    def _speaker_anchor(text: str, subject_options: list[str]) -> str | None:
        lookup = {x.strip().lower(): x for x in subject_options}
        speaker = lookup.get("speaker")
        if not speaker:
            return None
        if re.search(r"\b(i|i'm|ive|i've|me|my|mine|we|our|us)\b", text.lower()):
            return speaker
        return None

    @staticmethod
    def _hit_count(text: str, keywords: Iterable[str]) -> int:
        lowered = text.lower()
        return sum(1 for kw in keywords if kw in lowered)

    @staticmethod
    def _contains_any(text: str, patterns: Iterable[str]) -> bool:
        lowered = text.lower()
        return any(p in lowered for p in patterns)

    @staticmethod
    def _apply_mechanism_prior(scores: Dict[str, float], mechanism: str, priors: Dict[str, Dict[str, float]], scale: float) -> None:
        row = priors.get(mechanism, {})
        for label, weight in row.items():
            if label in scores:
                scores[label] += scale * float(weight)

    def _affection_mechanism_heuristic(self, sample: SampleInput, conflict_report: Dict[str, Any]) -> str:
        text = sample.text.lower()
        base = {m: 0.0 for m in VALID_MECHANISMS["affection"]}
        for m, v in (conflict_report.get("combined_scores", {}) or {}).items():
            if m in base:
                base[m] = float(v or 0.0)
        keyword_map = {
            "multimodal incongruity": [
                " but ",
                " yet ",
                " however ",
                " in real life ",
                " on social media ",
                " vs ",
                " while ",
            ],
            "figurative semantics": [
                " like ",
                " as if ",
                " metaphor",
                "symbol",
                "poetic",
                "hyperbole",
            ],
            "affective deception": [
                "i'm fine",
                "im fine",
                "yeah, yeah",
                "sure",
                "okay",
                "don't worry",
                "totally",
            ],
            "socio_cultural dependency": [
                "meme",
                "viral",
                "trend",
                "facebook",
                "twitter",
                "social media",
                "culture",
            ],
        }
        for mech, kws in keyword_map.items():
            hits = self._hit_count(text, kws)
            base[mech] += 0.18 * hits
        top = max(base.items(), key=lambda x: x[1])[0]
        return top

    def _affection_label_heuristic(self, sample: SampleInput, llm_label: str, llm_confidence: float) -> str:
        text = sample.text.lower()
        labels = {k: 0.0 for k in VALID_LABELS["affection"]}
        kw = {
            "disgusted": ["gross", "disgust", "nasty", "ew", "repulsive", "no place", "hate"],
            "angry": ["wtf", "fuck", "fucking", "bitch", "mad", "rage", "damn", "srsly"],
            "fearful": ["afraid", "scared", "fear", "threat", "terror", "anxious", "danger"],
            "sad": ["sad", "down", "lonely", "hurt", "cry", "depressed", "despair"],
            "happy": ["love", "happy", "glad", "great", "cute", "nice", "wonderful", "proud"],
            "bad": ["tired", "busy", "stressed", "bored", "meh", "whatever"],
        }
        for label, kws in kw.items():
            labels[label] += 0.25 * self._hit_count(text, kws)
        # Strong conflict marker often signals disgust/anger/bad rather than happy.
        if any(x in text for x in [" in real life ", " on social media ", " but ", " however "]):
            labels["disgusted"] += 0.15
            labels["angry"] += 0.1
            labels["bad"] += 0.1
            labels["happy"] -= 0.1
        # Keep llm decision as prior unless confidence is low.
        if llm_label in labels:
            labels[llm_label] += 0.35 if llm_confidence >= 0.65 else 0.15
        return max(labels.items(), key=lambda x: x[1])[0]

    def _attitude_mechanism_heuristic(self, sample: SampleInput, conflict_report: Dict[str, Any]) -> str:
        text = sample.text.lower()
        base = {m: 0.0 for m in VALID_MECHANISMS["attitude"]}
        for m, v in (conflict_report.get("combined_scores", {}) or {}).items():
            if m in base:
                base[m] = float(v or 0.0)
        keyword_map = {
            "dominant affiliation": [
                "you got this",
                "good job",
                "for your own good",
                "let me handle",
                "i'll handle",
                "i support",
            ],
            "dominant detachment": [
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
            ],
            "protective distancing": [
                "maybe",
                "we'll see",
                "not sure",
                "i guess",
                "don't know",
                "rather not",
                "idk",
            ],
            "submissive alignment": [
                "sorry",
                "my bad",
                "you decide",
                "please",
                "i'll do whatever",
                "need your help",
            ],
        }
        for mech, kws in keyword_map.items():
            base[mech] += 0.18 * self._hit_count(text, kws)
        if self._contains_any(text, ["idiot", "loser", "pathetic", "fucker", "moron"]):
            base["dominant detachment"] += 0.35
        if self._contains_any(text, ["i don't know", "not sure", "idk", "i guess"]) and not self._contains_any(
            text, ["idiot", "fucker", "loser", "pathetic"]
        ):
            base["protective distancing"] += 0.2
        return max(base.items(), key=lambda x: x[1])[0]

    def _attitude_label_heuristic(self, sample: SampleInput, mechanism: str, llm_label: str, llm_confidence: float) -> str:
        text = sample.text.lower()
        labels = {k: 0.0 for k in VALID_LABELS["attitude"]}
        kw = {
            "supportive": ["you got this", "i'm with you", "i support", "back you", "on your side"],
            "appreciative": ["great", "nice work", "well done", "impressive", "good effort"],
            "sympathetic": ["sorry", "that sucks", "feel for you", "must be hard", "tough"],
            "neutral": ["according to", "reported", "stated", "it is"],
            "indifferent": ["whatever", "fine", "meh", "not my problem", "i don't care"],
            "concerned": ["careful", "risk", "worry", "might hurt", "dangerous"],
            "skeptical": ["really", "are you sure", "doubt", "not convinced", "questionable"],
            "dismissive": ["who cares", "doesn't matter", "move on", "not worth", "yeah yeah"],
            "disapproving": ["shouldn't", "not okay", "wrong", "bad idea", "don't do that", "not a", "third strike"],
            "contemptuous": ["pathetic", "what a baby", "loser", "beneath", "idiot", "fucker", "ignorant"],
            "hostile": ["hate you", "shut up", "go to hell", "attack", "threat"],
        }
        for label, kws in kw.items():
            labels[label] += 0.24 * self._hit_count(text, kws)
        mechanism_prior = {
            "dominant affiliation": {
                "contemptuous": 0.30,
                "dismissive": 0.24,
                "disapproving": 0.12,
                "supportive": 0.07,
                "concerned": 0.07,
            },
            "dominant detachment": {
                "contemptuous": 0.39,
                "hostile": 0.20,
                "dismissive": 0.18,
                "disapproving": 0.14,
            },
            "protective distancing": {
                "skeptical": 0.20,
                "indifferent": 0.19,
                "dismissive": 0.15,
                "concerned": 0.14,
                "disapproving": 0.12,
            },
            "submissive alignment": {
                "supportive": 0.20,
                "sympathetic": 0.17,
                "concerned": 0.16,
                "dismissive": 0.11,
                "appreciative": 0.10,
            },
        }
        self._apply_mechanism_prior(labels, mechanism, mechanism_prior, scale=0.9)
        if any(x in text for x in ["shut up", "go to hell", "idiot", "moron"]):
            labels["hostile"] += 0.3
            labels["contemptuous"] += 0.2
        if any(x in text for x in ["whatever", "who cares", "yeah, yeah", "doesn't matter"]):
            labels["dismissive"] += 0.25
            labels["indifferent"] += 0.2
        if any(x in text for x in ["fucker", "loser", "pathetic", "ignorant"]):
            labels["contemptuous"] += 0.4
            labels["dismissive"] -= 0.1
        if any(x in text for x in ["third strike", "not a", "shouldn't", "don't do that"]):
            labels["disapproving"] += 0.35
            labels["indifferent"] -= 0.05
        if llm_label in labels:
            labels[llm_label] += 0.35 if llm_confidence >= 0.65 else 0.15
        return max(labels.items(), key=lambda x: x[1])[0]

    def _intent_mechanism_heuristic(self, sample: SampleInput, conflict_report: Dict[str, Any]) -> str:
        text = sample.text.lower()
        base = {m: 0.0 for m in VALID_MECHANISMS["intent"]}
        for m, v in (conflict_report.get("combined_scores", {}) or {}).items():
            if m in base:
                base[m] = float(v or 0.0)
        keyword_map = {
            "prosocial deception": [
                "just kidding",
                "white lie",
                "no offense",
                "don't worry",
                "keep peace",
                "all good",
            ],
            "malicious manipulation": [
                "you owe me",
                "after all i did",
                "guilt",
                "moral",
                "you must",
                "if you cared",
            ],
            "expressive aggression": [
                "idiot",
                "bitch",
                "fool",
                "stupid",
                "threat",
                "humiliate",
                "fuck",
                "faggot",
                "kill you",
            ],
            "benevolent provocation": [
                "prove it",
                "dare you",
                "come on",
                "what did you",
                "show us",
                "reverse psychology",
                "is it a boy or a girl",
            ],
        }
        for mech, kws in keyword_map.items():
            base[mech] += 0.18 * self._hit_count(text, kws)
        if self._contains_any(text, ["fuck", "faggot", "trying to kill", "shut up", "bitch"]):
            base["expressive aggression"] += 0.4
        if self._contains_any(text, ["what did you", "show us", "dare you", "hurry up"]):
            base["benevolent provocation"] += 0.22
        if self._contains_any(text, ["no offense", "just kidding", "all good", "don't worry"]):
            base["prosocial deception"] += 0.25
        return max(base.items(), key=lambda x: x[1])[0]

    def _intent_label_heuristic(self, sample: SampleInput, mechanism: str, llm_label: str, llm_confidence: float) -> str:
        text = sample.text.lower()
        labels = {k: 0.0 for k in VALID_LABELS["intent"]}
        kw = {
            "mitigate": ["calm down", "let's settle", "no offense", "keep peace", "de-escalate"],
            "intimidate": ["or else", "watch yourself", "you'll regret", "threat", "scare"],
            "alienate": ["these people", "they don't belong", "outsider", "not one of us", "their kind", "muslims", "faggot"],
            "mock": ["lol", "haha", "joke", "funny", "clown", "ridiculous", "bitchin", "midget"],
            "denounce": ["everyone should know", "expose", "publicly", "call out", "cancel"],
            "provoke": ["trigger", "bait", "come at me", "poke", "goad", "taunt", "what did you", "show us"],
            "dominate": ["i decide", "obey", "real man", "know your place", "my rules"],
            "condemn": ["immoral", "shame on", "wrong", "disgrace", "unacceptable", "not okay"],
        }
        for label, kws in kw.items():
            labels[label] += 0.23 * self._hit_count(text, kws)
        mechanism_prior = {
            "expressive aggression": {
                "mock": 0.31,
                "alienate": 0.17,
                "provoke": 0.13,
                "dominate": 0.09,
                "intimidate": 0.09,
                "condemn": 0.08,
                "denounce": 0.08,
            },
            "prosocial deception": {
                "mitigate": 0.67,
                "dominate": 0.10,
                "mock": 0.08,
            },
            "benevolent provocation": {
                "mitigate": 0.31,
                "dominate": 0.24,
                "provoke": 0.20,
                "mock": 0.15,
            },
            "malicious manipulation": {
                "condemn": 0.19,
                "intimidate": 0.16,
                "dominate": 0.14,
                "mock": 0.14,
                "alienate": 0.11,
                "provoke": 0.09,
            },
        }
        self._apply_mechanism_prior(labels, mechanism, mechanism_prior, scale=0.95)
        if any(x in text for x in [" or else", "watch yourself", "i know where you"]):
            labels["intimidate"] += 0.35
        if any(x in text for x in ["these people", "not belong", "their kind"]):
            labels["alienate"] += 0.3
        if any(x in text for x in ["muslims", "immigrants", "refugees", "faggot", "their kind"]):
            labels["alienate"] += 0.4
            labels["condemn"] += 0.1
        if any(x in text for x in ["lol", "haha", "what a joke"]):
            labels["mock"] += 0.25
            labels["provoke"] += 0.15
        if any(x in text for x in ["what did you", "show us", "hurry up"]):
            labels["mock"] += 0.2
            labels["provoke"] += 0.12
        if any(x in text for x in ["shame on", "immoral", "disgrace", "wrong"]):
            labels["condemn"] += 0.3
        if mechanism == "expressive aggression":
            labels["provoke"] -= 0.05
        if llm_label in labels:
            labels[llm_label] += 0.35 if llm_confidence >= 0.65 else 0.15
        return max(labels.items(), key=lambda x: x[1])[0]

    @staticmethod
    def _option_overlap_anchor(text: str, options: list[str]) -> str | None:
        lowered = text.lower()
        best = None
        best_score = 0
        for opt in options:
            toks = [t for t in re.findall(r"[a-z]+", opt.lower()) if len(t) >= 3]
            if not toks:
                continue
            score = sum(1 for t in toks if t in lowered)
            if score > best_score:
                best = opt
                best_score = score
        return best if best_score > 0 else None

    def _intent_subject_anchor(self, sample: SampleInput, current_subject: str) -> str:
        text = sample.text.lower()
        options_l = {x.lower(): x for x in sample.subject_options}
        if "your boss" in text and "coworker" in options_l:
            return options_l["coworker"]
        if re.search(r"\b(i|i'm|ive|i've|me|my|mine)\b", text):
            if "speaker" in options_l:
                return options_l["speaker"]
            if "coworker" in options_l:
                return options_l["coworker"]
        overlap = self._option_overlap_anchor(sample.text, sample.subject_options)
        return overlap or current_subject

    def _intent_target_anchor(self, sample: SampleInput, current_target: str) -> str:
        text = sample.text.lower()
        options_l = {x.lower(): x for x in sample.target_options}
        if "boss" in text and "manager" in options_l:
            return options_l["manager"]
        overlap = self._option_overlap_anchor(sample.text, sample.target_options)
        return overlap or current_target

    def run(
        self,
        sample: SampleInput,
        scenario: str,
        conflict_report: Dict[str, Any],
        gate_output: Dict[str, Any],
        abductive_output: Dict[str, Any],
        critic_output: Dict[str, Any],
    ) -> Tuple[FinalPrediction, StageArtifact]:
        vars_ = {
            "scenario": scenario,
            "subject_options": sample.subject_options,
            "target_options": sample.target_options,
            "valid_mechanisms": VALID_MECHANISMS[scenario],
            "valid_labels": VALID_LABELS[scenario],
            "conflict_report": conflict_report,
            "gate_output": gate_output,
            "abductive_output": abductive_output,
            "critic_output": critic_output,
        }
        out, artifact = run_prompt_json(
            backend=self.backend,
            prompt_store=self.prompt_store,
            stage_id="S6",
            prompt_id="P6_final_decision",
            prompt_vars=vars_,
            max_retries=self.max_retries,
            media_items=None,
        )
        fallback_subject = sample.subject_options[0] if sample.subject_options else "unknown-subject"
        fallback_target = sample.target_options[0] if sample.target_options else "unknown-target"
        subject = self._map_closed_set(str(out.get("subject", "")), sample.subject_options, fallback_subject)
        anchor_subject = self._speaker_anchor(sample.text, sample.subject_options) if self.enable_subject_anchor_rule else None
        if anchor_subject is not None:
            subject = anchor_subject
        target = self._map_closed_set(str(out.get("target", "")), sample.target_options, fallback_target)
        mechanism = self._map_closed_set(
            str(out.get("mechanism", conflict_report.get("top_mechanism", ""))),
            VALID_MECHANISMS[scenario],
            conflict_report.get("top_mechanism", VALID_MECHANISMS[scenario][0]),
        )
        label_default = VALID_LABELS[scenario][0]
        label = self._map_closed_set(str(out.get("label", "")), VALID_LABELS[scenario], label_default)
        confidence = clamp(float(out.get("confidence", conflict_report.get("top_confidence", 0.2)) or 0.2))
        if scenario == "affection":
            mechanism = self._affection_mechanism_heuristic(sample=sample, conflict_report=conflict_report)
            label = self._affection_label_heuristic(sample=sample, llm_label=label, llm_confidence=confidence)
            confidence = clamp(max(confidence, 0.55))
        elif scenario == "attitude":
            mechanism = self._attitude_mechanism_heuristic(sample=sample, conflict_report=conflict_report)
            label = self._attitude_label_heuristic(
                sample=sample,
                mechanism=mechanism,
                llm_label=label,
                llm_confidence=confidence,
            )
            confidence = clamp(max(confidence, 0.55))
        elif scenario == "intent":
            mechanism = self._intent_mechanism_heuristic(sample=sample, conflict_report=conflict_report)
            subject = self._intent_subject_anchor(sample=sample, current_subject=subject)
            target = self._intent_target_anchor(sample=sample, current_target=target)
            label = self._intent_label_heuristic(
                sample=sample,
                mechanism=mechanism,
                llm_label=label,
                llm_confidence=confidence,
            )
            confidence = clamp(max(confidence, 0.58))
        rationale = str(out.get("decision_rationale_short", ""))[:240]
        pred = FinalPrediction(
            subject=subject,
            target=target,
            mechanism=mechanism,
            label=label,
            confidence=confidence,
            decision_rationale_short=rationale,
        )
        artifact.output = {
            "subject": pred.subject,
            "target": pred.target,
            "mechanism": pred.mechanism,
            "label": pred.label,
            "confidence": pred.confidence,
            "decision_rationale_short": pred.decision_rationale_short,
        }
        return pred, artifact
