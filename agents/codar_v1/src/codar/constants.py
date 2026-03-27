from __future__ import annotations

SCENARIOS = ["affection", "attitude", "intent"]

VALID_MECHANISMS = {
    "affection": [
        "multimodal incongruity",
        "figurative semantics",
        "affective deception",
        "socio_cultural dependency",
    ],
    "intent": [
        "prosocial deception",
        "malicious manipulation",
        "expressive aggression",
        "benevolent provocation",
    ],
    "attitude": [
        "dominant affiliation",
        "dominant detachment",
        "protective distancing",
        "submissive alignment",
    ],
}

VALID_LABELS = {
    "affection": ["happy", "sad", "disgusted", "angry", "fearful", "bad"],
    "attitude": [
        "supportive",
        "appreciative",
        "sympathetic",
        "neutral",
        "indifferent",
        "concerned",
        "skeptical",
        "dismissive",
        "disapproving",
        "contemptuous",
        "hostile",
    ],
    "intent": [
        "mitigate",
        "intimidate",
        "alienate",
        "mock",
        "denounce",
        "provoke",
        "dominate",
        "condemn",
    ],
}

HYPOTHESES = {
    "H0": "no meaningful conflict",
    "H1": "perception error / missing context",
    "H2": "accidental mismatch / noise",
    "H3": "strategic social expression",
    "H4": "culture-specific or in-group code",
    "H5": "sarcasm / irony / indirect attack / face-saving",
}

