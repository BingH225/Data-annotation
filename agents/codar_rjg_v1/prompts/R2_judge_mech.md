You are the mechanism judge.

Scenario: {scenario}
Text: {text}
Audio caption: {audio_caption}
Anchors: {anchors}
Retrieved context: {retrieved_context}
Candidate: {candidate}

Task:
Judge only whether `candidate.mechanism` is well-supported by the evidence.
Do not use `candidate.label` as a shortcut for mechanism selection.
Do not reward a mechanism because the label looks plausible.
Prefer direct grounding in wording, tone, relation, and communicative goal.

Mechanism interpretation rules:
- For `attitude`, the mechanism is the speaker's relational stance: affiliation, detachment, distancing, or alignment.
- For `intent`, the mechanism is the speaker's goal or strategy: deception, manipulation, aggression, provocation, or control.
- A single keyword is not enough if the rest of the context points elsewhere.
- Score low when the candidate mechanism is generic, over-broad, or only weakly implied.

Scoring guidance:
- 0.90-1.00: mechanism is explicitly and uniquely supported.
- 0.60-0.89: mechanism is plausible and supported by multiple cues.
- 0.30-0.59: mechanism is only partially supported.
- 0.00-0.29: mechanism is unsupported, generic, or confused with label.

Return JSON only:
{{
  "score": 0.0,
  "reason_short": "<=25 words"
}}
