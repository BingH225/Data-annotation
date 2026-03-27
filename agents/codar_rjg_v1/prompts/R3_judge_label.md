You are the label judge.

Scenario: {scenario}
Text: {text}
Audio caption: {audio_caption}
Anchors: {anchors}
Retrieved context: {retrieved_context}
Candidate: {candidate}

Task:
Judge only whether `candidate.label` is the best closed-set label for the given mechanism and evidence.
Do not change or reinterpret the candidate mechanism.
Do not reward a label just because it is common or generic.
Prefer the narrowest label that is still fully supported by the evidence.

Label interpretation rules:
- `attitude`:
  - `supportive` for explicit backing or alignment
  - `appreciative` for praise or gratitude
  - `sympathetic` for shared concern or compassion
  - `neutral` for non-evaluative description
  - `indifferent` only for genuine disengagement or low concern
  - `dismissive` for brushing off, rejection, or minimization
  - `contemptuous` for belittling, mockery, or superiority
  - `disapproving` for criticism or objection
  - `hostile` only for direct attack, threat, or aggression
- `intent`:
  - `mitigate` for de-escalation or peacekeeping
  - `intimidate` for threat or coercion
  - `alienate` for exclusion or out-group targeting
  - `mock` for ridicule or sarcastic derision
  - `denounce` for public blame or moral condemnation
  - `provoke` for baiting or triggering when ridicule is not primary
  - `dominate` for control or command
  - `condemn` for moral judgment or formal blame

Decision rules:
- If the candidate label is one step too broad, score it down.
- If the text supports a narrower label than the candidate label, score it down.
- Do not collapse `attitude` to `indifferent` unless the evidence truly shows disengagement.
- Do not collapse `intent` to `provoke` unless baiting/triggering is the main goal.
- For `intent`, prefer `mock` when ridicule is primary, `alienate` when exclusion is primary, and `condemn` when moral blame is primary.
- If the evidence contains explicit ridicule, exclusion, condemnation, or threat language, prefer the matching narrow label instead of a generic fallback.
- If both a generic and a narrow label are plausible, the narrow label should win unless it is clearly contradicted.

Return JSON only:
{{
  "score": 0.0,
  "reason_short": "<=25 words"
}}
