You are a tie-break judge between two candidate tuples.

Scenario: {scenario}
Text: {text}
Audio caption: {audio_caption}
Candidate A: {candidate_a}
Candidate B: {candidate_b}

Task:
Choose the better candidate using evidence-grounded mechanism and label specificity.
Prefer the candidate with the sharper mechanism-label match.
Prefer the candidate with the narrower label when the evidence supports it.
Break ties against generic labels such as `indifferent` and `provoke` when a more specific label is supported.
For `attitude`, prefer `dismissive`, `contemptuous`, `disapproving`, or `hostile` over broader fallback labels when the text is clearly evaluative.
For `intent`, prefer `mock`, `alienate`, `condemn`, `intimidate`, or `dominate` over `provoke` when the goal is more specific.
If one candidate is generic and the other is narrow but still evidence-consistent, choose the narrow candidate.
When both candidates remain plausible, use this precedence:
- `attitude`: `hostile` > `contemptuous` > `disapproving` > `dismissive` > `indifferent`
- `intent`: `mock` > `alienate` > `condemn` > `intimidate` > `dominate` > `provoke`

Return JSON only:
{{
  "winner": "A",
  "confidence": 0.0,
  "reason_short": "<=20 words"
}}
