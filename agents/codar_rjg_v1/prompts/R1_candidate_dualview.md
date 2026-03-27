You are generating two candidate tuples for implicit social reasoning.

Scenario: {scenario}
View: {view}
TemperatureHint: {temperature}

Input text:
{text}

Audio caption:
{audio_caption}

Subject options:
{subject_options}

Target options:
{target_options}

Valid mechanisms:
{valid_mechanisms}

Valid labels:
{valid_labels}

Anchors:
{anchors}

Retrieved context:
{retrieved_context}

Hard rules:
1) Return exactly two candidates under `candidates`.
2) Candidate 1 must be the strongest literal/direct reading.
3) Candidate 2 must be the strongest pragmatic/relational reading.
4) Use only values from `subject_options`, `target_options`, `valid_mechanisms`, and `valid_labels`.
5) Never output null, empty string, empty list, or placeholder text.
6) Do not collapse mechanism and label into one decision. Mechanism is the social reasoning family; label is the closed-set outcome within that family.
7) Prefer the most specific label supported by evidence. Do not default to a broad label when a narrower one is warranted.
8) For `attitude`, distinguish:
   - `supportive` for explicit backing or alignment
   - `appreciative` for praise or gratitude
   - `sympathetic` for shared concern or compassion
   - `neutral` for descriptive or non-evaluative stance
   - `indifferent` only for genuine disengagement or low concern
   - `dismissive` for rejection, brushing off, or minimizing
   - `contemptuous` for belittling, mockery, or superiority
   - `disapproving` for criticism or moral objection
   - `hostile` for direct attack, threat, or aggression
9) For `intent`, distinguish:
   - `mitigate` for de-escalation or peacekeeping
   - `intimidate` for threat or coercion
   - `alienate` for exclusion or out-group targeting
   - `mock` for ridicule or sarcastic derision
   - `denounce` for public blame or moral condemnation
   - `provoke` for baiting or triggering without stronger ridicule
   - `dominate` for control or command
   - `condemn` for moral judgment or formal blame
10) If evidence is mixed, prefer the candidate with stronger lexical and pragmatic support, not the broader fallback.

Return exactly:
{{
  "candidates": [
    {{
      "subject": "string",
      "target": "string",
      "mechanism": "string",
      "label": "string",
      "confidence": 0.0,
      "view": "literal",
      "decision_rationale_short": "<=40 words"
    }},
    {{
      "subject": "string",
      "target": "string",
      "mechanism": "string",
      "label": "string",
      "confidence": 0.0,
      "view": "pragmatic",
      "decision_rationale_short": "<=40 words"
    }}
  ]
}}
