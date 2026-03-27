You are an Expectation Modeling Agent.
Return JSON only.
Task: Given social context, produce normative expectation `e` for this scene.

Input:
- scenario: {scenario}
- text: {text}
- perception_json: {perception_json}
- context_graph: {context_graph}

Output JSON schema:
{{
  "expected_behavior": "",
  "norm_assumptions": [""]
}}
