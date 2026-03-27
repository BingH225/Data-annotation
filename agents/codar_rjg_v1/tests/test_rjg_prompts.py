from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from codar.prompting import PromptStore


class TestRJGPromptContracts(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.prompt_store = PromptStore(root / "prompts")
        self.config_path = root / "config" / "runtime.internvl.local.yaml"

    def _base_vars(self):
        return {
            "scenario": "attitude",
            "view": "literal",
            "temperature": 0.0,
            "text": "Whatever, do it yourself.",
            "audio_caption": "speaker sounds dismissive",
            "subject_options": ["speaker", "manager"],
            "target_options": ["listener", "coworker"],
            "valid_mechanisms": ["dominant detachment", "protective distancing"],
            "valid_labels": ["dismissive", "indifferent", "contemptuous"],
            "anchors": {"subject_anchor": "speaker", "target_anchor": "listener"},
            "retrieved_context": [{"sample_id": "x1", "text": "similar sample"}],
            "candidate": {
                "subject": "speaker",
                "target": "listener",
                "mechanism": "dominant detachment",
                "label": "dismissive",
            },
            "candidate_a": {
                "subject": "speaker",
                "target": "listener",
                "mechanism": "dominant detachment",
                "label": "dismissive",
            },
            "candidate_b": {
                "subject": "speaker",
                "target": "listener",
                "mechanism": "protective distancing",
                "label": "indifferent",
            },
        }

    def test_prompt_templates_render(self):
        cases = [
            "R1_candidate_dualview",
            "R2_judge_mech",
            "R3_judge_label",
            "R4_judge_role",
            "R5_tiebreak",
        ]
        expected_snippets = {
            "R3_judge_label": ["narrow label should win", "explicit ridicule, exclusion, condemnation, or threat language"],
            "R5_tiebreak": ["attitude`: `hostile` > `contemptuous` > `disapproving`", "intent`: `mock` > `alienate` > `condemn`"],
        }
        for prompt_id in cases:
            with self.subTest(prompt_id=prompt_id):
                rendered = self.prompt_store.render(prompt_id, self._base_vars()).text
                self.assertNotIn("{scenario}", rendered)
                self.assertNotIn("{candidate}", rendered)
                self.assertNotIn("{candidate_a}", rendered)
                self.assertNotIn("{candidate_b}", rendered)
                self.assertIn("Return", rendered)
                for snippet in expected_snippets.get(prompt_id, []):
                    self.assertIn(snippet, rendered)

    def test_rjg_weights_prefer_label_and_mechanism(self):
        with self.config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        weights = cfg["rjg"]["weights"]
        total = sum(float(v) for v in weights.values())
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertGreater(float(weights["judge_label"]), float(weights["retrieve_support"]))
        self.assertGreater(float(weights["judge_mech"]), float(weights["retrieve_support"]))
        self.assertGreaterEqual(float(weights["judge_label"]), float(weights["judge_mech"]))


if __name__ == "__main__":
    unittest.main()
