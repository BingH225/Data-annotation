from __future__ import annotations

import unittest

from codar.rjg.fusion import (
    RJGConstraintConfig,
    RJGWeights,
    compute_penalty,
    compute_total_score,
    repair_inconsistent_label,
    score_penalty_components,
)


class TestRJGFusion(unittest.TestCase):
    def test_penalty_for_option_miss(self):
        p = compute_penalty(
            scenario="intent",
            mechanism="expressive aggression",
            label="mock",
            subject="unknown",
            target="unknown",
            subject_options=["speaker", "manager"],
            target_options=["listener", "muslims"],
            parser_non_empty=True,
            anchors={"subject_anchor": "speaker", "target_anchor": "listener"},
        )
        self.assertGreaterEqual(p, 0.30)

    def test_total_score_monotonic(self):
        w = RJGWeights()
        s1 = compute_total_score(w, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1)
        s2 = compute_total_score(w, 0.4, 0.2, 0.2, 0.2, 0.2, 0.1)
        self.assertGreater(s2, s1)

    def test_constraint_penalties_increase_with_worse_structure(self):
        cfg = RJGConstraintConfig()
        base_kwargs = dict(
            scenario="intent",
            mechanism="expressive aggression",
            label="mock",
            subject="speaker",
            target="listener",
            subject_options=["speaker", "manager"],
            target_options=["listener", "muslims"],
            parser_non_empty=True,
            anchors={"subject_anchor": "speaker", "target_anchor": "listener"},
            constraint_config=cfg,
        )
        aligned_penalty, aligned_detail = score_penalty_components(**base_kwargs)
        missing_anchor_penalty, _ = score_penalty_components(
            **{**base_kwargs, "anchors": {"subject_anchor": "", "target_anchor": "listener"}}
        )
        role_mismatch_penalty, _ = score_penalty_components(
            **{**base_kwargs, "subject": "outsider"}
        )
        low_compat_penalty, low_compat_detail = score_penalty_components(
            **{**base_kwargs, "label": "mitigate"}
        )

        self.assertLess(aligned_penalty, missing_anchor_penalty)
        self.assertLess(aligned_penalty, role_mismatch_penalty)
        self.assertGreater(low_compat_penalty, aligned_penalty)
        self.assertLess(low_compat_detail.get("compatibility_floor", 0.0), low_compat_penalty)
        self.assertEqual(aligned_detail, {})

    def test_constraint_details_report_breakdown(self):
        penalty, detail = score_penalty_components(
            scenario="attitude",
            mechanism="dominant detachment",
            label="hostile",
            subject="unknown",
            target="other",
            subject_options=["speaker", "actor"],
            target_options=["listener", "audience"],
            parser_non_empty=False,
            anchors={"subject_anchor": "speaker", "target_anchor": ""},
        )
        self.assertGreater(penalty, 0.0)
        self.assertIn("parser_missing", detail)
        self.assertIn("subject_anchor_mismatch", detail)
        self.assertIn("target_anchor_missing", detail)

    def test_repair_inconsistent_label(self):
        repaired, info = repair_inconsistent_label(
            scenario="intent",
            mechanism="expressive aggression",
            label="mitigate",
            text="lol idiot stop talking, you clown",
        )
        self.assertEqual(repaired, "mock")
        self.assertTrue(info["repaired"])
        self.assertIn(info["reason"], {"compatibility_gate_repair", "intent_mitigate_collapse_repair"})
        self.assertIn("best_label", info)

    def test_consistent_label_not_repaired(self):
        repaired, info = repair_inconsistent_label(
            scenario="attitude",
            mechanism="dominant detachment",
            label="contemptuous",
            text="pathetic loser, go away",
        )
        self.assertEqual(repaired, "contemptuous")
        self.assertFalse(info["repaired"])
        self.assertEqual(info["reason"], "compatibility_ok")

    def test_empty_text_is_stable(self):
        repaired, info = repair_inconsistent_label(
            scenario="intent",
            mechanism="expressive aggression",
            label="mock",
            text="",
        )
        self.assertEqual(repaired, "mock")
        self.assertFalse(info["repaired"])
        self.assertEqual(info["reason"], "empty_text_no_repair")

    def test_intent_generic_provoke_is_repaired(self):
        repaired, info = repair_inconsistent_label(
            scenario="intent",
            mechanism="expressive aggression",
            label="provoke",
            text="lol you clown, what a ridiculous joke",
        )
        self.assertEqual(repaired, "mock")
        self.assertTrue(info["repaired"])
        self.assertEqual(info["reason"], "intent_generic_collapse_repair")

    def test_attitude_generic_indifferent_is_repaired(self):
        repaired, info = repair_inconsistent_label(
            scenario="attitude",
            mechanism="dominant detachment",
            label="indifferent",
            text="pathetic loser, shut up and go away",
        )
        self.assertNotEqual(repaired, "indifferent")
        self.assertTrue(info["repaired"])
        self.assertEqual(info["reason"], "attitude_generic_collapse_repair")

    def test_intent_mitigate_is_repaired_under_toxic_cues(self):
        repaired, info = repair_inconsistent_label(
            scenario="intent",
            mechanism="prosocial deception",
            label="mitigate",
            text="the master race says know your place, what a joke",
        )
        self.assertNotEqual(repaired, "mitigate")
        self.assertTrue(info["repaired"])
        self.assertEqual(info["reason"], "intent_mitigate_collapse_repair")

    def test_affection_disgust_is_repaired_under_non_disgust_cues(self):
        repaired, info = repair_inconsistent_label(
            scenario="affection",
            mechanism="figurative semantics",
            label="disgusted",
            text="wtf this is so damn tiring, i'm seriously mad right now",
        )
        self.assertNotEqual(repaired, "disgusted")
        self.assertTrue(info["repaired"])
        self.assertEqual(info["reason"], "affection_disgust_collapse_repair")

    def test_penalty_includes_generic_collapse_components(self):
        intent_penalty, intent_detail = score_penalty_components(
            scenario="intent",
            mechanism="expressive aggression",
            label="provoke",
            subject="speaker",
            target="listener",
            subject_options=["speaker", "manager"],
            target_options=["listener", "audience"],
            parser_non_empty=True,
            text="lol clown, everyone laugh at him",
            anchors={"subject_anchor": "speaker", "target_anchor": "listener"},
        )
        attitude_penalty, attitude_detail = score_penalty_components(
            scenario="attitude",
            mechanism="dominant detachment",
            label="indifferent",
            subject="speaker",
            target="listener",
            subject_options=["speaker", "manager"],
            target_options=["listener", "audience"],
            parser_non_empty=True,
            text="what a pathetic loser",
            anchors={"subject_anchor": "speaker", "target_anchor": "listener"},
        )
        self.assertGreater(intent_penalty, 0.0)
        self.assertIn("intent_provoke_conflict", intent_detail)
        self.assertGreater(attitude_penalty, 0.0)
        self.assertIn("attitude_indifferent_conflict", attitude_detail)
        toxic_penalty, toxic_detail = score_penalty_components(
            scenario="intent",
            mechanism="prosocial deception",
            label="mitigate",
            subject="speaker",
            target="listener",
            subject_options=["speaker", "manager"],
            target_options=["listener", "audience"],
            parser_non_empty=True,
            text="master race and crack whore joke",
            anchors={"subject_anchor": "speaker", "target_anchor": "listener"},
        )
        self.assertGreater(toxic_penalty, 0.0)
        self.assertIn("intent_mitigate_toxic_conflict", toxic_detail)
        self.assertIn("intent_prosocial_toxic_conflict", toxic_detail)

    def test_affection_penalty_includes_disgust_and_mm_conflict(self):
        aff_penalty, aff_detail = score_penalty_components(
            scenario="affection",
            mechanism="multimodal incongruity",
            label="disgusted",
            subject="speaker",
            target="listener",
            subject_options=["speaker", "manager"],
            target_options=["listener", "audience"],
            parser_non_empty=True,
            text="wtf this is damn tiring and i'm mad",
            anchors={"subject_anchor": "speaker", "target_anchor": "listener"},
        )
        self.assertGreater(aff_penalty, 0.0)
        self.assertIn("affection_disgust_conflict", aff_detail)
        self.assertIn("affection_mm_without_signal", aff_detail)


if __name__ == "__main__":
    unittest.main()
