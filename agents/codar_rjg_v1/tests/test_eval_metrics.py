from __future__ import annotations

import unittest

from codar.eval.metrics import build_nus_compat_records, calculate_metrics_for_subset, compute_metrics


class TestEvalMetricsCompatibility(unittest.TestCase):
    def test_compute_metrics_accepts_prediction_payload_and_normalizes_taxonomy(self):
        gt_by_id = {
            "s1": {
                "scenario": "attitude",
                "subject": "speaker",
                "target": "listener",
                "mechanism": "dominant detachment",
                "label": "contemptuous",
                "domain": "",
                "culture": "",
            }
        }
        predictions = [
            {
                "sample_id": "s1",
                "prediction": {
                    "subject": "speaker",
                    "target": "listener",
                    "mechanism": "dominant_detachment",
                    "label": "contemptuous",
                },
            }
        ]
        metrics = compute_metrics(gt_by_id, predictions)
        self.assertEqual(metrics["total"], 1)
        self.assertEqual(metrics["accuracy"]["subject"], 1.0)
        self.assertEqual(metrics["accuracy"]["target"], 1.0)
        self.assertEqual(metrics["accuracy"]["mechanism"], 1.0)
        self.assertEqual(metrics["accuracy"]["label"], 1.0)
        self.assertEqual(metrics["accuracy"]["joint"], 1.0)

    def test_nus_compat_records_strict_match_with_normalized_taxonomy(self):
        gt_by_id = {
            "s2": {
                "scenario": "intent",
                "subject": "speaker",
                "target": "listener",
                "mechanism": "expressive aggression",
                "label": "mock",
                "domain": "Workplace",
                "culture": "General Culture",
            }
        }
        predictions = [
            {
                "sample_id": "s2",
                "final_prediction": {
                    "subject": "speaker",
                    "target": "listener",
                    "mechanism": "expressive aggression",
                    "label": "mock",
                },
            }
        ]
        records = build_nus_compat_records(gt_by_id, predictions)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["strict_match"])
        self.assertFalse(records[0]["error_analysis"]["mech_mismatch"])
        self.assertFalse(records[0]["error_analysis"]["label_mismatch"])
        subset_metrics = calculate_metrics_for_subset(records)
        self.assertEqual(subset_metrics["Strict_All4_Acc"], 1.0)
        self.assertEqual(subset_metrics["mechanism_Accuracy"], 1.0)
        self.assertEqual(subset_metrics["label_Accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
