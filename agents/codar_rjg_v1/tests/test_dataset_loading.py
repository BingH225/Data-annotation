import unittest
from pathlib import Path

from codar.io.dataset import load_samples


class TestDatasetLoading(unittest.TestCase):
    def test_load_affection_subset(self):
        project_root = Path(__file__).resolve().parents[1]
        candidate1 = project_root / "data" / "datasetv3.18_hf_319_updatev1.json"
        candidate2 = Path(__file__).resolve().parents[3] / "Data" / "datasetv3.18_hf_319_updatev1.json"
        data_path = candidate1 if candidate1.exists() else candidate2
        samples = load_samples(data_path, scenario_filter="affection", limit=5)
        self.assertEqual(len(samples), 5)
        for s in samples:
            self.assertEqual(s.scenario, "affection")
            self.assertTrue(s.sample_id)
            self.assertIsInstance(s.subject_options, list)
            self.assertIsInstance(s.target_options, list)


if __name__ == "__main__":
    unittest.main()
