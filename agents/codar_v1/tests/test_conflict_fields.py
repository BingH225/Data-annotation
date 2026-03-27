import unittest

from codar.backends.mock_backend import MockBackend
from codar.types import SampleInput

from test_helpers import build_pipeline


class TestConflictFields(unittest.TestCase):
    def test_conflict_report_has_required_fields(self):
        runtime_cfg = {
            "pipeline": {
                "media_mode": "off",
                "max_stage_retries": 1,
                "alpha_rule": 0.6,
                "alpha_llm": 0.4,
                "max_video_frames": 4,
            },
            "media": {"remote_root": "/scratch/e1561245/Implicit_dataset", "allow_url_fallback": True},
        }
        scenario_policy = {
            "scenarios": {
                "affection": {
                    "rule_keywords": {
                        "multimodal incongruity": ["but", "irony"],
                        "figurative semantics": ["like"],
                        "affective deception": ["fine"],
                        "socio_cultural dependency": ["meme"],
                    }
                }
            }
        }
        pipeline = build_pipeline(MockBackend(), runtime_cfg, scenario_policy, max_backtrack_rounds=2)
        sample = SampleInput(
            sample_id="affection_0001",
            scenario="affection",
            text="Great job, but this is ironic meme content.",
            media={},
            subject_options=["speaker", "friend"],
            target_options=["audience", "crowd"],
            diversity={},
            ground_truth={},
        )
        result = pipeline.run_sample(sample, backend_meta={"provider": "mock"})
        self.assertIsNone(result.error)
        stage_conflict = [x for x in result.stage_artifacts if x.stage_id == "S3.1"][-1].output
        self.assertTrue(stage_conflict["conflicts"])
        c = stage_conflict["conflicts"][0]
        for k in ("trigger_evidence", "deviation_object", "deviation_direction", "confidence"):
            self.assertIn(k, c)


if __name__ == "__main__":
    unittest.main()
