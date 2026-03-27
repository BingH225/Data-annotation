import unittest

from codar.backends.mock_backend import MockBackend
from codar.types import LLMResponse, SampleInput

from test_helpers import build_pipeline


class CriticAlwaysFailBackend(MockBackend):
    def complete_json(self, prompt_text, prompt_id, media_items=None, temperature_override=None):
        if prompt_id == "P5_critic":
            out = {
                "pass": False,
                "issues": ["mock issue"],
                "revision_instructions": "revise conflict alignment",
                "backtrack_to": "S3",
            }
            return LLMResponse(parsed_json=out, raw_text=str(out), usage={})
        return super().complete_json(prompt_text, prompt_id, media_items, temperature_override=temperature_override)


class TestRetryLimit(unittest.TestCase):
    def test_backtrack_round_limit(self):
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
                        "multimodal incongruity": ["but"],
                        "figurative semantics": ["like"],
                        "affective deception": ["fine"],
                        "socio_cultural dependency": ["meme"],
                    }
                }
            }
        }
        pipeline = build_pipeline(CriticAlwaysFailBackend(), runtime_cfg, scenario_policy, max_backtrack_rounds=2)
        sample = SampleInput(
            sample_id="affection_0002",
            scenario="affection",
            text="I am fine but not really.",
            media={},
            subject_options=["speaker"],
            target_options=["listener"],
            diversity={},
            ground_truth={},
        )
        result = pipeline.run_sample(sample, backend_meta={"provider": "mock"})
        critic_calls = [x for x in result.stage_artifacts if x.stage_id == "S5"]
        self.assertEqual(len(critic_calls), 3)
        self.assertEqual(result.trace.get("backtrack_rounds"), 2)


if __name__ == "__main__":
    unittest.main()
