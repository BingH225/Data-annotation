from __future__ import annotations

import json
import unittest
from pathlib import Path

from codar.backends.base import BaseBackend
from codar.media import MediaResolver
from codar.prompting import PromptStore
from codar.rjg.pipeline import RJGPipeline, should_rejudge_branch
from codar.types import LLMResponse, SampleInput


class FakeBackend(BaseBackend):
    name = "fake"

    def __init__(self, parsed_json: dict[str, object]):
        self.parsed_json = parsed_json
        self.calls: list[dict[str, object]] = []

    def complete_json(self, prompt_text, prompt_id, media_items=None, temperature_override=None):
        self.calls.append(
            {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "temperature_override": temperature_override,
                "media_items_count": len(media_items or []),
            }
        )
        return LLMResponse(parsed_json=self.parsed_json, raw_text=json.dumps(self.parsed_json, ensure_ascii=False))

    def metadata(self):
        return {"provider": "fake", "model": "fake-model"}


class TestRJGPipelineTieBreak(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.prompt_store = PromptStore(root / "prompts")
        self.media_resolver = MediaResolver(
            runtime_cfg={
                "media": {
                    "remote_root": str(root),
                    "allow_url_fallback": False,
                    "local_cache_dir": str(root / "cache" / "media"),
                },
                "pipeline": {"media_mode": "off"},
            }
        )

    def _pipeline(self, backend: BaseBackend) -> RJGPipeline:
        return RJGPipeline(
            backend=backend,
            prompt_store=self.prompt_store,
            media_resolver=self.media_resolver,
            memory_index=object(),
            scenario_policy={},
            max_retries=0,
        )

    def _scored_pair(self):
        return [
            {
                "total_score": 0.64,
                "components": {
                    "judge_label": 0.31,
                    "judge_mech": 0.33,
                    "heuristic_agreement": 0.29,
                },
            },
            {
                "total_score": 0.61,
                "components": {
                    "judge_label": 0.39,
                    "judge_mech": 0.28,
                    "heuristic_agreement": 0.34,
                },
            },
        ]

    def test_low_confidence_branch_attempts_llm_tiebreak(self):
        backend = FakeBackend({"winner": "B", "reason_short": "candidate B is more specific"})
        pipeline = self._pipeline(backend)
        scored = self._scored_pair()
        triggered, reason, metrics = should_rejudge_branch(scored)
        self.assertTrue(triggered)
        self.assertTrue(pipeline._should_attempt_llm_tie_break(reason, metrics))

        idx, record, artifact = pipeline._llm_tie_break(
            scenario="attitude",
            text="Whatever, I don't care.",
            audio_caption="speaker sounds dismissive",
            scored=scored,
            media_items=[],
        )

        self.assertEqual(idx, 1)
        self.assertEqual(record["mode"], "llm")
        self.assertEqual(record["winner"], "B")
        self.assertEqual(artifact.status, "ok")
        self.assertEqual(backend.calls[0]["prompt_id"], "R5_tiebreak")

    def test_llm_parse_failure_falls_back_to_deterministic(self):
        backend = FakeBackend({"reason_short": "missing winner"})
        pipeline = self._pipeline(backend)
        scored = self._scored_pair()

        idx, record, artifact = pipeline._llm_tie_break(
            scenario="intent",
            text="lol go ahead and try it",
            audio_caption="speaker is taunting",
            scored=scored,
            media_items=[],
        )

        self.assertIsNone(idx)
        self.assertEqual(record["mode"], "llm_invalid")
        self.assertEqual(artifact.status, "ok")
        fallback_idx, fallback_record = pipeline._deterministic_tie_break(scored, "low_total_close_gap")
        self.assertEqual(fallback_idx, 1)
        self.assertEqual(fallback_record["winner"], "B")
        self.assertEqual(backend.calls[0]["prompt_id"], "R5_tiebreak")

    def test_clear_winner_does_not_attempt_llm_tiebreak(self):
        backend = FakeBackend({"winner": "A", "reason_short": "not needed"})
        pipeline = self._pipeline(backend)
        scored = [
            {
                "total_score": 0.86,
                "components": {
                    "judge_label": 0.82,
                    "judge_mech": 0.79,
                    "heuristic_agreement": 0.74,
                },
            },
            {
                "total_score": 0.63,
                "components": {
                    "judge_label": 0.52,
                    "judge_mech": 0.50,
                    "heuristic_agreement": 0.48,
                },
            },
        ]
        triggered, reason, metrics = should_rejudge_branch(scored)
        self.assertFalse(triggered)
        self.assertEqual(reason, "clear_winner")
        self.assertFalse(pipeline._should_attempt_llm_tie_break(reason, metrics))
        self.assertEqual(backend.calls, [])

    def test_affection_candidate_diversity_branches_are_generated(self):
        backend = FakeBackend({"winner": "A"})
        pipeline = self._pipeline(backend)
        sample = SampleInput(
            sample_id="aff_test_001",
            scenario="affection",
            text="wtf this is so damn tiring but i guess fine",
            media={},
            subject_options=["speaker", "friend"],
            target_options=["partner", "friend"],
            diversity={},
            ground_truth={},
        )
        anchors = {"subject_anchor": "speaker", "target_anchor": "partner"}
        candidates = pipeline._build_deterministic_view_candidates(
            sample=sample,
            scenario="affection",
            anchors=anchors,
            text=sample.text,
        )
        sources = {str(x.get("source", "")) for x in candidates}
        self.assertGreaterEqual(len(candidates), 8)
        self.assertIn("heuristic_affection_label_diversity", sources)
        self.assertIn("heuristic_affection_cross_mech_label", sources)

    def test_affection_label_arbitration_prefers_baseline_non_disgust(self):
        backend = FakeBackend({"winner": "A"})
        pipeline = self._pipeline(backend)
        label, info = pipeline._arbitrate_affection_label(
            text="wtf this is damn annoying and i am mad",
            rjg_label="disgusted",
            baseline_label="angry",
        )
        self.assertEqual(label, "angry")
        self.assertEqual(info["reason"], "baseline_default")

    def test_affection_label_arbitration_allows_disgust_override_on_strong_signal(self):
        backend = FakeBackend({"winner": "A"})
        pipeline = self._pipeline(backend)
        label, info = pipeline._arbitrate_affection_label(
            text="ew this is gross and repulsive",
            rjg_label="disgusted",
            baseline_label="happy",
        )
        self.assertEqual(label, "disgusted")
        self.assertEqual(info["reason"], "rjg_disgust_override")

    def test_affection_candidate_adjustment_penalizes_disgust_overuse(self):
        backend = FakeBackend({"winner": "A"})
        pipeline = self._pipeline(backend)
        adjust_disgust = pipeline._affection_candidate_adjustment(
            text="wtf this is damn annoying and i am mad",
            candidate_label="disgusted",
            baseline_label="angry",
        )
        adjust_baseline = pipeline._affection_candidate_adjustment(
            text="wtf this is damn annoying and i am mad",
            candidate_label="angry",
            baseline_label="angry",
        )
        self.assertGreater(adjust_disgust["penalty_add"], 0.0)
        self.assertGreater(adjust_baseline["label_bonus"], 0.0)


if __name__ == "__main__":
    unittest.main()
