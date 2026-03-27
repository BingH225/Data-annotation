from __future__ import annotations

import unittest

from codar.rjg.memory import build_memory_index, retrieve_similar_entries
from codar.types import SampleInput


def _sample(sample_id: str, scenario: str, text: str) -> SampleInput:
    return SampleInput(
        sample_id=sample_id,
        scenario=scenario,
        text=text,
        media={"audio_caption": ""},
        subject_options=["speaker", "friend"],
        target_options=["listener", "friend"],
        diversity={},
        ground_truth={},
    )


class TestRJGMemory(unittest.TestCase):
    def test_retrieve_loo_excludes_self(self):
        samples = [
            _sample("affection_0001", "affection", "i am happy but also annoyed"),
            _sample("affection_0002", "affection", "you are annoying and i am angry"),
            _sample("attitude_0001", "attitude", "yeah whatever i do not care"),
        ]
        index = build_memory_index(samples)
        query = build_memory_index([samples[0]]).entries[0]
        out = retrieve_similar_entries(
            index=index,
            scenario="affection",
            query_tf=query["token_freq"],
            query_subject_options=["speaker", "friend"],
            query_target_options=["listener", "friend"],
            query_sample_id="affection_0001",
            top_k=10,
            rerank_k=5,
            loo=True,
        )
        ids = [x["sample_id"] for x in out]
        self.assertNotIn("affection_0001", ids)


if __name__ == "__main__":
    unittest.main()
