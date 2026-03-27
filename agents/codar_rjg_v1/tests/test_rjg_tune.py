from __future__ import annotations

import unittest

from codar.cli import _normalize_weights


class TestRJGTune(unittest.TestCase):
    def test_normalize_weights_sum_one(self):
        w = _normalize_weights(
            {
                "retrieve_support": 0.4,
                "judge_mech": 0.25,
                "judge_label": 0.2,
                "judge_role": 0.1,
                "rule_cue": 0.05,
            }
        )
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
