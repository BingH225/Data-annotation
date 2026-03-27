import unittest

from codar.constants import VALID_LABELS, VALID_MECHANISMS
from codar.utils import map_to_closed_set


class TestClosedSet(unittest.TestCase):
    def test_map_mechanism_fuzzy(self):
        allowed = VALID_MECHANISMS["affection"]
        got = map_to_closed_set("multimodal_incongruity", allowed, allowed[0])
        self.assertEqual(got, "multimodal incongruity")

    def test_map_label_fallback(self):
        allowed = VALID_LABELS["intent"]
        got = map_to_closed_set("unknown_label", allowed, "mock")
        self.assertEqual(got, "mock")


if __name__ == "__main__":
    unittest.main()
