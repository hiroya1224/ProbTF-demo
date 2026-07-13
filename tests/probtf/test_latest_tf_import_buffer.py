import unittest

from probtf_ros.bridge import LatestTfImportBuffer


class _Transform:
    def __init__(self, parent, child, stamp):
        self.header = type("Header", (), {})()
        self.header.frame_id = parent
        self.header.stamp = stamp
        self.child_frame_id = child


class LatestTfImportBufferTest(unittest.TestCase):
    def test_new_sample_replaces_pending_work_for_same_edge(self):
        buffer = LatestTfImportBuffer()
        buffer.put(_Transform("/base", "/ref/tool", 1.0), "old")
        latest = _Transform("base", "ref/tool", 2.0)
        buffer.put(latest, "new")

        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.drain(), ((latest, "new"),))
        self.assertEqual(len(buffer), 0)

    def test_different_edges_are_drained_in_deterministic_order(self):
        buffer = LatestTfImportBuffer()
        edge_b = _Transform("world", "b", 1.0)
        edge_a = _Transform("world", "a", 1.0)
        buffer.put(edge_b, "source")
        buffer.put(edge_a, "source")

        self.assertEqual(
            buffer.drain(),
            ((edge_a, "source"), (edge_b, "source")),
        )

    def test_out_of_order_sample_does_not_replace_newer_pending_sample(self):
        buffer = LatestTfImportBuffer()
        latest = _Transform("world", "tool", 2.0)
        buffer.put(latest, "new")

        self.assertFalse(buffer.put(_Transform("world", "tool", 1.0), "old"))
        self.assertEqual(buffer.drain(), ((latest, "new"),))

    def test_empty_frame_ids_are_not_staged(self):
        buffer = LatestTfImportBuffer()

        self.assertFalse(buffer.put(_Transform("", "ref/tool", 1.0), "source"))
        self.assertFalse(buffer.put(_Transform("world", "/", 1.0), "source"))
        self.assertEqual(buffer.drain(), ())


if __name__ == "__main__":
    unittest.main()
