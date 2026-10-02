import unittest

from utils.rank_generator import RankGenerator


class TestRankGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = RankGenerator(
            dp=2,
            tp=2,
            sp1=2,
            sp2=2,
            order="tp-sp1-sp2-dp",
        )

    def test_atomic_groups(self):
        self.assertEqual(self.generator.get_ranks("tp")[0], [0, 1])
        self.assertEqual(self.generator.get_ranks("sp1")[0], [0, 2])
        self.assertEqual(self.generator.get_ranks("sp2")[0], [0, 4])
        self.assertEqual(self.generator.get_ranks("dp")[0], [0, 8])

    def test_spatial_plane_groups(self):
        self.assertEqual(
            self.generator.get_ranks("sp1-sp2"),
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
                [8, 10, 12, 14],
                [9, 11, 13, 15],
            ],
        )

    def test_model_parallel_groups(self):
        self.assertEqual(
            self.generator.get_ranks("tp-sp1-sp2"),
            [list(range(8)), list(range(8, 16))],
        )
        self.assertEqual(
            self.generator.get_ranks("dp-tp-sp1-sp2"),
            [list(range(16))],
        )


if __name__ == "__main__":
    unittest.main()
