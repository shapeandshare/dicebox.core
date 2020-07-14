import unittest

from src.shapeandshare.dicebox.models.optimizers import select_random_optimizer


class OptimizersTest(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    def test_get_random_optimizer(self):
        self.assertNotEqual(select_random_optimizer().value, select_random_optimizer().value)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(OptimizersTest())
