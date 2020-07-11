import unittest

from src.shapeandshare.dicebox.models.optimizers import select_random_optimizer


class OptimizersTest(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    def test_get_random_optimizer(self):
        opt_one = select_random_optimizer()
        opt_two = select_random_optimizer()
        self.assertNotEqual(opt_one.value, opt_two.value)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(OptimizersTest())
