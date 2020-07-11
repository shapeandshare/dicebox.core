import unittest


class OptimizersTest(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(OptimizersTest())
