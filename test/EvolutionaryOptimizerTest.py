import unittest

from src.shapeandshare.dicebox.core import DiceboxNetwork, EvolutionaryOptimizer


class EvolutionaryOptimizerTest(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    TEST_DATA_BASE = 'test/fixtures'
    local_config_file = '%s/dicebox.config' % TEST_DATA_BASE
    local_lonestar_model_file = '%s/dicebox.lonestar.json' % TEST_DATA_BASE

    def test_breed(self):
        mother = DiceboxNetwork(config_file=self.local_config_file,
                                 lonestar_model_file=self.local_lonestar_model_file)
        mother.create_random()

        father = DiceboxNetwork(config_file=self.local_config_file,
                                 lonestar_model_file=self.local_lonestar_model_file)
        father.create_random()

        op = EvolutionaryOptimizer(retain=0.4,
                                   random_select=0.1,
                                   mutate_chance=0.2,
                                   config_file=self.local_config_file,
                                   lonestar_model_file=self.local_lonestar_model_file)

        # Breed them.
        babies = op.breed(mother, father)
        self.assertTrue(len(babies) == 2)
        for child in babies:
            child.print_network()
            self.assertNotEqual(mother, child)
            self.assertNotEqual(father, child)
            self.assertNotEqual(mother, father)

    def test_mutate(self):
        op = EvolutionaryOptimizer(retain=0.4,
                                   random_select=0.1,
                                   mutate_chance=1,
                                   config_file=self.local_config_file,
                                   lonestar_model_file=self.local_lonestar_model_file)

        individual = DiceboxNetwork(config_file=self.local_config_file,
                                 lonestar_model_file=self.local_lonestar_model_file)
        individual.create_random()
        individual.model = individual.compile_model(individual.network)
        before_network = individual.network

        mutant = op.mutate(individual)

        mutant.model = mutant.compile_model(mutant.network)
        after_network = mutant.network
        self.assertNotEqual(id(individual), id(mutant))
        self.assertNotEqual(id(individual.network), id(mutant.network))
        self.assertNotEqual(id(before_network), id(after_network))
        self.assertNotEqual(before_network, after_network)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(EvolutionaryOptimizerTest())
