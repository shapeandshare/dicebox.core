import unittest

from src.shapeandshare.dicebox.config import NetworkConfig
from src.shapeandshare.dicebox.models.dicebox_network import DiceboxNetwork
from src.shapeandshare.dicebox.models.network import Network
from src.shapeandshare.dicebox.optimizers.evolutionary_optimizer import EvolutionaryOptimizer
from src.shapeandshare.dicebox.config.dicebox_config import DiceboxConfig
from src.shapeandshare.dicebox.factories.network_factory import NetworkFactory


class EvolutionaryOptimizerTest(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    TEST_DATA_BASE = 'test/fixtures'
    local_config_file = '%s/dicebox.config' % TEST_DATA_BASE
    local_lonestar_model_file = '%s/dicebox.lonestar.json' % TEST_DATA_BASE

    def test_breed(self):
        dc: DiceboxConfig = DiceboxConfig(config_file=self.local_config_file)
        nf = NetworkFactory(config=dc)

        mother_network: Network = nf.create_random_network()
        # mother_network_config: NetworkConfig = nf.create_network_config(mother_network.decompile())
        # mother = DiceboxNetwork(config=dc, network_config=mother_network_config, create_fsc=True, disable_data_indexing=True)

        father_network: Network = nf.create_random_network()
        # father_network_config: NetworkConfig = nf.create_network_config(father_network.decompile())
        # father = DiceboxNetwork(config=dc, network_config=father_network_config, create_fsc=True, disable_data_indexing=True)

        op = EvolutionaryOptimizer(config=dc,
                                   retain=0.4,
                                   random_select=0.1,
                                   mutate_chance=0.2)

        # Breed them.
        babies = op.breed(mother_network.decompile(), father_network.decompile())
        self.assertTrue(len(babies) == 2)
        for child in babies:
            # child.print_network()
            self.assertNotEqual(mother_network.decompile(), child)
            self.assertNotEqual(father_network.decompile(), child)
            self.assertNotEqual(mother_network.decompile(), father_network.decompile())

    # def test_mutate(self):
    #     op = EvolutionaryOptimizer(retain=0.4,
    #                                random_select=0.1,
    #                                mutate_chance=1,
    #                                config_file=self.local_config_file,
    #                                lonestar_model_file=self.local_lonestar_model_file)
    #
    #     individual = DiceboxNetwork(config_file=self.local_config_file,
    #                              lonestar_model_file=self.local_lonestar_model_file)
    #     individual.create_random()
    #     individual.__model = individual.compile_model(individual.__network)
    #     before_network = individual.__network
    #
    #     mutant = op.mutate(individual)
    #
    #     mutant.__model = mutant.compile_model(mutant.__network)
    #     after_network = mutant.__network
    #     self.assertNotEqual(id(individual), id(mutant))
    #     self.assertNotEqual(id(individual.__network), id(mutant.__network))
    #     self.assertNotEqual(id(before_network), id(after_network))
    #     self.assertNotEqual(before_network, after_network)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(EvolutionaryOptimizerTest())
