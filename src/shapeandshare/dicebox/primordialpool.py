import logging
from typing import List

from tqdm import tqdm

from .config.dicebox_config import DiceboxConfig
from .models.dicebox_network import DiceboxNetwork
from .optimizers.evolutionary_optimizer import EvolutionaryOptimizer


class PrimordialPool:
    config: DiceboxConfig

    def __init__(self, config: DiceboxConfig) -> None:
        self.config = config

    # Updates (Trains) the set of networks passed in.
    @staticmethod
    def train_networks(networks: List[DiceboxNetwork]) -> None:
        pbar = tqdm(total=len(networks))
        for network in networks:
            accuracy: float = network.train()
            logging.info("accuracy: %f" % accuracy)
            pbar.update(1)
        pbar.close()

    @staticmethod
    def get_average_accuracy(networks: List[DiceboxNetwork]) -> float:
        """Get the average accuracy for a group of networks."""
        total_accuracy: float = 0.0
        for network in networks:
            total_accuracy += network.get_accuracy()
        return total_accuracy / len(networks)

    # TODO: JSON serialization..
    @staticmethod
    def export_networks(networks: List[DiceboxNetwork]) -> None:
        """Print a list of networks."""
        logging.info('-' * 80)
        for network in networks:
            logging.info(network.decompile())

    # The real entry point.  Invocation of this will apply the entire process.
    def generate(self) -> None:
        generations: int = self.config.GENERATIONS
        population_size: int = self.config.POPULATION
        logging.info('Generations: %s' % generations)
        logging.info('Population: %s' % population_size)

        optimizer: EvolutionaryOptimizer = EvolutionaryOptimizer(config=self.config,
                                                                 retain=0.4,
                                                                 random_select=0.1,
                                                                 mutate_chance=0.2)
        networks: List[DiceboxNetwork] = optimizer.create_population(population_size)

        # Evolve over the specified number of generations.
        for i in range(generations):
            logging.info("***Doing generation %d of %d***" % (i + 1, generations))
            logging.info('-' * 80)
            logging.info('Individuals in current generation')
            PrimordialPool.export_networks(networks)
            logging.info('-' * 80)

            # Train and get accuracy for networks.
            PrimordialPool.train_networks(networks)

            # Get the average accuracy for this generation.
            average_accuracy = PrimordialPool.get_average_accuracy(networks)

            # Print out the average accuracy each generation.
            logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
            logging.info('-' * 80)

            logging.info('Top 5 individuals in current generation')

            # Sort our final population.
            current_networks = sorted(networks, key=lambda x: x.get_accuracy(), reverse=True)

            # Print out the top 5 networks.
            PrimordialPool.export_networks(current_networks[:5])

            # Evolve, except on the last iteration.
            if i != generations - 1:
                # Do the evolution.
                networks = optimizer.evolve(networks)

        # Sort our final population.
        networks = sorted(networks, key=lambda x: x.get_accuracy(), reverse=True)

        # Print out the top 5(at max) networks.
        PrimordialPool.export_networks(networks[:5])
