import logging
from typing import List

from tqdm import tqdm

from .config.dicebox_config import DiceboxConfig
from .models.dicebox_network import DiceboxNetwork
from .optimizers.evolutionary_optimizer import EvolutionaryOptimizer


class PrimordialPool:
    config: DiceboxConfig

    def __init__(self, config: DiceboxConfig):
        self.config = config

    def train_networks(self, networks):
        """Train each network.

        Args:
            networks (list): Current population of networks
            dataset (str): Dataset to use for training/evaluating
        """
        pbar = tqdm(total=len(networks))
        for network in networks:
            network.train()
            pbar.update(1)
        pbar.close()

    def get_average_accuracy(self, networks):
        """Get the average accuracy for a group of networks.

        Args:
            networks (list): List of networks

        Returns:
            float: The average accuracy of a population of networks.

        """
        total_accuracy = 0
        for network in networks:
            total_accuracy += network.accuracy

        return total_accuracy / len(networks)

    def generate(self):
        """Apply evolution"""

        generations: int = self.config.GENERATIONS
        population_size: int = self.config.POPULATION
        logging.info('Generations: %s' % generations)
        logging.info('Population: %s' % population_size)

        optimizer: EvolutionaryOptimizer = EvolutionaryOptimizer(config=self.config, retain=0.4, random_select=0.1,
                                                                 mutate_chance=0.2)
        networks: List[DiceboxNetwork] = optimizer.create_population(population_size)

        # Evolve the generation.
        for i in range(generations):
            logging.info("***Doing generation %d of %d***" % (i + 1, generations))
            logging.info('-' * 80)
            logging.info('Individuals in current generation')
            self.export_networks(networks)
            logging.info('-' * 80)

            # Train and get accuracy for networks.
            self.train_networks(networks)

            # Get the average accuracy for this generation.
            average_accuracy = self.get_average_accuracy(networks)

            # Print out the average accuracy each generation.
            logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
            logging.info('-' * 80)

            logging.info('Top 5 individuals in current generation')

            # Sort our final population.
            current_networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

            # Print out the top 5 networks.
            self.export_networks(current_networks[:5])

            # Evolve, except on the last iteration.
            if i != generations - 1:
                # Do the evolution.
                networks = optimizer.evolve(networks)

        # Sort our final population.
        networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

        # Print out the top 5 networks.
        self.export_networks(networks[:5])

    def export_networks(self, networks):
        """Print a list of networks.

        Args:
            networks (list): The population of networks

        """
        logging.info('-' * 80)
        for network in networks:
            network.print_network()
