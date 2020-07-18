import json
import logging
import os
from typing import List

import uuid
from tqdm import tqdm

from .config.dicebox_config import DiceboxConfig
from .models.dicebox_network import DiceboxNetwork
from .optimizers.evolutionary_optimizer import EvolutionaryOptimizer
from .utils.helpers import make_sure_path_exists


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

    @staticmethod
    def print_networks(networks: List[DiceboxNetwork]) -> None:
        """Print a list of networks."""
        logging.info('-' * 80)
        for network in networks:
            logging.info(network.decompile())

    def export_population(self, population_id: str, generation: int, population: List[DiceboxNetwork]) -> None:
        output_directory: str = str(os.path.join(self.config.POPULATION_DIR, population_id, str(generation)))
        output_file: str = str(os.path.join(output_directory, 'population.json'))
        logging.info("Writing populations to: (%s)" % output_file)
        make_sure_path_exists(output_directory)

        with open(output_file, 'w') as file:
            population_genome = []
            for individual in population:
                individual_full = {
                    'accuracy': individual.get_accuracy(),
                    'genome': individual.decompile()

                }
                population_genome.append(individual_full)
            output = {
                'population': population_genome,
                'average_accuracy': self.get_average_accuracy(population)
            }
            file.write(json.dumps(output))

    # The real entry point.  Invocation of this will apply the entire process.
    def generate(self, population_file: str = None) -> str:
        population_id = str(uuid.uuid4())
        generations: int = self.config.GENERATIONS
        population_size: int = self.config.POPULATION
        logging.info("Population ID: %s" % population_id)
        logging.info('Generations: %s' % generations)
        logging.info('Population size: %s' % population_size)

        optimizer: EvolutionaryOptimizer = EvolutionaryOptimizer(config=self.config,
                                                                 retain=0.4,
                                                                 random_select=0.1,
                                                                 mutate_chance=0.2)

        # Determine if we are loading a previous population.
        if population_file is None:
            # Do not load a previous population, generate a new one.
            networks: List[DiceboxNetwork] = optimizer.create_population(population_size)
        else:
            # load the specified population.
            with open(population_file, 'r') as file:
                population_raw = json.loads(file.read())
                networks: List[DiceboxNetwork] = optimizer.create_population(size=population_size, population_definition=population_raw)

        # Evolve over the specified number of generations.
        for i in range(generations):
            logging.info("***Doing generation %d of %d***" % (i + 1, generations))
            logging.info('-' * 80)
            logging.info('Individuals in current generation')
            PrimordialPool.print_networks(networks)
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
            PrimordialPool.print_networks(current_networks[:5])

            self.export_population(population_id=population_id, generation=i, population=networks)

            # Evolve, except on the last iteration.
            if i != generations - 1:
                # Do the evolution.
                networks = optimizer.evolve(networks)

        # Sort our final population.
        networks = sorted(networks, key=lambda x: x.get_accuracy(), reverse=True)

        # Print out the top 5(at max) networks.
        PrimordialPool.print_networks(networks[:5])

        return population_id

