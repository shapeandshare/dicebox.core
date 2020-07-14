import copy
from functools import reduce
from operator import add
from typing import List, Any, Tuple

from ..config.dicebox_config import DiceboxConfig
from ..factories.network_factory import NetworkFactory
from ..models.dicebox_network import DiceboxNetwork
from ..models.network import Network
from ..models.optimizers import select_random_optimizer
from ..utils.helpers import lucky, random_index, random_index_between, dicebox_random, random_strict


class EvolutionaryOptimizer(NetworkFactory):
    """Class that implements genetic algorithm for MLP optimization."""

    mutate_chance: float
    random_select: float
    retain: float

    def __init__(self,
                 config: DiceboxConfig,
                 retain: float = 0.4,
                 random_select: float = 0.1,
                 mutate_chance: float = 0.2):
        super().__init__(config=config)

        self.mutate_chance: float = mutate_chance
        self.random_select: float = random_select
        self.retain: float = retain

    def create_population(self, count: int) -> List[DiceboxNetwork]:
        # Create a population of random networks.
        population: List[DiceboxNetwork] = []
        for _ in range(0, count):
            # Create a random network.

            random_network: Network = self.create_random_network()
            dn: DiceboxNetwork = DiceboxNetwork(config=self.config,
                                                input_shape=random_network.get_input_shape(),
                                                output_size=random_network.get_output_size(),
                                                optimizer=random_network.get_optimizer(),
                                                layers=random_network.get_layers())

            # Add the network to our population.
            population.append(dn)
        return population

    @staticmethod
    def fitness(network: DiceboxNetwork) -> float:
        """Return the accuracy, which is our fitness function."""
        return network.get_accuracy()

    def grade(self, population: List[DiceboxNetwork]) -> float:
        # Find average fitness for a population.
        summed: float = reduce(add, (self.fitness(network) for network in population))
        return summed / float((len(population)))

    ## This function operates on the genome of the individual, and not the higher order constructs.
    # TODO: this should include variation between the N parents as well.
    # TODO: what would it mean if the config 's came from the parents..?
    def breed(self, mother, father, offspringCount: int = 2) -> List[Any]:
        # Creates offspring
        children: List[Any] = []

        for _ in range(offspringCount):
            #
            # build our network genome
            #

            child = {
                'input_shape': self.config.INPUT_SHAPE,
                'output_size': self.config.NB_CLASSES
            }

            #
            # Pick which parent's optimization function is passed on to offspring
            #
            if lucky(0.5):
                child['optimizer'] = mother['optimizer']
            else:
                child['optimizer'] = father['optimizer']

            #
            # Determine the number of layers
            #
            if lucky(0.5):
                layer_count: int = len(mother['layers'])
            else:
                layer_count: int = len(father['layers'])

            #
            # build layers
            #
            child['layers'] = []
            for layer_index in range(0, layer_count):
                # Pick which parent's layer is passed on to the offspring
                # TODO: this should include variation between the N parents as well.
                if lucky(0.5):
                    if layer_index < len(mother['layers']):
                        layer = mother['layers'][layer_index]
                        child['layers'].append(layer)
                    elif layer_index < len(father['layers']):
                        layer = father['layers'][layer_index]
                        child['layers'].append(layer)
                    else:
                        raise Exception('impossible breeding event occurred?')
                else:
                    if layer_index < len(father['layers']):
                        layer = father['layers'][layer_index]
                        child['layers'].append(layer)
                    elif layer_index < len(mother['layers']):
                        layer = mother['layers'][layer_index]
                        child['layers'].append(layer)
                    else:
                        raise Exception('impossible breeding event occurred?')
            children.append(child)
        return children

    ## Note:
    ## This function operates on the genome of the individual, and not the higher order constructs.
    def mutate(self, individual):
        mutant = copy.deepcopy(individual)

        # this introduces chaos into the new entity
        local_noise: float = self.mutate_chance

        # TODO: possibly only of the parents types..
        # see if the optimizer is mutated
        if lucky(local_noise):
            mutant['optimizer'] = select_random_optimizer().value

        # Determine the number of layers..
        layer_count = len(mutant['layers'])

        # TODO: adjust the number of layers (its easy to remove, adding could be random)?
        # now mess around within the layers
        for index in range(0, layer_count):
            # see if the layer is mutated
            if lucky(local_noise):
                # then change the layer type
                mutant['layers'][index - 1] = self.decompile_layer(self.build_random_layer())
            else:
                layer = mutant['layers'][index - 1]

                # keep checking the individual layer attributes
                if layer['type'] == 'dropout':
                    if lucky(local_noise):
                        # mutate the dropout rate
                        layer['rate'] = random_strict()
                elif layer['type'] == 'dense':
                    if lucky(local_noise):
                        # mutate the layer size
                        layer['size'] = random_index_between(self.config.TAXONOMY['min_neurons'],
                                                             self.config.TAXONOMY['max_neurons'])
                    if lucky(local_noise):
                        # mutate activation function
                        activation_index = random_index(len(self.config.TAXONOMY['activation']))
                        layer['activation'] = self.config.TAXONOMY['activation'][activation_index - 1]
                else:
                    raise Exception('Not yet implemented!')
        return mutant

    def evolve(self, population: List[DiceboxNetwork]) -> List[DiceboxNetwork]:
        """Evolve a population of networks."""

        # Get scores for each network.
        graded_decompiled_population: List[Tuple[float, Any]] = [(self.fitness(network), network.decompile()) for
                                                                 network in population]

        # Sort on the scores.
        ranked_population: List[Any] = [x[1] for x in
                                        sorted(graded_decompiled_population, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length: int = int(len(ranked_population) * self.retain)

        # The parents are every network we want to keep.
        parent_genomes: List[Any] = copy.deepcopy(ranked_population[:retain_length])

        # For those we aren't keeping, randomly keep some anyway.
        for individual in ranked_population[retain_length:]:
            if self.random_select > dicebox_random():
                parent_genomes.append(copy.deepcopy(individual))

        # Randomly mutate some of the networks we're keeping.
        for individual in parent_genomes:
            if lucky(self.mutate_chance):
                individual = self.mutate(individual=individual)

        # Now find out how many spots we have left to fill.
        parents_length: int = len(parent_genomes)
        desired_length: int = len(population) - parents_length
        children: List[Any] = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male_index: int = random_index_between(0, parents_length - 1)
            female_index: int = random_index_between(0, parents_length - 1)

            if parents_length == 1:
                # then we do not have enough parents to breed,
                # so mutate..
                mutant: Any = self.mutate(individual=parent_genomes[0])
                children.append(mutant)
            elif parents_length < 1:
                # then there are no parents..
                # generate a random network.
                random_network: Any = self.create_random_network().decompile()
                children.append(random_network)
            else:
                # then we can bread normally
                # Assuming they aren't the same network...
                # for a very small populations this might be required..
                print('-----------------------------------------------------------------------------------------------')
                print(male_index)
                print(parent_genomes[male_index])
                print(female_index)
                print(parent_genomes[female_index])
                print('-----------------------------------------------------------------------------------------------')
                if male_index != female_index:
                    male = parent_genomes[male_index]
                    female = parent_genomes[female_index]

                    # Breed them.
                    offspring: List[Any] = self.breed(male, female)

                    # Add the children one at a time.
                    for baby in offspring:
                        # Don't grow larger than desired length.
                        if len(children) < desired_length:
                            children.append(baby)

        parent_genomes.extend(children)
        parent_networks: List[Network] = [(self.create_network(genome)) for genome in parent_genomes]
        parents: List[DiceboxNetwork] = [(self.build_dicebox_network(network=network)) for network in parent_networks]
        return parents

    def build_dicebox_network(self, network: Network) -> DiceboxNetwork:
        return DiceboxNetwork(config=self.config,
                              input_shape=network.get_input_shape(),
                              output_size=network.get_output_size(),
                              optimizer=network.get_optimizer(),
                              layers=network.get_layers())
