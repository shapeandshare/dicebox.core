import copy
from functools import reduce
from operator import add
from typing import List

from ..config.dicebox_config import DiceboxConfig
from ..dicebox_network import DiceboxNetwork
from ..factories.network_factory import NetworkFactory
from ..models.network import Network
from ..models.network_config import NetworkConfig
from ..utils.helpers import lucky, random_index, random_index_between, random, random_strict


class EvolutionaryOptimizer(NetworkFactory):
    """Class that implements genetic algorithm for MLP optimization."""

    mutate_chance: float
    random_select: float
    retain: float

    def __init__(self, config: DiceboxConfig, retain=0.4, random_select=0.1, mutate_chance=0.2):
        super().__init__(config=config)

        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain

    def create_population(self, count: int) -> List[DiceboxNetwork]:
        # Create a population of random networks.
        population: List[DiceboxNetwork] = []
        for _ in range(0, count):
            # Create a random network.
            network: Network = self.create_random_network()
            dn = DiceboxNetwork(config=self.config)
            dn.load_network(network)

            # Add the network to our population.
            population.append(dn)
        return population

    @staticmethod
    def fitness(network: DiceboxNetwork) -> float:
        """Return the accuracy, which is our fitness function."""
        return network.get_accuracy()

    def grade(self, pop: List[DiceboxNetwork]) -> float:
        # Find average fitness for a population.
        summed: float = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother: DiceboxNetwork, father: DiceboxNetwork, offspringCount: int = 2) -> List[DiceboxNetwork]:
        # Creates offspring
        children: List[DiceboxNetwork] = []
        for _ in range(offspringCount):
            child = DiceboxNetwork(config=self.config)  # TODO: this should come from a parent... (though in practice are probably the same)

            # build our network definition

            network_definition = {
                'input_shape': self.config.INPUT_SHAPE,
                'output_size': self.config.NB_CLASSES
            }

            # Pick which parent's optimization function is passed on to offspring
            if lucky(0.5):
                network_definition['optimizer'] = mother.get_optimizer().value
            else:
                network_definition['optimizer'] = father.get_optimizer().value

            # Determine the number of layers
            if lucky(0.5):
                layer_count = mother.get_layer_count()
            else:
                layer_count = father.get_layer_count()

            # build layers
            network_definition['layers'] = []
            for layer_index in range(0, layer_count):
                # Pick which parent's layer is passed on to the offspring
                if lucky(0.5):
                    if layer_index < mother.get_layer_count():
                        layer = mother.get_layer(layer_index=layer_index)
                        network_definition['layers'].append(self.decompile_layer(layer))
                    elif layer_index < father.get_layer_count():
                        layer = father.get_layer(layer_index=layer_index)
                        network_definition['layers'].append(self.decompile_layer(layer))
                    else:
                        raise Exception('impossible breeding event occurred?')
                else:
                    if layer_index < father.get_layer_count():
                        layer = father.get_layer(layer_index=layer_index)
                        network_definition['layers'].append(self.decompile_layer(layer))
                    elif layer_index < mother.get_layer_count():
                        layer = mother.get_layer(layer_index=layer_index)
                        network_definition['layers'].append(self.decompile_layer(layer))
                    else:
                        raise Exception('impossible breeding event occurred?')
            child_network = self.create_network(network_definition=network_definition)
            child.load_network(network=child_network)
            children.append(child)
        return children

    def mutate(self, individual: DiceboxNetwork) -> DiceboxNetwork:

        # this is introduces chaos into the new entity
        local_noise = self.mutate_chance

        raw_individual_definition = individual.deconstruct_network()
        mutant = DiceboxNetwork(config=individual.get_config())

        # see if the optimizer is mutated
        if lucky(local_noise):
            clone.__network['optimizer'] = clone.select_random_optimizer()

        # Determine the number of layers..
        layer_count = len(clone.__network['layers'])

        # now mess around within the layers
        for index in range(0, layer_count):
            # see if the layer is mutated
            if lucky(local_noise):
                # then change the layer type
                # how does this affect the weights, etc? :/
                # logging.debug("layer = (%s)", layer)
                clone.__network['layers'][index - 1] = clone.build_random_layer()
                # mutations += 1
                # logging.debug("layer = (%s)", layer)
            else:
                layer = clone.__network['layers'][index - 1]

                # keep checking the individual layer attributes
                if layer['type'] == 'dropout':
                    if lucky(local_noise):
                        # mutate the dropout rate
                        # logging.debug("rate = (%s)", layer['rate'])
                        layer['rate'] = random_strict()
                        # mutations += 1
                        # logging.debug("rate = (%s)", layer['rate'])
                elif layer['type'] == 'dense':
                    if lucky(local_noise):
                        # mutate the layer size
                        # logging.debug('Mutating layer size')
                        # logging.debug("size = (%s)", layer['size'])
                        layer['size'] = random_index_between(clone.__config.TAXONOMY['min_neurons'],
                                                             clone.__config.TAXONOMY['max_neurons'])
                        # mutations += 1
                        # logging.debug("size = (%s)", layer['size'])
                    if lucky(local_noise):
                        # mutate activation function
                        # logging.debug("activation = (%s)", layer['activation'])
                        activation_index = random_index(len(clone.__config.TAXONOMY['activation']))
                        layer['activation'] = clone.__config.TAXONOMY['activation'][activation_index - 1]
                        # mutations += 1
                        # logging.debug("activation = (%s)", layer['activation'])
                else:
                    # logging.debug('Unknown layer type')
                    raise Exception('Not yet implemented!')
        # logging.debug("mutations: (%s)", mutations)
        # logging.debug("***************************************************")
        # return clone

        return mutant

    def evolve(self, pop):
        """Evolve a population of networks.

        Args:
            pop (list): A list of __network parameters

        Returns:
            (list): The evolved population of networks

        """
        # Get scores for each __network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every __network we want to keep.
        # TODO: can not deepcopy the keras senquences...
        parents = copy.deepcopy(graded[:retain_length])

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random():
                # TODO: deepcopy no longer works for sequentials... needs to be updated..
                parents.append(copy.deepcopy(individual))

        # Randomly mutate some of the networks we're keeping.
        for individual in parents:
            if lucky(self.mutate_chance):
                individual = self.mutate(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male_index = random_index_between(0, parents_length - 1)
            female_index = random_index_between(0, parents_length - 1)

            # Assuming they aren't the same __network...
            if male_index != female_index:
                male = parents[male_index]
                female = parents[female_index]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
