# Derived from https://github.com/harvitronix/neural-network-genetic-algorithm
# Derived source copyright: Matt Harvey, 2017, Derived source license: The MIT License
# See docs/Matt Harvey.LICENSE.txt

"""
Class that holds a genetic algorithm for evolving a __network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import copy

from typing import List

from .config import DiceboxConfig
from .dicebox_network import DiceboxNetwork
from .utils import lucky, random_index, random_index_between, random, random_strict


class EvolutionaryOptimizer:
    """Class that implements genetic algorithm for MLP optimization."""

    config_file = None
    lonestar_model_file = None
    config = None

    def __init__(self,
                 config: DiceboxConfig,
                 retain=0.4,
                 random_select=0.1,
                 mutate_chance=0.2):

        # self.lonestar_model_file: str = lonestar_model_file
        self.config: DiceboxConfig = config

        self.mutate_chance: float = mutate_chance
        self.random_select: float = random_select
        self.retain: float = retain

    def create_population(self, count: int) -> List[DiceboxNetwork]:
        # Create a population of random networks.
        population: List[DiceboxNetwork] = []
        for _ in range(0, count):
            # Create a random __network.
            network = DiceboxNetwork(self.config)
            network.generate_random_network()

            # Add the network to our population.
            population.append(network)

        return population

    @staticmethod
    def fitness(network: DiceboxNetwork) -> float:
        """Return the accuracy, which is our fitness function."""
        return network.accuracy()

    def grade(self, pop: List[DiceboxNetwork]) -> float:
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks

        Returns:
            (float): The average accuracy of the population

        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother: DiceboxNetwork, father: DiceboxNetwork) -> List[DiceboxNetwork]:
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects

        """
        children = []
        for _ in range(2):
            child = DiceboxNetwork(config=self.config)
            # if child.__network is None:
            #     child.__network = {}
            # if 'layers' not in child.__network:
            #     child.__network['layers'] = []

            # build our network definition
            network_definition = {
                'input_shape': self.config.INPUT_SHAPE,
                'output_size': self.config.NB_CLASSES
            }

            # Set unchange-ables
            # child.__network['input_shape'] = child.__config.INPUT_SHAPE
            # child.__network['output_size'] = child.__config.NB_CLASSES

            # Pick which parent's optimization function is passed on to offspring
            if lucky(0.5):
                # logging.debug("child.__network['optimizer'] = mother(%s)", mother.__network['optimizer'])
                # child.__network['optimizer'] = mother.network['optimizer']
                network_definition['optimizer'] = mother.get_optimizer().value
            else:
                # logging.debug("child.__network['optimizer'] = father(%s)", father.__network['optimizer'])
                # child.__network['optimizer'] = father.network['optimizer']
                network_definition['optimizer'] = father.get_optimizer().value

            # Determine the number of layers
            if lucky(0.5):
                # logging.debug("child layer length = mother(%s)", len(mother.__network['layers']))
                # layer_count = len(mother.network['layers'])
                layer_count = mother.get_layer_count()
            else:
                # logging.debug("child layer length = father(%s)", len(father.__network['layers']))
                # layer_count = len(father.network['layers'])
                layer_count = father.get_layer_count()

            network_definition['layers'] = []
            for layer_index in range(0, layer_count):
                # logging.debug("layer (%s/%s)", layer_index, layer_count)
                # Pick which parent's layer is passed on to the offspring
                if lucky(0.5):
                    if layer_index < len(mother.network['layers']):
                        # child.__network['layers'].append(mother.network['layers'][layer_index])
                        network_definition['layers'].append(mother.get_layer_definition(layer_index))

                    elif layer_index < len(father.network['layers']):
                        child.__network['layers'].append(father.network['layers'][layer_index])
                    else:
                        raise Exception('impossible breeding event occurred!')
                else:
                    if layer_index < len(father.network['layers']):
                        child.__network['layers'].append(father.network['layers'][layer_index])
                    elif layer_index < len(mother.network['layers']):
                        child.__network['layers'].append(mother.network['layers'][layer_index])
                    else:
                        raise Exception('impossible breeding event occurred!')

            child.__model = child.compile_model(child.__network)
            children.append(child)
        return children

    def mutate(self, individual: DiceboxNetwork) -> DiceboxNetwork:
        # we will be performing a deepcopy on the incoming object.
        # It looks like Keras Sequencials no longer support this.
        # so we need to ensure we remove any compiled models on
        # the inbound object before proceeding.
        individual.__model = {}

        # mutations = 0
        local_noise = self.mutate_chance
        # logging.debug("***************************************************")
        clone = copy.deepcopy(individual)
        # see if the optimizer is mutated
        if lucky(local_noise):
            # yep..  Select an optimizer
            # logging.debug("optimizer = (%s)", clone.__network['optimizer'])
            clone.__network['optimizer'] = clone.select_random_optimizer()
            # mutations += 1
            # logging.debug("optimizer = (%s)", clone.__network['optimizer'])

        # Determine the number of layers..
        layer_count = len(clone.__network['layers'])

        # now mess around within the layers
        for index in range(1, layer_count):
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
        return clone

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
        retain_length = int(len(graded)*self.retain)

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
