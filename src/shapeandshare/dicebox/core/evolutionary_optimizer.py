# Derived from https://github.com/harvitronix/neural-network-genetic-algorithm
# Derived source copyright: Matt Harvey, 2017, Derived source license: The MIT License
# See docs/Matt Harvey.LICENSE.txt

"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import copy

from .config import DiceboxConfig
from .dicebox_network import DiceboxNetwork
from .utils import lucky, random_index, random_index_between, random, random_strict


class EvolutionaryOptimizer:
    """Class that implements genetic algorithm for MLP optimization."""

    config_file = None
    lonestar_model_file = None
    config = None

    def __init__(self,
                 retain=0.4,
                 random_select=0.1,
                 mutate_chance=0.2,
                 config_file='./dicebox.config',
                 lonestar_model_file='./dicebox.lonestar.json'):
        if self.config_file is None:
            self.config_file = config_file

        if self.lonestar_model_file is None:
            self.lonestar_model_file = lonestar_model_file

        if self.config is None:
            self.config = DiceboxConfig(config_file=config_file,
                                        lonestar_model_file=lonestar_model_file)

        """Create an optimizer.

        Args:
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain

    def create_population(self, count):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): Population of network objects

        """
        pop = []
        for _ in range(0, count):
            # Create a random network.
            network = DiceboxNetwork(config_file=self.config_file,
                                     lonestar_model_file=self.lonestar_model_file)
            network.create_random_v2()

            # Add the network to our population.
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy_v2

    def grade(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks

        Returns:
            (float): The average accuracy of the population

        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects

        """
        children = []
        for _ in range(2):
            child = DiceboxNetwork(config_file=self.config_file,
                                     lonestar_model_file=self.lonestar_model_file)
            if child.network_v2 is None:
                child.network_v2 = {}
            if 'layers' not in child.network_v2:
                child.network_v2['layers'] = []

            # Set unchange-ables
            child.network_v2['input_shape'] = child.config.INPUT_SHAPE
            child.network_v2['output_size'] = child.config.NB_CLASSES

            # Pick which parent's optimization function is passed on to offspring
            if lucky(0.5):
                # logging.debug("child.network_v2['optimizer'] = mother(%s)", mother.network_v2['optimizer'])
                child.network_v2['optimizer'] = mother.network_v2['optimizer']
            else:
                # logging.debug("child.network_v2['optimizer'] = father(%s)", father.network_v2['optimizer'])
                child.network_v2['optimizer'] = father.network_v2['optimizer']

            # Determine the number of layers
            if lucky(0.5):
                # logging.debug("child layer length = mother(%s)", len(mother.network_v2['layers']))
                layer_count = len(mother.network_v2['layers'])
            else:
                # logging.debug("child layer length = father(%s)", len(father.network_v2['layers']))
                layer_count = len(father.network_v2['layers'])

            for layer_index in range(0, layer_count):
                # logging.debug("layer (%s/%s)", layer_index, layer_count)
                # Pick which parent's layer is passed on to the offspring
                if lucky(0.5):
                    if layer_index < len(mother.network_v2['layers']):
                        child.network_v2['layers'].append(mother.network_v2['layers'][layer_index])
                    elif layer_index < len(father.network_v2['layers']):
                        child.network_v2['layers'].append(father.network_v2['layers'][layer_index])
                    else:
                        raise Exception('impossible breeding event occurred!')
                else:
                    if layer_index < len(father.network_v2['layers']):
                        child.network_v2['layers'].append(father.network_v2['layers'][layer_index])
                    elif layer_index < len(mother.network_v2['layers']):
                        child.network_v2['layers'].append(mother.network_v2['layers'][layer_index])
                    else:
                        raise Exception('impossible breeding event occurred!')

            child.model_v2 = child.compile_model_v2(child.network_v2)
            children.append(child)
        return children

    def mutate(self, individual: DiceboxNetwork):
        # we will be performing a deepcopy on the incoming object.
        # It looks like Keras Sequencials no longer support this.
        # so we need to ensure we remove any compiled models on
        # the inbound object before proceeding.
        individual.model_v2 = {}

        # mutations = 0
        local_noise = self.mutate_chance
        # logging.debug("***************************************************")
        clone = copy.deepcopy(individual)
        # see if the optimizer is mutated
        if lucky(local_noise):
            # yep..  Select an optimizer
            # logging.debug("optimizer = (%s)", clone.network_v2['optimizer'])
            clone.network_v2['optimizer'] = clone.select_random_optimizer()
            # mutations += 1
            # logging.debug("optimizer = (%s)", clone.network_v2['optimizer'])

        # Determine the number of layers..
        layer_count = len(clone.network_v2['layers'])

        # now mess around within the layers
        for index in range(1, layer_count):
            # see if the layer is mutated
            if lucky(local_noise):
                # then change the layer type
                # how does this affect the weights, etc? :/
                # logging.debug("layer = (%s)", layer)
                clone.network_v2['layers'][index - 1] = clone.build_random_layer()
                # mutations += 1
                # logging.debug("layer = (%s)", layer)
            else:
                layer = clone.network_v2['layers'][index - 1]

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
                        layer['size'] = random_index_between(clone.config.TAXONOMY['min_neurons'],
                                                                     clone.config.TAXONOMY['max_neurons'])
                        # mutations += 1
                        # logging.debug("size = (%s)", layer['size'])
                    if lucky(local_noise):
                        # mutate activation function
                        # logging.debug("activation = (%s)", layer['activation'])
                        activation_index = random_index(len(clone.config.TAXONOMY['activation']))
                        layer['activation'] = clone.config.TAXONOMY['activation'][activation_index - 1]
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
            pop (list): A list of network parameters

        Returns:
            (list): The evolved population of networks

        """
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)

        # The parents are every network we want to keep.
        parents = copy.deepcopy(graded[:retain_length])

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random():
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

            # Assuming they aren't the same network...
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
