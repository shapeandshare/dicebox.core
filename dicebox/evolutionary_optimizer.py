# Derived from https://github.com/harvitronix/neural-network-genetic-algorithm
# Derived source copyright: Matt Harvey, 2017, Derived source license: The MIT License
# See docs/Matt Harvey.LICENSE

"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import random
import logging
import dicebox.utils.helpers as helpers
from dicebox.config.dicebox_config import DiceboxConfig
from dicebox.dicebox_network import DiceboxNetwork


class EvolutionaryOptimizer:
    """Class that implements genetic algorithm for MLP optimization."""

    config_file = None
    lonestar_model_file = None
    config = None

    def __init__(self,
                 # nn_param_choices,
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
            nn_param_choices (dict): Possible network paremters
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
        # self.nn_param_choices = nn_param_choices

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
            network = DiceboxNetwork(nn_param_choices=self.nn_param_choices,
                                     config_file=self.config_file,
                                     lonestar_model_file=self.lonestar_model_file)
            network.create_random_v2()

            # Add the network to our population.
            pop.append(network)

        return pop

    # def create_lonestar(self, count):
    #     """Create a population of random networks.
    #
    #     Args:
    #         count (int): Number of networks to generate, aka the
    #             size of the population
    #
    #     Returns:
    #         (list): Population of network objects
    #
    #     """
    #     pop = []
    #     for _ in range(0, count):
    #         # Create a random network.
    #         network = DiceboxNetwork(nn_param_choices=self.nn_param_choices,
    #                                  config_file=self.config_file,
    #                                  lonestar_model_file=self.lonestar_model_file)
    #         network.create_lonestar_v2()
    #
    #         # Add the network to our population.
    #         pop.append(network)
    #
    #     return pop

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

    # TODO:
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

            child = {}

            # Loop through the parameters and pick params for the kid.
            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Now create a network object.
            network = DiceboxNetwork(nn_param_choices=self.nn_param_choices,
                                     config_file=self.config_file,
                                     lonestar_model_file=self.lonestar_model_file)
            network.create_set(child)

            children.append(network)

        return children

    def mutate(self, individual):
        # """Randomly mutate one part of the network.
        #
        # Args:
        #     individual (dict): The network parameters to mutate
        #
        # Returns:
        #     (Network): A randomly mutated network object
        #
        # """

        # v1 mutate
        # # Choose a random key.
        # mutation = random.choice(list(self.nn_param_choices.keys()))
        #
        # # Mutate one of the params.
        # individual.network[mutation] = random.choice(self.nn_param_choices[mutation])
        #
        # return individual


        # v2 mutate
        local_noise = self.mutate_chance

        # see if the optimizer is mutated
        if helpers.lucky(local_noise):
            # yep..  Select an optimizer
            optimizer_index = helpers.random_index(len(individual.config.TAXONOMY['optimizer']))
            optimizer = individual.config.TAXONOMY['optimizer'][optimizer_index]
            individual.network_v2['optimizer'] = optimizer

        # Determine the number of layers..
        layer_count = len(individual.layers)

        # now mess around within the layers
        for index in range(1, layer_count):
            layer = individual.layers[index - 1]
            # see if the layer is mutated
            if helpers.lucky(local_noise):
                # then change the layer type
                # how does this affect the weights, etc? :/
                logging.error('network layer type was mutated.')
                raise Exception('Not yet implemented!')
            else:
                # keep checking the individual layer attributes
                if layer.type == 'dropout' and helpers.lucky(local_noise):
                    # mutate the dropout rate
                    layer['rate'] = helpers.random()

                elif layer.type == 'dense':
                    if helpers.lucky(local_noise):
                        # mutate the layer size
                        logging.debug('Mutating layer size')
                        raise Exception('Not yet implemented!')

                    if helpers.lucky(local_noise):
                        # mutate activation function
                        activation_index = helpers.random_index(len(individual.config.TAXONOMY['activation']))
                        layer['activation'] = individual.config.TAXONOMY['activation'][activation_index - 1]
                else:
                    logging.debug('Unknown layer type')
                    raise Exception('Not yet implemented!')


        return individual

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
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Randomly mutate some of the networks we're keeping.
        for individual in parents:
            if self.mutate_chance > random.random():
                individual = self.mutate(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
