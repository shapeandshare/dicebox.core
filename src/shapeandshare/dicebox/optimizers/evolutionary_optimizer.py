import copy
from functools import reduce
from operator import add
from typing import List, Any, Tuple

from ..config import NetworkConfig
from ..config.dicebox_config import DiceboxConfig
from ..models.dicebox_network import DiceboxNetwork
from ..factories.network_factory import NetworkFactory
from ..models.network import Network
from ..models.optimizers import Optimizers
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
            network_config: NetworkConfig = self.create_network_config(network_definition=random_network.decompile())
            dn: DiceboxNetwork = DiceboxNetwork(config=self.config, network_config=network_config)

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

    # TODO: Support parents of N numbers..
    # offspring should would not be deterministic (the number should varry by some hyper-parameter controllable value)
    # def breed(self, mother: DiceboxNetwork, father: DiceboxNetwork, offspringCount: int = 2) -> List[DiceboxNetwork]:
    #     # Creates offspring
    #     children: List[DiceboxNetwork] = []
    #     for _ in range(offspringCount):
    #         #
    #         # TODO: what would it mean if the config 's came from the parents..?
    #         #
    #
    #         child: DiceboxNetwork = DiceboxNetwork(config=self.config)
    #
    #         #
    #         # build our network definition
    #         #
    #
    #         # TODO: what would it mean if the config 's came from the parents..?
    #         network_definition: Any = {
    #             'input_shape': self.config.INPUT_SHAPE,
    #             'output_size': self.config.NB_CLASSES
    #         }
    #
    #         #
    #         # Pick which parent's optimization function is passed on to offspring
    #         #
    #         # TODO: Support N parents
    #         if lucky(0.5):
    #             network_definition['optimizer'] = mother.get_optimizer().value
    #         else:
    #             network_definition['optimizer'] = father.get_optimizer().value
    #
    #         #
    #         # Determine the number of layers
    #         #
    #         # TODO: this should include variation between the N parents as well.
    #         if lucky(0.5):
    #             layer_count: int = mother.get_layer_count()
    #         else:
    #             layer_count: int = father.get_layer_count()
    #
    #         #
    #         # build layers
    #         #
    #         network_definition['layers'] = []
    #         for layer_index in range(0, layer_count):
    #             # Pick which parent's layer is passed on to the offspring
    #             # TODO: this should include variation between the N parents as well.
    #             if lucky(0.5):
    #                 if layer_index < mother.get_layer_count():
    #                     layer = mother.get_layer(layer_index=layer_index)
    #                     network_definition['layers'].append(self.decompile_layer(layer))
    #                 elif layer_index < father.get_layer_count():
    #                     layer = father.get_layer(layer_index=layer_index)
    #                     network_definition['layers'].append(self.decompile_layer(layer))
    #                 else:
    #                     raise Exception('impossible breeding event occurred?')
    #             else:
    #                 if layer_index < father.get_layer_count():
    #                     layer = father.get_layer(layer_index=layer_index)
    #                     network_definition['layers'].append(self.decompile_layer(layer))
    #                 elif layer_index < mother.get_layer_count():
    #                     layer = mother.get_layer(layer_index=layer_index)
    #                     network_definition['layers'].append(self.decompile_layer(layer))
    #                 else:
    #                     raise Exception('impossible breeding event occurred?')
    #         child_network = self.create_network(network_definition=network_definition)
    #         child.load_network(network=child_network)
    #         children.append(child)
    #     return children

    def breed(self, mother: Any, father: Any, offspringCount: int = 2) -> List[Any]:
        # Creates offspring
        children: List[Any] = []
        for _ in range(offspringCount):
            #
            # build our network definition
            #

            # TODO: what would it mean if the config 's came from the parents..?
            child: Any = {
                'input_shape': self.config.INPUT_SHAPE,
                'output_size': self.config.NB_CLASSES
            }

            #
            # Pick which parent's optimization function is passed on to offspring
            #
            # TODO: Support N parents
            if lucky(0.5):
                child['optimizer'] = mother['optimizer']
            else:
                child['optimizer'] = father['optimizer']

            #
            # Determine the number of layers
            #
            # TODO: this should include variation between the N parents as well.
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
                    elif layer_index < mother.get_layer_count():
                        layer = mother['layers'][layer_index]
                        child['layers'].append(layer)
                    else:
                        raise Exception('impossible breeding event occurred?')
            # child_network = self.create_network(network_definition=network_definition)
            # child.load_network(network=child_network)
            # children.append(child)
            children.append(child)
        return children

    def mutate(self, individual: DiceboxNetwork) -> DiceboxNetwork:

        # this introduces chaos into the new entity
        local_noise: float = self.mutate_chance

        # Review the 'raw' genome of the individual
        raw_individual_definition: Any = individual.decompile()

        # TODO: possibly only of the parents types..
        # see if the optimizer is mutated
        if lucky(local_noise):
            raw_individual_definition['optimizer'] = Optimizers.select_random_optimizer().value

        # Determine the number of layers..
        layer_count = len(raw_individual_definition['layers'])

        # TODO: adjust the number of layers (its easy to remove, adding could be random)?
        # now mess around within the layers
        for index in range(0, layer_count):
            # see if the layer is mutated
            if lucky(local_noise):
                # then change the layer type
                # how does this affect the weights, etc? :/
                # logging.debug("layer = (%s)", layer)
                # clone.__network['layers'][index - 1] = clone.build_random_layer()
                raw_individual_definition['layers'][index - 1] = self.decompile_layer(self.build_random_layer())

                # mutations += 1
                # logging.debug("layer = (%s)", layer)
            else:
                layer = raw_individual_definition['layers'][index - 1]

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
                        layer['size'] = random_index_between(self.config.TAXONOMY['min_neurons'],
                                                             self.config.TAXONOMY['max_neurons'])
                        # mutations += 1
                        # logging.debug("size = (%s)", layer['size'])
                    if lucky(local_noise):
                        # mutate activation function
                        # logging.debug("activation = (%s)", layer['activation'])
                        activation_index = random_index(len(self.config.TAXONOMY['activation']))
                        layer['activation'] = self.config.TAXONOMY['activation'][activation_index - 1]
                        # mutations += 1
                        # logging.debug("activation = (%s)", layer['activation'])
                else:
                    # logging.debug('Unknown layer type')
                    raise Exception('Not yet implemented!')
        # logging.debug("mutations: (%s)", mutations)
        # logging.debug("***************************************************")
        # return clone

        mutant_network_config: NetworkConfig = self.create_network_config(network_definition=raw_individual_definition)
        mutant: DiceboxNetwork = DiceboxNetwork(config=self.config, network_config=mutant_network_config)
        return mutant

    def evolve(self, population: List[DiceboxNetwork]) -> List[DiceboxNetwork]:
        """Evolve a population of networks."""

        # Get scores for each network.
        # graded_population: List[Tuple[float, DiceboxNetwork]] = [(self.fitness(network), network) for network in population]
        graded_decompiled_population: List[Tuple[float, Any]] = [(self.fitness(network), network.decompile()) for network in population]

        # Sort on the scores.
        ranked_population: List[Any] = [x[1] for x in sorted(graded_decompiled_population, key=lambda x: x[0], reverse=True)]

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
                individual = self.mutate(individual)

        # Now find out how many spots we have left to fill.
        parents_length: int = len(parent_genomes)
        desired_length: int = len(population) - parents_length
        children: List[Any] = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male_index: int = random_index_between(0, parents_length - 1)
            female_index: int = random_index_between(0, parents_length - 1)

            # Assuming they aren't the same network...
            if male_index != female_index:
                male = parent_genomes[male_index]
                female = parent_genomes[female_index]

                # Breed them.
                babies: List[Any] = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parent_genomes.extend(children)

        parent_networks: List[Network] = [(self.create_network(genome)) for genome in parent_genomes]
        parents: List[DiceboxNetwork] = [(self.build_dicebox_network(network=network)) for network in parent_networks]
        return parents

    def build_dicebox_network(self, network: Network) -> DiceboxNetwork:
            network_config: NetworkConfig = self.create_network_config(network_definition=network.decompile())
            dicebox_network: DiceboxNetwork = DiceboxNetwork(config=self.config, network_config=network_config)
            return dicebox_network
