"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score
from train import train_and_score_and_save
from train import load_and_score
from train import load_and_score_single

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_lonestar(self):
        # Lonestar /mnist approx 98.6% acc
        # {'nb_layers': 2, 'activation': 'relu', 'optimizer': 'adagrad', 'nb_neurons': 1597}
        self.network['nb_layers'] = 2
        self.network['activation'] = 'relu'
        self.network['optimizer'] = 'adagrad'
        self.network['nb_neurons'] = 1597

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.network, dataset)

    def train_and_save(self, dataset):
        if self.accuracy == 0.:
            self.accuracy = train_and_score_and_save(self.network, dataset)

    def load_n_score(self, dataset):
        self.accuracy = load_and_score(self.network, dataset)

    def load_n_score_single(self, dataset):
        self.accuracy = load_and_score_single(self.network, dataset)

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
