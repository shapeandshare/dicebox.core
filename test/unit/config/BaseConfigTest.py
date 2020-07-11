import json
import unittest
from src.config.base_config import BaseConfig


class BaseConfigTest(unittest.TestCase):
    fixtures_base = 'test/fixtures'
    local_config_file = '%s/dicebox.config' % fixtures_base

    def setUp(self):
        self.dc = BaseConfig(config_file=self.local_config_file)

    def test_config(self):
        self.assertEqual(self.dc.DATASET, 'mnist_training')
        self.assertEqual(self.dc.DICEBOX_COMPLIANT_DATASET, True)
        self.assertEqual(self.dc.IMAGE_WIDTH, 28)
        self.assertEqual(self.dc.IMAGE_HEIGHT, 28)
        self.assertEqual(self.dc.NB_CLASSES, 10)
        self.assertEqual(self.dc.DATA_BASE_DIRECTORY, '/dicebox/datasets')

        # composite
        self.assertEqual(self.dc.NETWORK_NAME, 'mnist_training_28x28')

        # composite
        self.assertEqual(self.dc.INPUT_SHAPE, (784,))

        # composite
        self.assertEqual(self.dc.DATA_DIRECTORY, '/dicebox/datasets/mnist_training_28x28/data/')

        self.assertEqual(self.dc.MIN_NEURONS, 1)
        self.assertEqual(self.dc.MAX_NEURONS, 1597)
        self.assertEqual(self.dc.MIN_LAYERS, 1)
        self.assertEqual(self.dc.MAX_LAYERS, 21)
        self.assertEqual(self.dc.LAYER_TYPES, '["dropout", "dense"]')
        self.assertEqual(self.dc.ACTIVATION, '["softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]')
        self.assertEqual(self.dc.OPTIMIZER, '["rmsprop", "adam", "sgd", "adagrad", "adadelta", "adamax", "nadam"]')

        # composite ..
        self.assertEqual(self.dc.TAXONOMY, {
            'min_neurons': 1,
            'max_neurons': 1597,
            'min_layers': 1,
            'max_layers': 21,
            'layer_types': json.loads('["dropout", "dense"]'),
            'activation': json.loads(
                '["softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]'),
            'optimizer': json.loads('["rmsprop", "adam", "sgd", "adagrad", "adadelta", "adamax", "nadam"]')
        })

        self.assertEqual(self.dc.EPOCHS, 10000)
        self.assertEqual(self.dc.GENERATIONS, 100)
        self.assertEqual(self.dc.POPULATION, 50)
        self.assertEqual(self.dc.NOISE, 0.0)
        self.assertEqual(self.dc.BATCH_SIZE, 100)
        self.assertEqual(self.dc.TRAIN_BATCH_SIZE, 1000)
        self.assertEqual(self.dc.TEST_BATCH_SIZE, 100)
        self.assertEqual(self.dc.LOAD_BEST_WEIGHTS_ON_START, False)
        self.assertEqual(self.dc.LOGS_DIR, '/dicebox/logs')
        self.assertEqual(self.dc.WEIGHTS_DIR, '/dicebox/weights')
        self.assertEqual(self.dc.TMP_DIR, '/tmp')
        self.assertEqual(self.dc.API_ACCESS_KEY, '6e249b5f-b483-4e0d-b50b-81d95e3d9a59')
        self.assertEqual(self.dc.API_VERSION, '0.3.0')
        self.assertEqual(self.dc.LISTENING_HOST, '0.0.0.0')
        self.assertEqual(self.dc.FLASK_DEBUG, False)
        self.assertEqual(self.dc.MODEL_WEIGHTS_FILENAME, 'weights.best.hdf5')
        self.assertEqual(self.dc.SENSORY_SERVER, 'localhost')
        self.assertEqual(self.dc.SENSORY_PORT, 443)
        self.assertEqual(self.dc.SENSORY_URI, 'https://')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_EXCHANGE, 'sensory.exchange')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY, 'task_queue')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE, 'sensory.batch.request.task.queue')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_URI, 'amqps://')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_USERNAME, 'sensory_service')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_PASSWORD, 'sensory_service!123')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_SERVER, 'localhost')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_PORT, 5671)
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_VHOST, 'sensory')

        # composite
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_URL,
                         'amqps://sensory_service:sensory_service!123@localhost:5671/sensory')

        self.assertEqual(self.dc.SENSORY_SERVICE_SHARD_SIZE, 5000)
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_EXCHANGE, 'training.exchange')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY, 'task_queue')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE, 'train.request.task.queue')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI, 'amqps://')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_USERNAME, 'training_service')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_PASSWORD, 'training_service!123')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_SERVER, 'localhost')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_PORT, 5671)
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_VHOST, 'training')

        # composite
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_URL,
                         'amqps://training_service:training_service!123@localhost:5671/training')

        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE, 'training.exchange')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY, 'task_queue')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE,
                         'train.request.task.queue')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_URI, 'amqps://')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME, 'training_processor_service')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD, 'training_processor_service!123')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER, 'localhost')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT, 5671)
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST, 'training')

        # composite
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_URL,
                         'amqps://training_processor_service:training_processor_service!123@localhost:5671/training')

        self.assertEqual(self.dc.CLASSIFICATION_SERVER, 'localhost')
        self.assertEqual(self.dc.CLASSIFICATION_SERVER_PORT, 5000)
        self.assertEqual(self.dc.CLASSIFICATION_SERVER_URI, 'https://')


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(BaseConfigTest())
