import os
import unittest
from src.config.dicebox_config import DiceboxConfig
from dotenv import load_dotenv
import json

class DiceboxConfigTest(unittest.TestCase):
    fixtures_base = 'test/fixtures'
    local_config_file = '%s/dicebox.env.override.test.config' % fixtures_base

    def setUp(self):
        self.maxDiff=None
        load_dotenv(verbose=True)
        self.dc = DiceboxConfig(config_file=self.local_config_file)

    def tearDown(self) -> None:
        # unset all those env vars...
        del os.environ['DATASET']
        del os.environ['DICEBOX_COMPLIANT_DATASET']
        del os.environ['NB_CLASSES']
        del os.environ['IMAGE_WIDTH']
        del os.environ['IMAGE_HEIGHT']
        del os.environ['DATA_BASE_DIRECTORY']
        del os.environ['MIN_NEURONS']
        del os.environ['MAX_NEURONS']
        del os.environ['MIN_LAYERS']
        del os.environ['MAX_LAYERS']
        del os.environ['LAYER_TYPES']
        del os.environ['ACTIVATION']
        del os.environ['OPTIMIZER']
        del os.environ['EPOCHS']
        del os.environ['GENERATIONS']
        del os.environ['POPULATION']
        del os.environ['BATCH_SIZE']
        del os.environ['TRAIN_BATCH_SIZE']
        del os.environ['TEST_BATCH_SIZE']
        del os.environ['LOAD_BEST_WEIGHTS_ON_START']
        del os.environ['LOGS_DIR']
        del os.environ['WEIGHTS_DIR']
        del os.environ['TMP_DIR']
        del os.environ['API_ACCESS_KEY']
        del os.environ['API_VERSION']
        del os.environ['LISTENING_HOST']
        del os.environ['FLASK_DEBUG']
        del os.environ['MODEL_WEIGHTS_FILENAME']
        del os.environ['SENSORY_SERVICE_RABBITMQ_URI']
        del os.environ['SENSORY_URI']
        del os.environ['SENSORY_SERVER']
        del os.environ['SENSORY_PORT']
        del os.environ['SENSORY_SERVICE_RABBITMQ_EXCHANGE']
        del os.environ['SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY']
        del os.environ['SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE']
        del os.environ['SENSORY_SERVICE_RABBITMQ_USERNAME']
        del os.environ['SENSORY_SERVICE_RABBITMQ_PASSWORD']
        del os.environ['SENSORY_SERVICE_RABBITMQ_SERVER']
        del os.environ['SENSORY_SERVICE_RABBITMQ_PORT']
        del os.environ['SENSORY_SERVICE_RABBITMQ_VHOST']
        del os.environ['SENSORY_SERVICE_SHARD_SIZE']
        del os.environ['TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI']
        del os.environ['TRAINING_SERVICE_RABBITMQ_PORT']
        del os.environ['TRAINING_SERVICE_RABBITMQ_SERVER']
        del os.environ['TRAINING_SERVICE_RABBITMQ_USERNAME']
        del os.environ['TRAINING_SERVICE_RABBITMQ_PASSWORD']
        del os.environ['TRAINING_SERVICE_RABBITMQ_VHOST']
        del os.environ['TRAINING_SERVICE_RABBITMQ_EXCHANGE']
        del os.environ['TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY']
        del os.environ['TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE']
        del os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_URI']
        del os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT']
        del os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER']
        del os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME']
        del os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD']
        del os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST']
        del os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE']
        del os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY']
        del os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE']
        del os.environ['NOISE']
        del os.environ['CLASSIFICATION_SERVER']
        del os.environ['CLASSIFICATION_SERVER_PORT']
        del os.environ['CLASSIFICATION_SERVER_URI']


    def test_config(self):
        self.assertEqual(self.dc.DATASET, 'DATASET')
        self.assertEqual(self.dc.DICEBOX_COMPLIANT_DATASET, True)
        self.assertEqual(self.dc.IMAGE_WIDTH, 200)
        self.assertEqual(self.dc.IMAGE_HEIGHT, 300)
        self.assertEqual(self.dc.NB_CLASSES, 100)
        self.assertEqual(self.dc.DATA_BASE_DIRECTORY, '/some/datasets')

        # composite
        self.assertEqual(self.dc.NETWORK_NAME, 'DATASET_200x300')

        # composite
        self.assertEqual(self.dc.INPUT_SHAPE, (60000,))

        # composite
        self.assertEqual(self.dc.DATA_DIRECTORY, '/some/datasets/DATASET_200x300/data/')

        self.assertEqual(self.dc.MIN_NEURONS, 2)
        self.assertEqual(self.dc.MAX_NEURONS, 1598)
        self.assertEqual(self.dc.MIN_LAYERS, 3)
        self.assertEqual(self.dc.MAX_LAYERS, 22)
        self.assertEqual(self.dc.LAYER_TYPES, '["some", "type"]')
        self.assertEqual(self.dc.ACTIVATION, '["low", "medium", "high"]')
        self.assertEqual(self.dc.OPTIMIZER, '["option", "someotheroption", "differentoption"]')

        # composite ..
        self.assertEqual(self.dc.TAXONOMY, {
            'min_neurons': 2,
            'max_neurons': 1598,
            'min_layers': 3,
            'max_layers': 22,
            'layer_types': json.loads('["some", "type"]'),
            'activation': json.loads('["low", "medium", "high"]'),
            'optimizer': json.loads('["option", "someotheroption", "differentoption"]')
        })

        self.assertEqual(self.dc.EPOCHS, 27)
        self.assertEqual(self.dc.GENERATIONS, 8)
        self.assertEqual(self.dc.POPULATION, 78)
        self.assertEqual(self.dc.NOISE, 1.0)
        self.assertEqual(self.dc.BATCH_SIZE, 120)
        self.assertEqual(self.dc.TRAIN_BATCH_SIZE, 1002)
        self.assertEqual(self.dc.TEST_BATCH_SIZE, 101)
        self.assertEqual(self.dc.LOAD_BEST_WEIGHTS_ON_START, False)
        self.assertEqual(self.dc.LOGS_DIR, '/logs')
        self.assertEqual(self.dc.WEIGHTS_DIR, '/weights')
        self.assertEqual(self.dc.TMP_DIR, '/some/tmp')
        self.assertEqual(self.dc.API_ACCESS_KEY, '01234567890')
        self.assertEqual(self.dc.API_VERSION, '0.3.1')
        self.assertEqual(self.dc.LISTENING_HOST, '0.0.0.1')
        self.assertEqual(self.dc.FLASK_DEBUG, False)
        self.assertEqual(self.dc.MODEL_WEIGHTS_FILENAME, 'weights.1.best.hdf5')
        self.assertEqual(self.dc.SENSORY_SERVER, 'localhost')
        self.assertEqual(self.dc.SENSORY_PORT, 441)
        self.assertEqual(self.dc.SENSORY_URI, 'https://')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_EXCHANGE, 'sensory.exchange1')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY, 'task_queue1')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE, 'sensory.batch.request.task.queue1')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_URI, 'amqpz://')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_USERNAME, 'sensory_service1')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_PASSWORD, 'sensory_service!1231')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_SERVER, 'localhost1')
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_PORT, 56711)
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_VHOST, 'sensory1')

        # composite
        self.assertEqual(self.dc.SENSORY_SERVICE_RABBITMQ_URL,
                         'amqpz://sensory_service1:sensory_service!1231@localhost1:56711/sensory1')

        self.assertEqual(self.dc.SENSORY_SERVICE_SHARD_SIZE, 51)
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_EXCHANGE, 'training.exchange')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY, 'task_queue')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE, 'train.request.task.queue')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI, 'amqpz://')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_USERNAME, 'training_service1')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_PASSWORD, 'training_service!1231')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_SERVER, 'localhost1')
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_PORT, 56711)
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_VHOST, 'training')

        # composite
        self.assertEqual(self.dc.TRAINING_SERVICE_RABBITMQ_URL,
                         'amqpz://training_service1:training_service!1231@localhost1:56711/training')

        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE, 'training.exchange2')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY, 'task_queue2')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE,
                         'train.request.task.queue2')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_URI, 'amqpqq://')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME, 'training_processor_service2')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD, 'training_processor_service!1232')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER, 'localhost2')
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT, 56712)
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST, 'training2')

        # composite
        self.assertEqual(self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_URL,
                         'amqpqq://training_processor_service2:training_processor_service!1232@localhost2:56712/training2')

        self.assertEqual(self.dc.CLASSIFICATION_SERVER, 'localhost4')
        self.assertEqual(self.dc.CLASSIFICATION_SERVER_PORT, 50004)
        self.assertEqual(self.dc.CLASSIFICATION_SERVER_URI, 'http://')


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(DiceboxConfigTest())
