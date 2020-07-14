import os
import unittest
from src.shapeandshare.dicebox.config.dicebox_config import DiceboxConfig
from dotenv import load_dotenv
import json
from pathlib import Path

class DiceboxConfigTest(unittest.TestCase):
    fixtures_base = 'test/fixtures'
    local_config_file = '%s/dicebox.env.override.test.config' % fixtures_base

    def setUp(self):
        self.maxDiff=None


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


    def test_config_over_one(self):
        env_path = Path("%s/.one.env" % self.fixtures_base)
        load_dotenv(env_path)
        dc: DiceboxConfig = DiceboxConfig(config_file=self.local_config_file)

        self.assertEqual(dc.DATASET, 'DATASET')
        self.assertEqual(dc.DICEBOX_COMPLIANT_DATASET, True)
        self.assertEqual(dc.IMAGE_WIDTH, 200)
        self.assertEqual(dc.IMAGE_HEIGHT, 300)
        self.assertEqual(dc.NB_CLASSES, 100)
        self.assertEqual(dc.DATA_BASE_DIRECTORY, '/some/datasets')

        # composite
        self.assertEqual(dc.NETWORK_NAME, 'DATASET_200x300')

        # composite
        self.assertEqual(dc.INPUT_SHAPE, (60000,))

        # composite
        self.assertEqual(dc.DATA_DIRECTORY, '/some/datasets/DATASET_200x300/data/')

        self.assertEqual(dc.MIN_NEURONS, 2)
        self.assertEqual(dc.MAX_NEURONS, 1598)
        self.assertEqual(dc.MIN_LAYERS, 3)
        self.assertEqual(dc.MAX_LAYERS, 22)
        self.assertEqual(dc.LAYER_TYPES, '["some", "type"]')
        self.assertEqual(dc.ACTIVATION, '["low", "medium", "high"]')
        self.assertEqual(dc.OPTIMIZER, '["option", "someotheroption", "differentoption"]')

        # composite ..
        self.assertEqual(dc.TAXONOMY, {
            'min_neurons': 2,
            'max_neurons': 1598,
            'min_layers': 3,
            'max_layers': 22,
            'layer_types': json.loads('["some", "type"]'),
            'activation': json.loads('["low", "medium", "high"]'),
            'optimizer': json.loads('["option", "someotheroption", "differentoption"]')
        })

        self.assertEqual(dc.EPOCHS, 27)
        self.assertEqual(dc.GENERATIONS, 8)
        self.assertEqual(dc.POPULATION, 78)
        self.assertEqual(dc.NOISE, 1.0)
        self.assertEqual(dc.BATCH_SIZE, 120)
        self.assertEqual(dc.TRAIN_BATCH_SIZE, 1002)
        self.assertEqual(dc.TEST_BATCH_SIZE, 101)
        self.assertEqual(dc.LOAD_BEST_WEIGHTS_ON_START, False)
        self.assertEqual(dc.LOGS_DIR, '/logs')
        self.assertEqual(dc.WEIGHTS_DIR, '/weights')
        self.assertEqual(dc.TMP_DIR, '/some/tmp')
        self.assertEqual(dc.API_ACCESS_KEY, '01234567890')
        self.assertEqual(dc.API_VERSION, '0.3.1')
        self.assertEqual(dc.LISTENING_HOST, '0.0.0.1')
        self.assertEqual(dc.FLASK_DEBUG, False)
        self.assertEqual(dc.MODEL_WEIGHTS_FILENAME, 'weights.1.best.hdf5')
        self.assertEqual(dc.SENSORY_SERVER, 'localhost')
        self.assertEqual(dc.SENSORY_PORT, 441)
        self.assertEqual(dc.SENSORY_URI, 'https://')
        self.assertEqual(dc.SENSORY_SERVICE_RABBITMQ_EXCHANGE, 'sensory.exchange1')
        self.assertEqual(dc.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY, 'task_queue1')
        self.assertEqual(dc.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE, 'sensory.batch.request.task.queue1')
        self.assertEqual(dc.SENSORY_SERVICE_RABBITMQ_URI, 'amqpz://')
        self.assertEqual(dc.SENSORY_SERVICE_RABBITMQ_USERNAME, 'sensory_service1')
        self.assertEqual(dc.SENSORY_SERVICE_RABBITMQ_PASSWORD, 'sensory_service!1231')
        self.assertEqual(dc.SENSORY_SERVICE_RABBITMQ_SERVER, 'localhost1')
        self.assertEqual(dc.SENSORY_SERVICE_RABBITMQ_PORT, 56711)
        self.assertEqual(dc.SENSORY_SERVICE_RABBITMQ_VHOST, 'sensory1')

        # composite
        self.assertEqual(dc.SENSORY_SERVICE_RABBITMQ_URL,
                         'amqpz://sensory_service1:sensory_service!1231@localhost1:56711/sensory1')

        self.assertEqual(dc.SENSORY_SERVICE_SHARD_SIZE, 51)
        self.assertEqual(dc.TRAINING_SERVICE_RABBITMQ_EXCHANGE, 'training.exchange')
        self.assertEqual(dc.TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY, 'task_queue')
        self.assertEqual(dc.TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE, 'train.request.task.queue')
        self.assertEqual(dc.TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI, 'amqpz://')
        self.assertEqual(dc.TRAINING_SERVICE_RABBITMQ_USERNAME, 'training_service1')
        self.assertEqual(dc.TRAINING_SERVICE_RABBITMQ_PASSWORD, 'training_service!1231')
        self.assertEqual(dc.TRAINING_SERVICE_RABBITMQ_SERVER, 'localhost1')
        self.assertEqual(dc.TRAINING_SERVICE_RABBITMQ_PORT, 56711)
        self.assertEqual(dc.TRAINING_SERVICE_RABBITMQ_VHOST, 'training')

        # composite
        self.assertEqual(dc.TRAINING_SERVICE_RABBITMQ_URL,
                         'amqpz://training_service1:training_service!1231@localhost1:56711/training')

        self.assertEqual(dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE, 'training.exchange2')
        self.assertEqual(dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY, 'task_queue2')
        self.assertEqual(dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE,
                         'train.request.task.queue2')
        self.assertEqual(dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_URI, 'amqpqq://')
        self.assertEqual(dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME, 'training_processor_service2')
        self.assertEqual(dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD, 'training_processor_service!1232')
        self.assertEqual(dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER, 'localhost2')
        self.assertEqual(dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT, 56712)
        self.assertEqual(dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST, 'training2')

        # composite
        self.assertEqual(dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_URL,
                         'amqpqq://training_processor_service2:training_processor_service!1232@localhost2:56712/training2')

        self.assertEqual(dc.CLASSIFICATION_SERVER, 'localhost4')
        self.assertEqual(dc.CLASSIFICATION_SERVER_PORT, 50004)
        self.assertEqual(dc.CLASSIFICATION_SERVER_URI, 'http://')


    def test_config_over_two(self):
        env_path = Path("%s/.one.env" % self.fixtures_base)
        load_dotenv(env_path)
        os.environ["DICEBOX_COMPLIANT_DATASET"] = "False"
        os.environ["LOAD_BEST_WEIGHTS_ON_START"] = "True"
        os.environ["FLASK_DEBUG"] = "True"
        dc: DiceboxConfig = DiceboxConfig(config_file=self.local_config_file)
        self.assertEqual(dc.DICEBOX_COMPLIANT_DATASET, False)
        self.assertEqual(dc.LOAD_BEST_WEIGHTS_ON_START, True)
        self.assertEqual(dc.FLASK_DEBUG, True)

    def test_config_over_three(self):
        env_path = Path("%s/.one.env" % self.fixtures_base)
        load_dotenv(env_path)
        os.environ['DICEBOX_COMPLIANT_DATASET'] = 'BLAH'
        try:
            dc: DiceboxConfig = DiceboxConfig(config_file=self.local_config_file)
            self.assertFalse(True, 'Exception should have been thrown')
        except Exception:
            self.assertTrue(True)

    def test_config_over_four(self):
        env_path = Path("%s/.one.env" % self.fixtures_base)
        load_dotenv(env_path)
        os.environ['LOAD_BEST_WEIGHTS_ON_START'] = 'BLAH'
        try:
            dc: DiceboxConfig = DiceboxConfig(config_file=self.local_config_file)
            self.assertFalse(True, 'Exception should have been thrown')
        except Exception:
            self.assertTrue(True)

    def test_config_over_five(self):
        env_path = Path("%s/.one.env" % self.fixtures_base)
        load_dotenv(env_path)
        os.environ['FLASK_DEBUG'] = 'BLAH'
        try:
            dc: DiceboxConfig = DiceboxConfig(config_file=self.local_config_file)
            self.assertFalse(True, 'Exception should have been thrown')
        except Exception:
            self.assertTrue(True)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(DiceboxConfigTest())
