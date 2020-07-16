import logging
import unittest

from src.shapeandshare.dicebox.primordialpool import PrimordialPool
from src.shapeandshare.dicebox.config.dicebox_config import DiceboxConfig

###############################################################################
# Setup logging.
###############################################################################
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filemode='w',
    filename="primordialpool.log"
)


class PrimordialPoolTest(unittest.TestCase):

    def test_pool(self):
        config_file = 'test/fixtures/mnist/dicebox.config'
        dicebox_config: DiceboxConfig = DiceboxConfig(config_file=config_file)

        logging.info("***Evolving %d generations with population %d***" % (dicebox_config.GENERATIONS, dicebox_config.POPULATION))
        pool_one: PrimordialPool = PrimordialPool(config=dicebox_config)
        population_id_one = pool_one.generate()
        del pool_one

        # perform a re-evolve using the last serialized population
        pool_two: PrimordialPool = PrimordialPool(config=dicebox_config)
        population_id_two = pool_two.generate(population_file="%s/%s/2/population.json" % (dicebox_config.POPULATION_DIR, population_id_one))
        print(population_id_two)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(PrimordialPoolTest())
