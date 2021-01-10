from shapeandshare.dicebox import PrimordialPool
from shapeandshare.dicebox.config.dicebox_config import DiceboxConfig
from shapeandshare.dicebox.utils.helpers import make_sure_path_exists
import logging
import os

VERSION = '0.6.0'


def main():
    # config_file = 'dicebox.config'
    config_file = './projects/mnist/dicebox.config'
    dicebox_config: DiceboxConfig = DiceboxConfig(config_file=config_file)

    ###############################################################################
    # Setup logging.
    ###############################################################################
    make_sure_path_exists(dicebox_config.LOGS_DIR)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
        filemode='w',
        filename="%s/primordialpool.%s.log" % (dicebox_config.LOGS_DIR, os.uname()[1])
    )

    logging.info("Application Version (%s), Dicebox API Version: (%s)", VERSION, dicebox_config.API_VERSION)
    logging.info(
        "***Evolving %d generations with population %d***" % (dicebox_config.GENERATIONS, dicebox_config.POPULATION))
    pool: PrimordialPool = PrimordialPool(config=dicebox_config)
    pool.generate()


if __name__ == '__main__':
    main()
