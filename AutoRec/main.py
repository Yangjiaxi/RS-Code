import logging

import toml

from model.autorec import AutoRec
from movielens import MovieLensDataLoader
from trainer import Trainer
from utils import logger_setup

if __name__ == "__main__":
    config = toml.load("config.toml")

    logger_setup(config)

    data_loader = MovieLensDataLoader(config)
    train_data, test_data = data_loader.split()

    model = AutoRec(config)
    if config['cuda']:
        model.cuda()
        logging.info('CUDA enabled')
    else:
        logging.info('CUDA disabled')

    agent = Trainer()
    agent.set_model(model)
    agent.set_data(train_data, test_data)
    agent.set_config(config)
    agent.build()

    agent.start()
