import logging

import toml

from model.autorec import AutoRec
from movielens import MovieLensDataLoader
from trainer import Trainer
from utils import logger_setup

import torch
import numpy

if __name__ == "__main__":
    config = toml.load("config.toml")

    logger_setup(config)

    if 'seed' in config:
        seed = config['seed']
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        logging.info('Random seed: {}'.format(seed))
    else:
        seed = numpy.random.randint(1, 10000, (1,))[0]
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        logging.info('Random seed: {}'.format(seed))

    if config['dataset'] == 'ml-1m':
        config.update({
            'total_rating': 1000209,
            'num_users': 6040,
            'num_items': 3952
        })
    elif config['dataset'] == 'ml-100k':
        config.update({
            'total_rating': 100000,
            'num_users': 943,
            'num_items': 1682
        })
    else:
        raise ValueError("Invalid dataset: `{}`, optional: [`ml-1m`, `ml-100k`]".format(config['dataset']))

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
