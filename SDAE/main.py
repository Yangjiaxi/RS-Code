import logging

import numpy
import toml
import torch

from noised_mnist import NoisedMNISTHolder
from sdae import SDAE
from trainer import Trainer
from utils import logger_setup

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

    data_holder = NoisedMNISTHolder(config)
    data_holder.show_image('output/noised_MNIST.jpg')
    train_loader, test_data = data_holder.split()

    model = SDAE()
    if config['cuda']:
        model.cuda()
        logging.info('CUDA enabled')
    else:
        logging.info('CUDA disabled')

    # ------------ Start ------------
    agent = Trainer()

    agent.set_model(model)
    agent.set_data(train_loader, test_data)
    agent.set_config(config)

    agent.build()

    agent.start()
