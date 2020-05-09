import logging

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np


class Trainer:
    def __init__(self):
        self.model = None
        self.config = None
        self.train_data = None
        self.test_data = None

    def set_model(self, model):
        self.model = model
        return self

    def set_data(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        return self

    def set_config(self, config):
        self.config = config
        return self

    def build(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
