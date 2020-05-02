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

        self.train_data_loader = None
        self.optimizer = None

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
        if self.model is None:
            raise AttributeError("using `set_model()` to initialize `model`")
        if self.config is None:
            raise AttributeError("using `set_config()` to initialize `config`")
        if self.train_data is None or self.test_data is None:
            raise AttributeError("using `set_data()` to initialize `data`")

        torch_data = TensorDataset(torch.from_numpy(self.train_data))
        self.train_data_loader = DataLoader(
            dataset=torch_data,
            batch_size=self.config['batch_size'],
            shuffle=True,
        )

        tmp = self.config['optimizer']
        lr = self.config['base_lr']

        if tmp == 'Adam':
            logging.info('Using `Adam` optimizer')
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.config['lambda'])
        elif tmp == 'RMSProp':
            logging.info('Using `RMSProp` optimizer')
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            raise ValueError('Invalid choice of optimizer `{}`, optional: [`Adam`, `RMSProp`]'.format(tmp))

        logging.info('Trainer build success')

    def start(self):
        logging.info('Start Training...')

        for epoch in range(self.config['epochs']):
            self.train(epoch)
            self.test(epoch)

    def train(self, epoch):
        loss = torch.tensor([0.])  # accumulate loss
        train_items = len(np.nonzero(self.train_data)[0])

        for step, (batch_x) in enumerate(self.train_data_loader):
            x = batch_x[0].type(torch.FloatTensor)
            if self.config['cuda']:
                x = x.cuda()
            y = self.model(x)

            step_loss = ((y - x) * x.type(torch.bool)).pow(2).sum()

            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()

            loss += step_loss

        loss = loss.item() / train_items
        loss = np.sqrt(loss)
        logging.info('Epoch: {:02d} / {:02d}, Training Loss: {:.4f},'.format(epoch + 1, self.config['epochs'], loss))

    def test(self, epoch):
        pass
