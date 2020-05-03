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
        self.train_size = None
        self.test_size = None

        self.unseen_test_user = None
        self.unseen_test_item = None

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

        self.train_size = len(np.nonzero(self.train_data)[0])
        self.test_size = len(np.nonzero(self.test_data)[0])

        item_train_set = set(np.sum(self.train_data, axis=0).nonzero()[0])
        item_test_set = set(np.sum(self.test_data, axis=0).nonzero()[0])
        self.unseen_test_item = list(item_test_set - item_train_set)

        user_train_set = set(np.sum(self.train_data, axis=1).nonzero()[0])
        user_test_set = set(np.sum(self.test_data, axis=1).nonzero()[0])
        self.unseen_test_user = list(user_test_set - user_train_set)

        torch_data = TensorDataset(torch.from_numpy(self.train_data))
        self.train_data_loader = DataLoader(
            dataset=torch_data,
            batch_size=self.config['batch_size'],
            shuffle=True,
        )

        tmp = self.config['optimizer']
        lr = self.config['base_lr']

        if tmp == 'SGD':
            logging.info('Using `SGD` optimizer')
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.config['lambda'])
        elif tmp == 'Adam':
            logging.info('Using `Adam` optimizer')
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.config['lambda'])
        elif tmp == 'RMSProp':
            logging.info('Using `RMSProp` optimizer')
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=self.config['lambda'])
        else:
            raise ValueError('Invalid choice of optimizer `{}`, optional: [`Adam`, `RMSProp`]'.format(tmp))

        logging.info('Trainer build success')

    def start(self):
        logging.info('Start Training...')

        epochs = self.config['epochs']
        for epoch in range(epochs):
            train_rmse = self.train()
            test_rmse = self.test()
            logging.info(
                'Epoch: {:02d} / {:02d}, Train Loss: {:.4f}, Test RMSE: {:.4f}'.format(epoch + 1, epochs, train_rmse,
                                                                                       test_rmse))

    def train(self):
        loss = torch.tensor([0.])  # accumulate loss

        for step, batch_x in enumerate(self.train_data_loader):
            x = batch_x[0].type(torch.FloatTensor)
            if self.config['cuda']:
                x = x.cuda()
            y = self.model(x)

            step_loss = ((y - x) * x.type(torch.bool)).pow(2).sum()

            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()

            loss += step_loss

        loss = loss.item() / self.train_size
        loss = np.sqrt(loss)
        return loss

    def test(self):
        test_data_torch = torch.from_numpy(self.test_data).type(torch.FloatTensor)
        if self.config['cuda']:
            test_data_torch = test_data_torch.cuda()
        # num_users * num_items
        output = self.model(test_data_torch)  # num_users * num_items

        for user in self.unseen_test_user:
            for item in self.unseen_test_item:
                if self.test_data[user, item] != 0:
                    output[user, item] = 3  # unseen (user, item) pair set to 3, an average of taste

        mse = ((output - test_data_torch) * test_data_torch.type(torch.bool)).pow(2).sum()
        RMSE = mse.detach().cpu().numpy() / self.test_size
        RMSE = np.sqrt(RMSE)
        return RMSE
