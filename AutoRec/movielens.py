import logging
import numpy as np
from os import path

import torch
from torch.utils.data import Dataset, TensorDataset, random_split


class MovieLensDataLoader:
    def __init__(self, config):
        # self.config = config
        self.root_path = config['root_path']
        self.num_users = config['num_users']
        self.num_items = config['num_items']

        self.total_rating = config['total_rating']
        self.train_ratio = config['train_ratio']

        self.data = None

        if config['dataset'] == 'ml-1m':
            self.load_1m()
        elif config['dataset'] == 'ml-100k':
            self.load_100k()

    def split(self):
        train_size = int(self.total_rating * self.train_ratio)

        mask = self.data != 0

        mask_idx = np.transpose(np.nonzero(mask))  # get non-zero element's [(x,y)]

        perm_mask_idx = np.random.permutation(mask_idx)  # randomly select (x,y)
        train_idx = perm_mask_idx[:train_size]

        x, y = np.transpose(train_idx)  # [(x, y)] => [x], [y]
        train_mask = np.zeros_like(self.data).astype(np.bool)
        train_mask[x, y] = True

        train_data = self.data * train_mask
        test_data = self.data * np.bitwise_not(train_mask)

        return train_data, test_data

    def load_1m(self):
        logging.info('Using dataset `ml-1m`')

        data_file = path.join(self.root_path, 'ml-1m', 'ratings.dat')

        '''
            input contains `self.total_rating` lines
            each line is made up of 4 parts and separate with `::` 
            i.e. `UserID::MovieID::Rating::Timestamp`
            `Timestamp` will not be used here
            
            1. we read all lines and construct a Dataset holds all
            2. use `torch.utils.data.random_split` to get two datasets
            
            at each step, loader gives a mini-batch which seems like (batch_size * num_items), 
            network should output like (batch_size * num_items), which means we predicate `A` using `A`.
            
            By doing so, we may find correct arguments in the network, 
            which may generalize to unseen (user, item) pairs.
        '''

        self.data = np.zeros((self.num_users, self.num_items))

        with open(data_file) as f:
            lines = f.readlines()

            if not len(lines) == self.total_rating:
                logging.warning(
                    "CAUTION: `total_rating` parameter in config file is {}, "
                    "but actually {}".format(self.total_rating, len(lines)))

            for line in lines:
                user, item, rating, _ = line.split("::")
                # user / item number is start from 1
                # BUT !
                # YOU KNOW !
                user_idx = int(user) - 1
                item_idx = int(item) - 1
                self.data[user_idx, item_idx] = int(rating)

    def load_100k(self):
        logging.info('Using dataset `ml-100k`')

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()
