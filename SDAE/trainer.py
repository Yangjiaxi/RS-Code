import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import optim


class Trainer:
    def __init__(self):
        self.model = None
        self.config = None
        self.train_loader = None
        self.test_data = None

        self.optimizer = None
        self.criterion = None

    def set_model(self, model):
        self.model = model
        return self

    def set_config(self, config):
        self.config = config
        return self

    def set_data(self, train_loader, test_data):
        self.train_loader = train_loader
        self.test_data = test_data
        pass

    def build(self):
        if self.model is None:
            raise AttributeError("using `set_model()` to initialize `model`")
        if self.config is None:
            raise AttributeError("using `set_config()` to initialize `config`")
        if self.train_loader is None or self.test_data is None:
            raise AttributeError("using `set_data()` to initialize `data`")

        self.criterion = nn.MSELoss()

        tmp = self.config["optimizer"]
        lr = self.config["base_lr"]
        wd = self.config['weight_decay']

        if tmp == "SGD":
            logging.info("Using `SGD` optimizer")
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        elif tmp == "Adam":
            logging.info("Using `Adam` optimizer")
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(0.5, 0.999)
            )
        else:
            raise ValueError(
                "Invalid choice of optimizer `{}`, "
                "optional: [`Adam`, `SGD`]".format(tmp)
            )

        logging.info("Trainer build success")

    def start(self):
        logging.info("Start Training...")

        epochs = self.config["epochs"]
        for epoch in range(epochs):
            train_loss = self.train()
            self.test(epoch)
            logging.info("Epoch: {}, Train Loss: {}".format(epoch, train_loss))

    def train(self):
        loss_all = torch.tensor([0.])
        for idx, (noise_img, clean_img, label) in enumerate(self.train_loader):
            noise_torch = noise_img.view(noise_img.size(0), -1).type(torch.FloatTensor)
            clean_torch = clean_img.view(clean_img.size(0), -1).type(torch.FloatTensor)

            if self.config['cuda']:
                noise_torch = noise_torch.cuda()
                clean_torch = clean_torch.cuda()

            output = self.model(noise_torch)
            loss = self.criterion(output, clean_torch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_all += loss.item()

        return loss_all.item()

    def test(self, epoch):
        f, axes = plt.subplots(6, 3, figsize=(20, 20))
        axes[0, 0].set_title("Original Image")
        axes[0, 1].set_title("Dirty Image")
        axes[0, 2].set_title("Cleaned Image")

        test_imgs = np.random.randint(0, len(self.test_data), size=6)

        for idx in range(6):
            noise_img, clean_img, label = self.test_data[test_imgs[idx]]
            noise_torch = noise_img.view(noise_img.size(0), -1).type(torch.FloatTensor)
            if self.config['cuda']:
                noise_torch = noise_torch.cuda()

            output = self.model(noise_torch)
            output = output.view(28, 28).detach().cpu().numpy()

            noise_img = noise_img.view(28, 28).detach().cpu().numpy()
            clean_img = clean_img.view(28, 28).detach().cpu().numpy()

            axes[idx, 0].imshow(clean_img, cmap="gray")
            axes[idx, 1].imshow(noise_img, cmap="gray")
            axes[idx, 2].imshow(output, cmap="gray")

        plt.savefig('output/{}-Test.png'.format(epoch))
        plt.show()
