import logging
from random import randint

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from utils import AddGaussianNoise, AddSpeckleNoise


class NoisedMNIST(MNIST):
    def __init__(self, root, train, noise_fn):
        super(NoisedMNIST, self).__init__(root=root,
                                          download=True,
                                          train=train,
                                          transform=Compose([ToTensor()]))
        self.noise_fn = noise_fn

    def __getitem__(self, index):
        clean_img, target = super(NoisedMNIST, self).__getitem__(index)
        noise_img = self.noise_fn(clean_img)
        return noise_img, clean_img, target


class NoisedMNISTHolder:
    def __init__(self, config):
        self.config = config

        n_type = config['noise_type']
        if n_type == 'gaussian':
            noise_generator = AddGaussianNoise(config)
        elif n_type == 'speckle':
            noise_generator = AddSpeckleNoise(config)
        else:
            raise ValueError('Invalid noise type `{}`, optional: [`gaussian`, `speckle`]'.format(n_type))

        logging.info("Using `{}` noise generator.".format(n_type))

        self.train_data = NoisedMNIST(root=config['data_root'], train=True, noise_fn=noise_generator)
        self.test_data = NoisedMNIST(root=config['data_root'], train=False, noise_fn=noise_generator)

        logging.info('Data set `MNIST` loaded.')

    def split(self):
        train_loader = DataLoader(self.train_data, batch_size=self.config['batch_size'],
                                  shuffle=True, num_workers=2)
        test_data = self.test_data
        return train_loader, test_data

    def show_image(self, output_dir=None):
        idx0 = randint(0, len(self.train_data))
        idx1 = randint(0, len(self.train_data))
        post0, pre0, _ = self.train_data[idx0]
        post1, pre1, _ = self.train_data[idx1]

        f, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(pre0.squeeze(), cmap="gray")
        axes[1, 0].imshow(post0.squeeze(), cmap='gray')
        axes[0, 1].imshow(pre1.squeeze(), cmap='gray')
        axes[1, 1].imshow(post1.squeeze(), cmap="gray")

        if output_dir is not None:
            plt.savefig(output_dir)

        plt.show()
