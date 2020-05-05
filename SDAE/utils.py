import logging
from os import path

import torch


def logger_setup(config: dict):
    dir_root = config['log_root']
    log_name = config['log_name']
    full_path = path.join(dir_root, log_name)
    if config['append_time']:
        from time import strftime, localtime
        full_path += strftime("-%m-%d|%H:%M:%S", localtime())
    full_path += ".log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s| %(message)s', '%m-%d|%H:%M:%S')

    file_hdl = logging.FileHandler(full_path)
    file_hdl.setFormatter(formatter)

    root_logger.addHandler(file_hdl)

    if config['console_output']:
        console_hdl = logging.StreamHandler()
        console_hdl.setFormatter(formatter)
        root_logger.addHandler(console_hdl)


class AddGaussianNoise(object):
    def __init__(self, config):
        self.std = config['gaussian_std']
        self.mean = config['gaussian_mean']

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class AddSpeckleNoise(object):
    def __init__(self, config):
        self.std = config['speckle_std']
        self.mean = config['speckle_mean']

    def __call__(self, tensor):
        return tensor + tensor * torch.randn(tensor.size()) * self.std + self.mean
