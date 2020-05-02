import torch


def RMSELoss(output, target):
    return torch.sqrt(torch.mean(output - target) ** 2)
