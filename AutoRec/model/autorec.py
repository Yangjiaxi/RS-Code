import torch.nn as nn
import torch


class AutoRec(nn.Module):
    def __init__(self, config: dict):
        super(AutoRec, self).__init__()

        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.hidden_units = config['hidden_units']
        self.lambda_v = config['lambda']

        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, self.hidden_units),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_units, self.num_items)
        )

    def forward(self, x):
        # => batch_size * num_items
        x = self.encoder(x)  # batch_size * hidden_units
        x = self.decoder(x)  # batch_size * num_items
        return x
