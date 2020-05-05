from torch import nn


class SDAE(nn.Module):
    def __init__(self):
        super(SDAE, self).__init__()
        self.encoder = nn.Sequential(  # batch_size * (28*28)
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # batch_size * 784
        x = self.encoder(x)  # batch_size * 64
        x = self.decoder(x)  # batch_size * (28*28)
        return x
