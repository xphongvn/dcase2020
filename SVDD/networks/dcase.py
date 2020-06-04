import torch.nn as nn
from SVDD.base.base_net import BaseNet

#################################################################################
# Torch Model for SVDD

class SVDD_Rep(BaseNet):

    def __init__(self, inputDim):
        super().__init__()
        self.rep_dim = 8
        self.encoder = nn.Sequential(
            nn.Linear(inputDim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, self.rep_dim),
            nn.BatchNorm1d(self.rep_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class SVDD_Rep_Autoencoder(BaseNet):
    def __init__(self, inputDim):
        super().__init__()
        self.rep_dim = 8
        # Encoder must be the same with above
        self.encoder = nn.Sequential(
            nn.Linear(inputDim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, self.rep_dim),
            nn.BatchNorm1d(self.rep_dim),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.rep_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, inputDim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



