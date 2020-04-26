

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaselineModel(nn.Module):
    def __init__(self, inputDim):
        super(BaselineModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inputDim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, inputDim),
        )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(x),
            return encoded, decoded

def get_model(inputDim):
    return BaselineModel(inputDim).to(device)

def load_model(file_path, inputDims):
    model = get_model(inputDims)
    model.load_state_dict(torch.load(file_path))
