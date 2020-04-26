

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
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_model(inputDim):
    return BaselineModel(inputDim).to(device)
    # return BaselineModel(inputDim)

def load_model(file_path, inputDims):
    model = get_model(inputDims)
    model.load_state_dict(torch.load(file_path))
