

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaselineModel(nn.Module):
    def __init__(self, inputDim):
        super(BaselineModel, self).__init__()

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

            nn.Linear(128, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 128),
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

# Reimplementation of pytorch for Baseline
class BaselineTorch(nn.Module):
    def __init__(self, inputDim):
        super(BaselineTorch, self).__init__()
        self.unit1 = _LinearUnit(inputDim, 128)
        self.unit2 = _LinearUnit(128, 128)
        self.unit3 = _LinearUnit(128, 128)
        self.unit4 = _LinearUnit(128, 128)
        self.unit5 = _LinearUnit(128, 8)
        self.unit6 = _LinearUnit(8, 128)
        self.unit7 = _LinearUnit(128, 128)
        self.unit8 = _LinearUnit(128, 128)
        self.unit9 = _LinearUnit(128, 128)
        self.output = torch.nn.Linear(128, inputDim)

    def forward(self, x):
        shape = x.shape
        x = self.unit1(x.view(x.size(0), -1))
        x = self.unit2(x)
        x = self.unit3(x)
        x = self.unit4(x)
        x = self.unit5(x)
        x = self.unit6(x)
        x = self.unit7(x)
        x = self.unit8(x)
        x = self.unit9(x)
        return self.output(x).view(shape)


class _LinearUnit(torch.nn.Module):
    """For use in Task2Baseline model."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return torch.relu(self.bn(self.lin(x.view(x.size(0), -1))))


def get_model(inputDim):
    #return BaselineModel(inputDim).to(device)
    return BaselineTorch(inputDim).to(device)

def load_model(file_path, inputDims):
    model = get_model(inputDims)
    model.load_state_dict(torch.load(file_path))
    return model