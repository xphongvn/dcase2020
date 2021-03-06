import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################################################
# ALL TORCH MODELS GO HERE

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

class BinaryClassifier(nn.Module):
    def __init__(self, inputDim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(inputDim,128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32,1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        layer1 = self.relu(self.bn1(self.fc1(x)))
        layer2 = self.relu(self.bn2(self.fc2(layer1)))
        layer3 = self.relu(self.bn3(self.fc3(layer2)))
        layer4 = self.relu(self.bn4(self.fc4(layer3)))
        y = self.sigmoid(self.out(layer4))
        return y

class AutoEncoder(nn.Module):
    """
        AutoEncoder
    """

    def __init__(self, inputDim):
        super(AutoEncoder, self).__init__()

        layers = [nn.Linear(inputDim, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, 8),
                  nn.Linear(8, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, inputDim), ]
        self.layers = nn.ModuleList(layers)

        bnorms = [nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(8),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128), ]
        self.bnorms = nn.ModuleList(bnorms)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        hidden = self.relu(self.bnorms[0](self.layers[0](inputs)))  # inputDim->128
        hidden = self.relu(self.bnorms[1](self.layers[1](hidden)))  # 128->128
        hidden = self.relu(self.bnorms[2](self.layers[2](hidden)))  # 128->128
        hidden = self.relu(self.bnorms[3](self.layers[3](hidden)))  # 128->128
        hidden = self.relu(self.bnorms[4](self.layers[4](hidden)))  # 128->8
        hidden = self.relu(self.bnorms[5](self.layers[5](hidden)))  # 8->128
        hidden = self.relu(self.bnorms[6](self.layers[6](hidden)))  # 128->128
        hidden = self.relu(self.bnorms[7](self.layers[7](hidden)))  # 128->128
        hidden = self.relu(self.bnorms[8](self.layers[8](hidden)))  # 128->128
        output = self.layers[9](hidden)                             # 128->inputDim

        return output

##################################################################################
# FUNCTION TO GET AND LOAD TORCH MODEL
##################################################################################
def get_model(inputDim, model_class="BaselineModel"):
    if model_class == "BaselineModel":
        return BaselineModel(inputDim).to(device)
    elif model_class == "BinaryClassifier":
        return BinaryClassifier(inputDim).to(device)
    elif model_class == "AutoEncoder":
        return AutoEncoder(inputDim).to(device)
    else:
        raise("Class not implemented")

def load_model(file_path, inputDims):
    model = get_model(inputDims)
    model.eval()
    model.load_state_dict(torch.load(file_path))
    return model