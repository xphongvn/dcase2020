from torch.utils.data import Dataset
import torch


class GetDataset(Dataset):
    def __init__(self, data, validation_split=0.1, isValidation=False):
        super(GetDataset, self).__init__()
        numpy_pred = torch.from_numpy(data).float()
        numpy_real = torch.from_numpy(data).float()

        if (validation_split):
            split = int(data.shape[0] * (1. - validation_split))

            if (isValidation):
                numpy_pred = numpy_pred[split:]
                numpy_real = numpy_real[split:]
            else:
                numpy_pred = numpy_pred[:split]
                numpy_real = numpy_real[:split]

        self.X = numpy_pred
        self.y = numpy_real

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return self.X.shape[0]

def train_epoch(model, optimizer, loss_fn, dataloader, device):
    model.train()
    epoch_loss = 0.0
    for i, (feature, target) in enumerate(dataloader):
        feature, target = feature.to(device), target.to(device)
        optimizer.zero_grad()
        #forward
        output = model(feature)
        loss = loss_fn(output, target)
        #backward
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss/len(dataloader)


def evaluate_epoch(model, loss_fn, dataloader, device):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for feature, target in dataloader:
            feature, target = feature.to(device), target.to(device)
            output = model(feature)
            loss = loss_fn(output, target)
            epoch_loss += loss.item()
    return epoch_loss/len(dataloader)