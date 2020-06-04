from SVDD.base.base_dataset import BaseADDataset
from torch.utils.data import Dataset, DataLoader
from torch_utils import GetDataset, GetDatasetBinaryLabel

class DCASE_Dataset(BaseADDataset):
    def __init__(self, train_data, test_data, train_labels, test_labels, normal_class=0):
        super().__init__(root="")
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 2))
        self.outlier_classes.remove(normal_class)

        # must be of type torch.utils.data.Dataset
        self.train_set = GetDatasetBinaryLabel(train_data, anomaly_label=train_labels)
        # must be of type torch.utils.data.Dataset
        self.test_set = GetDatasetBinaryLabel(test_data, anomaly_label=test_labels)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0):
        train_loader = DataLoader(dataset=self.train_set,
                                  batch_size=batch_size,
                                  shuffle=shuffle_train,
                                  num_workers=num_workers
                                  )

        test_loader = DataLoader(dataset=self.test_set,
                                 batch_size=batch_size,
                                 shuffle=shuffle_test,
                                 num_workers=num_workers)

        return (train_loader, test_loader)