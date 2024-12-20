
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

class TreeDataSet(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels
        dataset = TensorDataset(self.features, self.labels)


class TreeDataLoader(DataLoader):
    def __init__(self, tree_data_set):
        super().__init__()
        self.data_loader = DataLoader(tree_data_set)