
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import h5py
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

# these classes work with datasets output from the Format step in Phyddle
class TreeDataSet(Dataset):
    def __init__(self, phy_features, aux_features):
        super().__init__()
        self.phy_features = phy_features
        self.aux_features = aux_features
        self.length = self.phy_features.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return((self.phy_features[index], self.aux_features[index]))
    
    def __len__(self):
        return(self.length)
    


# testing

if __name__ == "__main__":
    import sys
    import time
    with h5py.File(sys.argv[1], "r") as f:
        print("keys in file: ", list(f.keys()))
        
    my_tree_data = TreeDataSet(sys.argv[1])
    rand_idx = np.random.randint(0, 100)
    print(my_tree_data.__getitem__(rand_idx))
    print(my_tree_data.__len__())
    time.sleep(20)
