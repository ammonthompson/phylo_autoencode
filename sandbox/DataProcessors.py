
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import h5py
import numpy as np

# these classes work with datasets output from the Format step in Phyddle
class TreeDataSet(Dataset):
    def __init__(self, hdf_file):
        with h5py.File(hdf_file, "r") as f:
            self.tree_features = torch.tensor(f['phy_data'][...], dtype=torch.float32)
            self.aux_features  = torch.tensor(f['aux_data'][...], dtype=torch.float32)
        self.length = self.tree_features.shape[0]

    def __getitem__(self, index):
        return(self.tree_features[index], self.aux_features[index])
    
    def __len__(self):
        return(self.length)
    

class TreeDataLoader(DataLoader):
    def __init__(self, tree_data_set):
        super().__init__()
        self.data_loader = DataLoader(tree_data_set)


# testing

if __name__ == "__main__":
    import sys
    import time
    with h5py.File(sys.argv[1], "r") as f:
        print("keys in file: ", list(f.keys()))
        # phy_data = f['phy_data']
        # aux_data = f['aux_data']
        # labels   = f['labels']
        
    my_tree_data = TreeDataSet(sys.argv[1])
    rand_idx = np.random.randint(0, 100)
    print(my_tree_data.__getitem__(rand_idx))
    print(my_tree_data.__len__())
    time.sleep(20)
