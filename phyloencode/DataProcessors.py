
import torch
from torch.utils.data import Dataset, DataLoader #, TensorDataset
import h5py
import numpy as np
import sklearn
import joblib
import sklearn.preprocessing as pp
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional, Union
import phyloencode.utils as utils
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt
from phyloencode.utils            import get_num_tips


# class that contains the DataSet and DataLoaders
# splits and normalizes/denormalizes it
# 
# Parameters: 
# (phy data, aux data) -> Tuple[torch.Tensor, torch.Tensor], 
# p: proportion of data fro training -> float
# nc: number of channels in the data for reshapeing phy data -> int
# 

class AEData(object):
    def __init__(self, 
                 phy_data   : torch.Tensor,
                 aux_data   : torch.Tensor,
                 prop_train : float, 
                 num_channels  : int,
                 char_data_type : str = "categorical", # "continuous" or "categorical"
                 num_chars  : int = 0,
                 num_tips   : Optional[int] = None):
        """
        Each tree in data is assumed to be flattend in column-major order
        of a matrix of dimensions (num_channels, max_tips).


        Args:
            phy_data (torch.Tensor): _description_
            aux_data (torch.Tensor): _description_
            prop_train (float): _description_
            num_channels (int): _description_
            char_data_type (str, optional): _description_. Defaults to "categorical".
            num_tips (Optional[int], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        

        self.num_channels = num_channels
        self.char_data_type = char_data_type
        self.num_chars = num_chars

        if self.char_data_type == "categorical" and (self.num_chars == 0 or self.num_channels <= 2):
            print("Warning: char_data_type is categorical but num_chars == 0 or num_channels <= 2. "
            "Setting char_data_type to continuous.")
            self.char_data_type = "continuous"

        # prepare phy_data and aux_data for splitting into train and val sets  
        self.max_tips = phy_data.shape[1] / num_channels
        self.phy_data = phy_data
        self.aux_data = aux_data

        # if not provided, get num_tips and concatenate to aux_data
        if num_tips is None:
            num_tips = get_num_tips(self.phy_data, self.phy_data / num_channels)
            self.aux_data = torch.hstack((num_tips, self.aux_data))

        # TODO: no good reason for concatenating phy and aux data before splitting
        if num_tips is None:
            self.data = np.hstack((self.phy_data, self.aux_data))
        else:
            self.data = np.hstack((self.phy_data, self.aux_data, num_tips))
            
        flat_phy_width = self.phy_data.shape[1]
        self.max_tips = flat_phy_width // num_channels

        # split data 
        self.prop_train = prop_train
        num_train = int(prop_train * self.phy_data.shape[0])
        train_data, val_data = train_test_split(self.data, train_size = num_train, shuffle=True)

        # separate phy and aux data and num_tips if not None
        # if num_tips is None:
        #     train_phy_data = train_data[:,:flat_phy_width]
        #     val_phy_data   = val_data[:,:flat_phy_width]
        #     train_aux_data = train_data[:,flat_phy_width:]
        #     val_aux_data   = val_data[:,flat_phy_width:]
        #     train_num_tips = None
        #     val_num_tips   = None
        # else:
        #     train_phy_data = train_data[:,:flat_phy_width]
        #     val_phy_data   = val_data[:,:flat_phy_width]
        #     train_aux_data = train_data[:,flat_phy_width:-1]
        #     val_aux_data   = val_data[:,flat_phy_width:-1]
        #     train_num_tips = train_data[:,-1]
        #     val_num_tips   = val_data[:,-1]
        train_phy_data = train_data[:,:flat_phy_width]
        val_phy_data   = val_data[:,:flat_phy_width]
        train_aux_data = train_data[:,flat_phy_width:-1]
        val_aux_data   = val_data[:,flat_phy_width:-1]
        train_num_tips = train_data[:,-1]
        val_num_tips   = val_data[:,-1]

 

        # standardize train data
        if self.char_data_type == "continuous":
            if num_tips is None:
                self.phy_ss = pp.StandardScaler()
            else:
                self.phy_ss = utils.PositiveStandardScaler()
        elif self.char_data_type == "categorical":
            self.phy_ss = utils.StandardScalerPhyCategorical(self.num_chars, 
                                                             self.num_channels, self.max_tips)
        else:
            raise ValueError("char_data_type must be 'continuous' or 'categorical'")
        self.aux_ss = pp.StandardScaler()

        self.phy_normalizer = self.phy_ss.fit(train_phy_data)
        self.aux_normalizer = self.aux_ss.fit(train_aux_data)
        self.norm_train_phy_data = self.phy_normalizer.transform(train_phy_data)
        self.norm_train_aux_data = self.aux_normalizer.transform(train_aux_data)
        self.norm_val_phy_data   = self.phy_normalizer.transform(val_phy_data)
        self.norm_val_aux_data   = self.aux_normalizer.transform(val_aux_data)

        # reshape phy data to (num examples, num channels, num tips)
        # (num examples, num channels x num tips) -> (num examples, num channels, num tips)
        assert(train_phy_data.shape[1] % num_channels == 0)
        self.norm_train_phy_data = self.norm_train_phy_data.reshape((self.norm_train_phy_data.shape[0], 
                                                        num_channels, 
                                                        int(self.norm_train_phy_data.shape[1]/num_channels)),
                                                        order = "F")
        self.norm_val_phy_data   = self.norm_val_phy_data.reshape((self.norm_val_phy_data.shape[0], 
                                                        num_channels, 
                                                        int(self.norm_val_phy_data.shape[1]/num_channels)),
                                                        order = "F")
        self.phy_width = self.norm_train_phy_data.shape[2]
        self.aux_width = self.norm_train_aux_data.shape[1]

        # create Datasets. __getitem__() returns a tuple (phy, aux)
        # used to create a DataLoader object. See get_dataloaders()
        self.train_dataset = TreeDataSet(self.norm_train_phy_data, self.norm_train_aux_data, train_num_tips)
        self.val_dataset   = TreeDataSet(self.norm_val_phy_data,   self.norm_val_aux_data, val_num_tips)


    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        return self.train_dataset, self.val_dataset

    def get_normalizers(self) -> Tuple[sklearn.preprocessing.StandardScaler, sklearn.preprocessing.StandardScaler]:
        return self.phy_normalizer, self.aux_normalizer
    
    def save_normalizers(self, file_prefix):
        joblib.dump(self.phy_normalizer, file_prefix + ".phy_normalizer.pkl")
        joblib.dump(self.aux_normalizer, file_prefix + ".aux_normalizer.pkl")
    
    def get_dataloaders(self, 
                        batch_size  = 32, 
                        shuffle     = True, 
                        num_workers = 0) -> Tuple[DataLoader, DataLoader]:
        # drop last batch if too small
        drop_last = True if (len(self.train_dataset) % batch_size) < 32 else False
        
        # data loaders
        self.train_dataloader = DataLoader(self.train_dataset, 
                                           batch_size   = batch_size, 
                                           shuffle      = shuffle, 
                                           num_workers  = num_workers,
                                           drop_last    = drop_last)
        self.val_dataloader   = DataLoader(self.val_dataset, 
                                           batch_size   = batch_size)
        
        return self.train_dataloader, self.val_dataloader




# these classes work with datasets output from the Format step in Phyddle
class TreeDataSet(Dataset):
    
    def __init__(self, phy_features: torch.Tensor, 
                 aux_features: torch.Tensor, 
                 num_tips: Optional[int] = None):
        """
        If num_tips is provided, a mask is output otherwise mask is None


        Args:
            phy_features (torch.Tensor): _description_
            aux_features (torch.Tensor): _description_
            num_tips (Optional[int], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """

        super().__init__()
        self.phy_features = phy_features
        self.aux_features = aux_features
        self.length = self.phy_features.shape[0]
        self.num_tips = num_tips

        # Basic shape checks
        N = self.phy_features.shape[0]
        if self.aux_features is not None and self.aux_features.shape[0] != N:
            raise ValueError("aux_features and phy_features must have the same N (batch dimension)")

        if num_tips is None:
            self.mask = None
        else:
            # create mask for phy features  
            num_tips = np.array(num_tips, dtype = int).flatten()
            mask = np.zeros(self.phy_features.shape, dtype = bool)
            for i in range(self.phy_features.shape[0]):
                mask[i, :, :num_tips[i]] = True                
            self.mask = torch.tensor(mask, dtype=torch.bool)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.mask is not None:
            return self.phy_features[index], self.aux_features[index], self.mask[index]
        else:
            return self.phy_features[index], self.aux_features[index]
    
    def __len__(self):
        return self.length 
    




###########
# testing #
###########

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
