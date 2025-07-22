#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import importlib.util
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


# Soft clipping function
class SoftClip(nn.Module):
    def __init__(self, low=1e-6):
        super().__init__()
        self.low = low

    def forward(self, x):
        return self.low + (1 - 2 * self.low) * torch.sigmoid(x)


# these classes work with datasets output from the Format step in Phyddle
# they are used to normalize the data before training
# they are used to create a DataSet object. See TreeDataSet class
class StandardScalerPhyCategorical(BaseEstimator, TransformerMixin):
    # this is a modified version of sklearn's StandardScaler. 
    # For one-hot encoded categorical data.
    # normalizes the data to have mean 0 and std 1.
    # ignores the categorical data
    def __init__(self, num_chars, num_chans, num_tips):
        super().__init__()

        # self.cblv_scaler = StandardScaler()
        self.cblv_scaler = PositiveStandardScaler()

        self.num_chars = num_chars
        self.num_chans = num_chans
        self.num_tips   = num_tips

        dim1_idx = np.arange(0, num_chans * num_tips, num_chans)
        char_idx = np.arange(num_chans - num_chars, num_chans)
        phy_idx  = np.arange(0, num_chans - num_chars)

        self.char_idxs = (dim1_idx[:, None] + char_idx).flatten()
        self.tree_idxs = (dim1_idx[:, None] + phy_idx).flatten()


    def fit(self, X, y=None):
        # Ensure X is a NumPy array
        X = np.asarray(X)
        self.cblv_scaler.fit(X[:, self.tree_idxs])
        return self
    
    def transform(self, X):
        X = np.asarray(X)
        
        # Standardize the cblv-like data
        X_cblv = self.cblv_scaler.transform(X[:, self.tree_idxs])
        char_data = X[:, self.char_idxs]

        # reshape -> then concatenate X_cblv and chardata -> un-reshape
        X_cblv    = X_cblv.reshape(X.shape[0], self.num_chans - self.num_chars, self.num_tips, order = "F")
        char_data = char_data.reshape(X.shape[0], self.num_chars, self.num_tips, order= "F")
        X = np.concatenate((X_cblv, char_data), axis=1)
        X = X.reshape(X.shape[0], -1, order="F")

        return X

    def inverse_transform(self, X):
        # Convert PyTorch tensor back to NumPy array if necessary
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        # separate cblv-like and character data
        X_cblv = X[:, self.tree_idxs]
        char_data = X[:, self.char_idxs]

        # Inverse transform the cblv-like data
        X_cblv = self.cblv_scaler.inverse_transform(X_cblv)

        # reshape before concatenating along axis=1 ( cblv-like data and character data)
        X_cblv = X_cblv.reshape(X.shape[0], self.num_chans - self.num_chars, self.num_tips, order="F")
        char_data = char_data.reshape(X.shape[0], self.num_chars, self.num_tips, order="F")

        # Concatenate the cblv-like and character data
        X = np.concatenate((X_cblv, char_data), axis=1)
        X = X.reshape(X.shape[0], -1, order="F")

        return X
    
class ShiftedStandardScaler(BaseEstimator, TransformerMixin):
    # this is a modified version of sklearn's StandardScaler. 
    # standardizes the data then shifts it to be positive
    # this is used for the cblv-like data

    def __init__(self, buffer_factor = 1., copy = True):
        '''arguments:
        buffer_factor: float, default=1.0
            The factor by which to multiply the minimum value of the
            scaled data to ensure all values are positive.
            A value of 1.0 means no change in scale, while a value of 
            2.0 will shift the data to be at least twice the minimum value.'''
        
        print("Using ShiftedStandardScaler")
        super().__init__()
        self.copy = copy
        self.scaler = StandardScaler()
        self.msm = None
        self.bf = buffer_factor # default no change

    def fit(self, X, y=None):
        # Ensure X is a NumPy array
        X = np.asarray(X)
        self.scaler.fit(X)
        # find minimum value for whole dataset
        self.msm = self.bf * np.abs(np.min(self.scaler.transform(X)))

        return self
    
    def transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)
        # Standardize the data
        X_scaled = self.scaler.transform(X)
        # Shift the data to make all values positive
        X_scaled_shifted = X_scaled + self.msm

        return X_scaled_shifted
    
    def inverse_transform(self, X):
        # Convert PyTorch tensor back to NumPy array if necessary
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        # Reverse the shift
        X_inv = X - self.msm
        # Reverse standardization
        return self.scaler.inverse_transform(X_inv)
   
class NoScalerNormalizer(BaseEstimator, TransformerMixin):
    # this just creates a normalizer object that 
    # returns the input data untransformed
    # for compatibility with downstream code
    def __init__(self, copy=True):
        self.copy = copy
        print("Using NoScalerNormalizer")
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return X.copy() if self.copy else X

    def inverse_transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return X.copy() if self.copy else X

class LogitStandardScaler(BaseEstimator, TransformerMixin):
    """
    Combines a logit transformation with standard scaling.

    - Applies logit: log(x / (1 - x)), after clipping to avoid 0/1 boundaries.
    - Applies StandardScaler afterward.
    """

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.scaler = StandardScaler()

    def _logit(self, x):
        x_clipped = np.clip(x, self.epsilon, 1 - self.epsilon)
        return np.log(x_clipped / (1 - x_clipped))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y=None):
        X_logit = self._logit(np.asarray(X))
        self.scaler.fit(X_logit)
        return self

    def transform(self, X):
        X_logit = self._logit(np.asarray(X))
        return self.scaler.transform(X_logit)

    def inverse_transform(self, X_scaled):
        X_logit = self.scaler.inverse_transform(X_scaled)
        return self._sigmoid(X_logit)
    
class StandardMinMaxScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[np.where(self.std == 0)] = 1.
        X_std = (X - self.mean) / self.std
        self.min = X_std.min(axis=0)
        self.max = X_std.max(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_std = (X - self.mean) / self.std
        # Avoid division by zero
        scale = np.where(self.max != self.min, self.max - self.min, 1)
        X_scaled = 2 * (X_std - self.min) / scale - 1
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_std = ((X_scaled + 1) / 2) * (self.max - self.min) + self.min
        return X_std * self.std + self.mean
    
class LogScaler(BaseEstimator, TransformerMixin):
    def __init__(self, min_pos = 1e-8):
        super().__init__()
        self.min_positive_values = min_pos

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        X = np.asarray(X)
        return(np.log(X + self.min_positive_values))

    def inverse_transform(self, X):
        return(np.exp(X) - self.min_positive_values)    

class LogStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, base=np.float32(np.e)):
        super().__init__()
        self.base = base
        self.scaler = StandardScaler()
        self.min_positive_values = None

    def fit(self, X, y=None):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Compute the smallest positive value for each feature
        self.min_positive_values = np.min(np.where(X > 0, X, np.inf), axis=0)

        # Replace infinities (features with no positive values) with 1e-8
        self.min_positive_values = np.where(np.isinf(self.min_positive_values), 
        1e-8, self.min_positive_values)

        # Shift the data to make all values strictly positive
        X_shifted = X + self.min_positive_values

        # Apply log transformation
        X_log = np.log(X_shifted) / np.log(self.base)

        # Fit the standard scaler on the log-transformed data
        self.scaler.fit(X_log)
        return self

    def transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Shift the data using the precomputed smallest positive values
        X_shifted = X + self.min_positive_values

        # Apply log transformation
        X_log = np.log(X_shifted) / np.log(self.base)

        # Standardize the log-transformed data
        X_scaled = self.scaler.transform(X_log)

        return X_scaled

    def inverse_transform(self, X):
        # Convert PyTorch tensor back to NumPy array if necessary
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        # Reverse standardization
        X_inv = self.scaler.inverse_transform(X)

        # Reverse log transformation
        X_exp = np.exp(X_inv * np.log(self.base))

        # Reverse the shift
        return X_exp - self.min_positive_values

class PositiveStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None
        self.mask = None
    
    def fit(self, X, y=None):
        self.mask = X > 0
        self.mask[:, 0:2] = True  # Always include the first elements

        sum_X = np.sum(X * self.mask, axis=0)
        num_nonzero = np.sum(self.mask, axis=0)

        invalid = num_nonzero <= 1
        num_nonzero_safe = np.where(invalid, 2, num_nonzero)

        self.mean = sum_X / num_nonzero_safe

        residuals = (X - self.mean) * self.mask
        self.std = np.sqrt(np.sum(residuals ** 2, axis=0) / (num_nonzero_safe - 1))

        self.mean[invalid] = 0.
        self.std[invalid] = 1.
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return np.array((X - self.mean) / self.std, dtype=np.float32)
    
    def inverse_transform(self, X):
        return np.array((X * self.std) + self.mean, dtype=np.float32)


# other functions
def read_config(config_file: str) -> Dict[str, Union[int, float, str]]:
    """
    Read a configuration file and return the settings as a dictionary.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Dictionary containing the settings.
    """
    # read a config file that contains one dictionary called settings
    spec = importlib.util.spec_from_file_location("settings", config_file)
    settings_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings_module)
    settings = settings_module.settings
    return settings

def get_outshape(sequential: nn.Sequential, input_channels: int, input_width: int) -> Tuple[int, int, int]:
    
    """
    Compute the output shape of a PyTorch Sequential model.
    Args:
        sequential (nn.Sequential): Sequential model.       
        input_shape (tuple): Input shape as (batch_size, channels, width).
    Returns:
        tuple: Output shape as (batch_size, channels, width).
    """
    # Initialize with input shape
    batch_size, channels, width = 1, input_channels, input_width
    # Iterate through the layers in the Sequential model
    for layer in sequential:
        if isinstance(layer, nn.Conv1d):
            width = conv1d_layer_outwidth(layer, width)
            channels = layer.out_channels
        elif isinstance(layer, nn.ConvTranspose1d):
            width = tconv1d_layer_outwidth(layer, width)
            channels = layer.out_channels
     
    return batch_size, channels, width

def conv1d_layer_outwidth(layer, input_width):
    # Extract Conv1d parameters
    kernel_size = layer.kernel_size[0]
    stride      = layer.stride[0]
    padding     = layer.padding[0]
    dilation    = layer.dilation[0]

    # Compute output width using the Conv1d formula
    width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return width

def tconv1d_layer_outwidth(layer, input_width):
    # Extract ConvTranspose1d parameters
    kernel_size    = layer.kernel_size[0]
    stride         = layer.stride[0]
    padding        = layer.padding[0]
    output_padding = layer.output_padding[0]
    dilation       = layer.dilation[0]
    # Compute output width using the ConvTranspose1d formula
    width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return width

def get_numtips(tree_data) ->  torch.Tensor:
    """
    Get the number of tips from the tree data.

    Args:
        tree_data (torch.Tensor): Tree data tensor.

    Returns:
        int: Number of tips.
    """
    # tree_data should have shape (batch_size, num_channels, num_tips)
    # get the number of tips from the length of zero paddings at the end of each tree

    # convert to numpy array
    if isinstance(tree_data, torch.Tensor):
        # check if tree_data is on GPU
         tree_data = tree_data.numpy()

    num_tips = []
    for i in range(tree_data.shape[0]):
        # get the number of tips from the length of zero paddings at the end of each tree
        tree_cumsum = tree_data.cumsum(tree_data[i,1,...] == 0, axis=2)
        tree_cummax = tree_cumsum.max()
        first_max = np.where(tree_cumsum == tree_cummax)[0]
        num_tips.append(first_max)

    return torch.Tensor(num_tips, dtype = torch.int32)

