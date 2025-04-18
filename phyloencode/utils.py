#!/usr/bin/env python3
import torch
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


def mmd_loss(latent_pred, target = None):

    device = latent_pred.device
    x = latent_pred
    if target is None:
        y = torch.randn(x.shape).to(device)
    else:
        y = target[0:x.shape[0],:]
    MMD = MMDLoss(device)(x, y)
            
    return MMD

def vz_loss(latent_pred, target = None):
    device = latent_pred.device
    x = latent_pred
    if target is None:
        y = torch.randn(x.shape).to(device)
    else:
        y = target[0:x.shape[0],:]
    VZ = VZLoss(device)(x, y)

    return VZ

def recon_loss(x, y, 
               num_chars = 0, 
               char_type = "categorical", # [categorical, continuous]
               char_weight = 0.5, 
               tip1_weight = 0.0):
    """
    Compute the reconstruction loss between the reconstructed and original cblv+S and aux data.
    Default is to weight the first two values in the cblv+S data equally to the rest of the data.
    And weight the first tip in the phylogenetic data equally to the rest of the data.

    Note: If Char_weight is 0.0, character data informs the tree reconstruction, but not character reconstruction.

    Args:
        x (torch.Tensor): Reconstructed data.
        y (torch.Tensor): Original data.
        char_weight (float): Weight for the character data in the loss calculation.
        tip1_weight (float): Weight for the first tip in the phylogenetic data.
    """
    # we assume that the first two values in the flattened clbv values
    # and the rest are the character values and potentially two additional cblv-like values
    
    # use cat_recon_loss(x, y) if x is character data

    # check that x is cblv-like data
    if len(x.shape) == 3:
        # phylogenetic data contains extra cblv elements and/or character data 
        # dims are (batch_size, num_channels, num_tips)
        if char_weight == 0. or x.shape[1] <= 2 or num_chars == 0:
            tree_mse_loss = fun.mse_loss(x, y)
            char_loss = torch.tensor(0.0)
        else:
            tree_mse_loss = fun.mse_loss(x[:,0:(x.shape[1]-num_chars)], y[:,0:(x.shape[1]-num_chars)])
            if char_type == "categorical": 
                char_loss = cat_recon_loss(x[:,(x.shape[1]-num_chars):], y[:,(x.shape[1]-num_chars):])
            else:
                char_loss = fun.mse_loss(x[:,(x.shape[1]-num_chars):], y[:,(x.shape[1]-num_chars):])

        # Phyddle normalizes tree heights to all = 1.
        # So first two values in flattened clbv (+s) should be [1,0]
        # adding a term for just the first two values is equivalent to
        # upweighting the first two values in the mse loss
        if tip1_weight > 0.:
            tip1_loss = fun.mse_loss(x[:,0:2,0], y[:,0:2,0]) 
        else:
            tip1_loss = 0.

        return (1 - char_weight) * tree_mse_loss + char_weight * char_loss + tip1_weight * tip1_loss

    else:
        # auxiliary data is (batch_size, num_features)
       return fun.mse_loss(x, y)


def cat_recon_loss(x, y):
    """
    Compute the reconstruction loss for character data.

    Args:
        x (torch.Tensor): Reconstructed character data ((nchannels - 2) x ntips or (nchannels - 4) x ntips).
        y (torch.Tensor): Original character data ((nchannels - 2) x ntips or (nchannels - 4) x ntips).
        is_categorical (bool): Whether the character data is categorical.

    Returns:
        torch.Tensor: Reconstruction loss for character data.
    """
    
    # Use cross-entropy loss for categorical data
    # Reshape to (batch_size * ntips, n_classes) for cross-entropy
    x = x.permute(0, 2, 1).reshape(-1, x.shape[1])  # (batch_size * ntips, n_classes)
    y = y.permute(0, 2, 1).reshape(-1, y.shape[1])  # (batch_size * ntips, n_classes)
    loss = fun.cross_entropy(x, y, reduction = 'mean')

    return loss


def conv1d_sequential_outshape(sequential: nn.Sequential, 
                               input_channels: int, 
                               input_width: int) -> Tuple[int, int, int]:
    """
    Compute the output shape of a PyTorch Sequential model consisting of Conv1d layers.

    Args:
        sequential (nn.Sequential): Sequential model with Conv1d layers.
        input_width (int): Width of the input data (e.g., number of time steps).
        input_channels (int): Number of input channels.

    Returns:
        tuple: Output shape as (batch_size, channels, width).
    """
    # Initialize with input shape
    batch_size = 1  # Assume batch size of 1
    channels = input_channels
    width = input_width

    # Iterate through the layers in the Sequential model
    for layer in sequential:
        if isinstance(layer, nn.Conv1d):
            width    = conv1d_layer_outwidth(layer, width)
            channels = layer.out_channels

    return batch_size, channels, width

def tconv1d_sequential_outshape(sequential: nn.Sequential, 
                                input_channels: int, 
                                input_width: int) -> Tuple[int, int, int]:
    """
    Compute the output shape of a PyTorch Sequential model consisting of ConvTranspose1d layers.

    Args:
        sequential (nn.Sequential): Sequential model with ConvTranspose1d layers.
        input_width (int): Width of the input data (e.g., number of time steps).
        input_channels (int): Number of input channels.

    Returns:
        tuple: Output shape as (batch_size, channels, width).
    """
    # Initialize with input shape
    batch_size = 1  # Assume batch size of 1
    channels = input_channels
    width = input_width

    # Iterate through the layers in the Sequential model
    for layer in sequential:
        if isinstance(layer, nn.ConvTranspose1d):
            width    = tconv1d_layer_outwidth(layer, width)
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


# classes for MMD2 loss to encourage latent space to be be N(0,1) distributed
class RBF(nn.Module):
    ''' Derived From: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py'''
    def __init__(self, device, n_kernels=5, mul_factor=2., bw=None):
        super().__init__()
        self.device = device
        self.bw_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        # print(mul_factor ** (torch.arange(n_kernels) - n_kernels // 2))
        self.bw_multipliers = self.bw_multipliers.to(self.device)
        if bw != None:
            self.bw = torch.tensor(bw).to(self.device)
        else:
            self.bw = bw

    def get_bw(self, L2_dists):
        if self.bw is None:
            n_samples = L2_dists.shape[0]
            return L2_dists.data.sum() / (n_samples ** 2 - n_samples)

        return self.bw

    def forward(self, X, Y = None):
        # X is 2N x D (N examples, D features) + N, D-dimensional samples from N(0,I)
        # returns a 2N x 2N matrix with the RBF kernel between all possible pairs of rows
        if Y is None:
            L2_dists = torch.cdist(X, X) ** 2
        else:
            L2_dists = torch.cdist(X, Y) ** 2

        sf = (self.get_bw(L2_dists) * self.bw_multipliers)[:, None, None]

        return torch.exp(-L2_dists[None, ...] / sf).sum(dim=0)

class MMDLoss(nn.Module):
    ''' From: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py'''
    def __init__(self, device):
        super().__init__()
        self.kernel = RBF(device, bw = None)

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        k_xx = K[:X_size, :X_size]
        k_xy = K[:X_size, X_size:]
        k_yy = K[X_size:, X_size:]
        # MMD_loss = torch.sqrt((k_xx - 2 * k_xy + k_yy).mean())    # aaa, ppp, qqq, sss, ttt, xxx, yyy, zzz
        MMD_loss = (k_xx - 2 * k_xy + k_yy).mean()    # aaa, ppp, qqq, sss, ttt, xxx, yyy, zzz
        return MMD_loss #+ var_mar
    
class VZLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.kernel = RBF(device)
    
    def forward(self, X, Y):
        return self.kernel(X.T, Y.T).var()

# this is a modified version of sklearn's StandardScaler. For one-hot encoded categorical data.
class StandardScalerPhyCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, num_chars, num_chans, num_tips):
        super().__init__()
        self.cblv_scaler = StandardScaler()
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

        # reshape -> concatenate X_cblv and chardata -> un-reshape
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
    


# this is a modified version of sklearn's StandardScaler
# it is used to transform the data to a log scale before standardizing
########### this performs pretty poorly. Dont use.
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
    



# Print profiling results