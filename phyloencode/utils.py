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
import importlib.util
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Loss functions
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
               tip1_weight = 0.0,
               mask = None):
    """
    Compute the reconstruction loss between the reconstructed and original cblv+S and aux data.
    Note: If Char_weight is 0.0, and num_chars > 0,
      character data informs the tree reconstruction, but not character reconstruction.

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
            tree_loss = phy_recon_loss(x, y, mask)
            char_loss = torch.tensor(0.0)
            char_weight = 0.
        else:
            tree_loss = phy_recon_loss(x[:,0:(x.shape[1]-num_chars)], 
                                       y[:,0:(x.shape[1]-num_chars)], 
                                       mask[:,0:(x.shape[1]-num_chars)])
            char_loss = char_recon_loss(x[:,(x.shape[1]-num_chars):], 
                                        y[:,(x.shape[1]-num_chars):], 
                                        char_type,
                                        mask[:,(x.shape[1]-num_chars):])


        # Phyddle normalizes tree heights to all = 1.
        # So first two values in flattened clbv (+s) should be [1,0]
        # adding a term for just the first two values is equivalent to
        # upweighting the first two values in the mse loss
        if tip1_weight > 0.:
            tip1_loss = fun.mse_loss(x[:,0:2,0], y[:,0:2,0]) 
        else:
            tip1_loss = 0.
        
        return (1 - char_weight) * tree_loss + char_weight * char_loss + tip1_weight * tip1_loss

    else:
        # auxiliary data is (batch_size, num_features)
       return fun.mse_loss(x, y)

def phy_recon_loss(x, y, mask = None):
    """
    Compute the reconstruction loss for phylogenetic data.

    Args:
        x (torch.Tensor): Reconstructed phylogenetic data in cblv-like format.
        y (torch.Tensor): Original phylogenetic data in same format.

    Returns:
        torch.Tensor: Reconstruction loss for phylogenetic data.
    """
    # if cblv-like data, use the first two channels in the loss
    # else if augmented cblv-like data, use the first four channels in the data
    if mask is not None:
        tree_loss = (fun.mse_loss(x, y, reduction='none') * mask).mean()  
    else:
        tree_loss = fun.mse_loss(x, y) 

    tip1_loss = fun.mse_loss(x[:,0:2,0], y[:,0:2,0]) 

    return tree_loss + tip1_loss

def char_recon_loss(x, y, char_type = "categorical", mask = None):
    """
    Compute the reconstruction loss for categorical character data.

    Args:
        x (torch.Tensor): Reconstructed character data ((nchannels - 2) x ntips or (nchannels - 4) x ntips).
        y (torch.Tensor): Original character data ((nchannels - 2) x ntips or (nchannels - 4) x ntips).
        is_categorical (bool): Whether the character data is categorical.

    Returns:
        torch.Tensor: Reconstruction loss for character data.
    """

    if char_type == "categorical": 
        # Use cross-entropy loss for categorical data
        # Reshape to (batch_size * ntips, n_classes) for cross-entropy
        x = x.permute(0, 2, 1).reshape(-1, x.shape[1])  # (batch_size * ntips, n_classes)
        y = y.permute(0, 2, 1).reshape(-1, y.shape[1])  # (batch_size * ntips, n_classes)
        mask = mask.permute(0, 2, 1).reshape(-1, mask.shape[1])

        # Convert one-hot to class indices
        y = y.argmax(dim=1)  # shape (N,)
        
        char_loss = fun.cross_entropy(x, y, reduction = 'none')
    else:
        char_loss = fun.mse_loss(x, y, reduction = 'none')

    if mask is not None:
        # Apply the mask to the loss
        char_loss = char_loss * mask[:,0] # match dims (all columns are the same in mask)

    return char_loss.mean()


def aux_recon_loss(x, y):
    """
    Compute the reconstruction loss for auxiliary data.

    Args:
        x (torch.Tensor): Reconstructed auxiliary data.
        y (torch.Tensor): Original auxiliary data.

    Returns:
        torch.Tensor: Reconstruction loss for auxiliary data.
    """
    # Use mean squared error for auxiliary data
    return fun.mse_loss(x, y)
    # return fun.l1_loss(x, y)

# developing
def losses(pred : torch.tensor, true : torch.tensor, mask : torch.tensor, char_type : str):
    # separate components
    phy_hat, char_hat, aux_hat, latent_hat = pred
    phy, char, aux, std_norm = true
    tree_mask, char_mask = mask

    device = phy_hat.device

    # recon loss
    phy_loss    = phy_recon_loss(phy_hat, phy, mask = tree_mask)
    char_loss   = char_recon_loss(char_hat, char, char_type, char_mask) if char is not None else torch.tensor(0.).to(device)
    aux_loss    = aux_recon_loss(aux_hat, aux)

    # latent loss
    latent_mmd_loss    = mmd_loss(latent_hat, std_norm) if latent_hat is not None else torch.tensor(0.).to(device)
    latent_vz_loss     = vz_loss(latent_hat, std_norm)  if latent_hat is not None else torch.tensor(0.).to(device)

    return phy_loss, char_loss, aux_loss, latent_mmd_loss, latent_vz_loss


class PhyLoss(object):
    # holds component losses over epochs
    # to be called for an individual batch
    # methods:
        # compute, stores, and returns component and total loss for val and train sets
        # plots loss curves
    def __init__(self, weights : torch.Tensor, char_type = None, latent_layer_Type = "GAUSS"):
        # weights for all components
        # initialize component loss vectors for train and validation losses
        
        # epoch losses are the average of the batch losses
        # epoch losses
        self.epoch_total_loss   = []
        self.epoch_phy_loss     = []
        self.epoch_char_loss    = []
        self.epoch_aux_loss     = []
        self.epoch_ntips_loss   = []
        self.epoch_mmd_loss     = []
        self.epoch_vz_loss      = []
        # batch losses 
        self.batch_total_loss   = []
        self.batch_phy_loss     = []
        self.batch_char_loss    = []
        self.batch_aux_loss     = []
        self.batch_ntips_loss   = []
        self.batch_mmd_loss     = []
        self.batch_vz_loss      = []
        # loss weights
        self.phy_w  = weights[0]
        self.char_w = weights[1]
        self.aux_w  = weights[2]
        self.mmd_w  = weights[3]
        self.vz_w   = weights[4]

        self.char_type = char_type
        self.latent_layer_type = latent_layer_Type
        

    def minibatch_loss(self, x : torch.Tensor, y : torch.Tensor, mask : torch.Tensor):
        # appends to the self.losses and returns the current batch loss
        # x is a tuple (dictionary maybe?) of predictions for a batch
        # y is a tuple of the true values for a batch
        # both contain: 
            # phy, char, mask, aux, ntips

        phy_loss, char_loss, aux_loss, mmd_loss, vz_loss = losses(x, y, mask, self.char_type)

        total_loss =    self.phy_w  * phy_loss  + \
                        self.char_w * char_loss + \
                        self.aux_w  * aux_loss  + \
                        self.mmd_w  * mmd_loss  + \
                        self.vz_w   * vz_loss
        
        self._append_minibatch_losses(total_loss, phy_loss, char_loss, aux_loss, mmd_loss, vz_loss)

        return total_loss


    def append_mean_batch_loss(self):
        # averages the batch loss arrays and return 
        mean_total_loss = torch.mean(torch.stack(self.batch_total_loss)).item()
        mean_phy_loss   = torch.mean(torch.stack(self.batch_phy_loss)).item()
        mean_char_loss  = torch.mean(torch.stack(self.batch_char_loss)).item()
        mean_aux_loss   = torch.mean(torch.stack(self.batch_aux_loss)).item()
        mean_mmd_loss   = torch.mean(torch.stack(self.batch_mmd_loss)).item()
        mean_vz_loss    = torch.mean(torch.stack(self.batch_vz_loss)).item()

        self._append_epoch_losses(mean_total_loss, mean_phy_loss, mean_char_loss, 
                                  mean_aux_loss, mean_mmd_loss, mean_vz_loss)
        
        # reset batch losses
        self.batch_total_loss   = []
        self.batch_phy_loss     = []
        self.batch_char_loss    = []
        self.batch_aux_loss     = []
        self.batch_ntips_loss   = []
        self.batch_mmd_loss     = []
        self.batch_vz_loss      = []

  

    def print_epoch_losses(self, elapsed_time):
        print(  f"Epoch {len(self.epoch_total_loss)},  " +
                f"Loss: {self.epoch_total_loss[-1]:.4f},  " +
                f"phy L: {self.epoch_phy_loss[-1]:.4f},  " +
                f"char L: {self.epoch_char_loss[-1]:.4f},  " +
                f"aux L: {self.epoch_aux_loss[-1]:.4f},  " +
                f"MMD L: {self.epoch_mmd_loss[-1]:.4f},  " +
                f"VZ L: {self.epoch_vz_loss[-1]:.4f},  " +
                f"Run time: {elapsed_time:.3f} sec" )

    # helpers
    def _append_minibatch_losses(self, total_loss, phy_loss, 
                                 char_loss, aux_loss, mmd_loss, vz_loss):
        self.batch_total_loss.append(total_loss)
        self.batch_phy_loss.append(phy_loss)
        self.batch_char_loss.append(char_loss)
        self.batch_aux_loss.append(aux_loss)
        self.batch_mmd_loss.append(mmd_loss)
        self.batch_vz_loss.append(vz_loss)

    def _append_epoch_losses(self, total_loss, phy_loss, 
                             char_loss, aux_loss, mmd_loss, vz_loss):
        self.epoch_total_loss.append(total_loss)
        self.epoch_phy_loss.append(phy_loss)
        self.epoch_char_loss.append(char_loss)
        self.epoch_aux_loss.append(aux_loss)
        self.epoch_mmd_loss.append(mmd_loss)
        self.epoch_vz_loss.append(vz_loss)


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
        # MMD_loss = torch.sqrt((k_xx - 2 * k_xy + k_yy).sum())/sum(X.shape)    # aaa, ppp, qqq, sss, ttt, xxx, yyy, zzz
        # MMD2_loss = k_xx.mean() - 2 * k_xy.mean() + k_yy.mean()   # aaa, ppp, qqq, sss, ttt, xxx, yyy, zzz
        MMD2_loss = (k_xx - 2 * k_xy + k_yy).mean()    # aaa, ppp, qqq, sss, ttt, xxx, yyy, zzz
        MMD_loss = torch.sqrt(MMD2_loss) #   # aaa, ppp, qqq, sss, ttt, xxx, yyy, zzz
        return MMD_loss
    
class VZLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.kernel = RBF(device)
    
    def forward(self, X, Y):

        # TESTING
        # X_centered = X - X.mean(dim=0, keepdim=True)
        # std = X.std(dim=0, unbiased=False, keepdim=True)
        # X_norm = X_centered / (std + 1e-8)  # Normalize each feature to zero mean and unit variance

        # corr_matrix = (X_norm.T @ X_norm) / X.shape[0]  # Shape: [latent_dim, latent_dim]
        # corr_off_diag = corr_matrix[~torch.eye(corr_matrix.size(0), dtype=bool, device=X.device)]
        
        return self.kernel(X.T, Y.T).var()
        # return corr_off_diag.abs().mean()
        # return self.kernel(X.T, Y.T).var() + corr_off_diag.abs().mean()
    

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