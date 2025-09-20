#!/usr/bin/env python3
import torch
import torch.nn as nn
# import torch.nn.functional as fun
# from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import importlib.util
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import argparse


# Soft clipping functionality
class SoftClip(nn.Module):
    def __init__(self, low=1e-6):
        super().__init__()
        self.low = low

    def forward(self, x):
        return self.low + (1 - 2 * self.low) * torch.sigmoid(x)

# these classes work with datasets output from the Format step in Phyddle
# they are used to normalize the data before training
# they are used to create a DataSet object. See phyloencode.TreeDataSet class
class PositiveStandardScaler(BaseEstimator, TransformerMixin):
    # this does not use zero-padded values for standardization.
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

class StandardScalerPhyCategorical(BaseEstimator, TransformerMixin):
    # this is a modified version of sklearn's StandardScaler. 
    # For one-hot encoded categorical data.
    # normalizes the tree data to have mean 0 and std 1.
    # ignores the categorical data
    # uses PositiveStandardScaler (see above)

    def __init__(self, num_chars, num_chans, max_tips):
        super().__init__()

        self.cblv_scaler = PositiveStandardScaler()

        self.num_chars = num_chars
        self.num_chans = num_chans
        self.max_tips   = max_tips

        dim1_idx = np.arange(0, num_chans * max_tips, num_chans)
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
        X_cblv    = X_cblv.reshape(X.shape[0], self.num_chans - self.num_chars, self.max_tips, order = "F")
        char_data = char_data.reshape(X.shape[0], self.num_chars, self.max_tips, order= "F")
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
        X_cblv = X_cblv.reshape(X.shape[0], self.num_chans - self.num_chars, self.max_tips, order="F")
        char_data = char_data.reshape(X.shape[0], self.num_chars, self.max_tips, order="F")

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


# other functions
def file_exists(fname):
    "Return error if no file"
    if not os.path.isfile(fname):
        raise argparse.ArgumentTypeError(f"File '{fname}' does not exist.")
    return fname

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
        elif isinstance(layer, torch.nn.AdaptiveAvgPool1d):
            width = layer.output_size
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

def get_num_tips(phydata: np.ndarray, max_tips: int):
    """
      This computes the number of tips by finding the 
      first column in the cblv which only contains zeros

    Args:
        phydata (np.ndarray): Flattened cblv(+s)
        max_tips (int): width of all cblv

    Returns:
        torch.Tensor: number of tips in each tree
    """
        # change consitent zero in in second position to 1.
    phydata = phydata.reshape((phydata.shape[0], phydata.shape[1] // max_tips, max_tips), order = "F")
    phydata = phydata[:,0:2,:] 
    phydata[:,:,0] = 1.
    nt = torch.tensor([[np.where(np.max(phydata[x,...], axis = 0) == 0)[0][0]]
                       for x in range(phydata.shape[0])], dtype=torch.float32).view(-1,1)
    return nt + 1

def set_pred_pad_to_zero(phy_pred: np.ndarray, pred_num_tips: np.ndarray) -> np.ndarray:
    """
    phy_pred: shape (batch_size, num_channels, max_tips)
    pred_num_tips: shape (batch_size,) or (batch_size, 1)
    """
    bs, nc, mt = phy_pred.shape

    # Ensure pred_num_tips is (bs, 1)
    pred_num_tips = pred_num_tips.reshape(-1, 1)

    # idx_grid: shape (bs, mt)
    idx_grid = np.arange(mt).reshape(1, -1)  # (1, mt)
    idx_grid = np.tile(idx_grid, (bs, 1))    # (bs, mt)

    # Mask: True for valid tips, False for padding
    ntip_mask = (idx_grid <= np.round(pred_num_tips, decimals = 0))   # (bs, mt)

    # Expand mask to (bs, nc, mt)
    ntip_mask = np.expand_dims(ntip_mask, axis=1)

    # masked_phy_pred = phy_pred * ntip_mask
    masked_phy_pred = np.where(ntip_mask, phy_pred, 0.0)

    return masked_phy_pred

def phylo_scatterplot(pred_phy_data, true_phy_data, 
                      pred_aux_data, true_aux_data, 
                      aux_names, output):
    """Create scatter plots true v pred for cblv.

    Args:
        pred_phy_data (_type_): _description_
        true_phy_data (_type_): _description_
        pred_aux_data (_type_): _description_
        true_aux_data (_type_): _description_
        aux_names (_type_): _description_
        output (_type_): _description_
    """
    # 
    with PdfPages(output + "_scatter_plots.pdf") as f:
        ndat = min(100, pred_phy_data.shape[1])
        # 9 plots per page
        # plot aux data
        for i in range(pred_aux_data.shape[1] // 9):
            fig, ax = plt.subplots(3, 3)
            fig.tight_layout(pad=2., h_pad=2., w_pad = 2.)
            aux_clabels = aux_names
            for j in range(9):
                d = 9 * i + j
                row = j // 3
                col = j % 3
                min_val = min([np.min(true_aux_data[0:ndat,d]), np.min(pred_aux_data[0:ndat,d])])
                max_val = max([np.max(true_aux_data[0:ndat,d]), np.max(pred_aux_data[0:ndat,d])])
                ax[row][col].scatter(true_aux_data[0:ndat,d], pred_aux_data[0:ndat,d], s = 5)
                ax[row][col].set_xlabel("True", fontsize = 8)
                ax[row][col].set_ylabel("Pred", fontsize = 8)
                ax[row][col].label_outer()
                ax[row][col].set_title(aux_clabels[d], fontsize = 8)
                ax[row][col].plot([min_val, max_val], [min_val, max_val],
                                   color = "r", linewidth=1)
            f.savefig(fig)
            plt.close()

        #plot cblv(+s) data
        for i in range(pred_phy_data.shape[1] // 9):
            fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
            fig.tight_layout(pad=2., h_pad=2., w_pad = 2.)
            for j in range(9):
                d = 9 * i + j
                row = j // 3
                col = j % 3
                min_val = min([np.min(true_phy_data[0:ndat,d]), np.min(pred_phy_data[0:ndat,d])])
                max_val = max([np.max(true_phy_data[0:ndat,d]), np.max(pred_phy_data[0:ndat,d])])
                ax[row][col].scatter(true_phy_data[0:ndat,d], pred_phy_data[0:ndat,d], s = 5)
                ax[row][col].set_xlabel("True", fontsize = 8)
                ax[row][col].set_ylabel("Pred", fontsize = 8)
                ax[row][col].label_outer()
                ax[row][col].set_title("dim " + str(d), fontsize = 8)
                ax[row][col].plot([min_val, max_val], [min_val, max_val],
                                   color = "r", linewidth=1)
            f.savefig(fig)
            plt.close()

def make_loss_plots(train_loss, val_loss = None,  *, latent_layer_type = None,
                out_prefix = "AElossplot", log = True, starting_epoch = 10):
    fig = plt.figure(figsize=(11, 8))
    plt.plot(list(range(len(train_loss.epoch_total_loss)))[starting_epoch:],
                np.log10(train_loss.epoch_total_loss[starting_epoch:]), 
                label='Training Loss', c="r")
    if val_loss:
        plt.plot(list(range(len(val_loss.epoch_total_loss)))[starting_epoch:],
                 np.log10(val_loss.epoch_total_loss[starting_epoch:]), 
                 label='Validation Loss', c='b')
    plt.xlabel('Epochs')
    plt.ylabel('log10 Loss')
    plt.legend()        
    plt.grid(True)
    
    range_y = [min(np.log10(np.concat((train_loss.epoch_total_loss[starting_epoch:], 
                                        val_loss.epoch_total_loss[starting_epoch:])))), 
                max(np.log10(np.concat((train_loss.epoch_total_loss[starting_epoch:], 
                                        val_loss.epoch_total_loss[starting_epoch:]))))]
    
    plt.yticks(ticks = np.linspace(range_y[0], range_y[1], num = 20))
    plt.xticks(ticks=np.arange(0, len(train_loss.epoch_total_loss), 
                                step=len(train_loss.epoch_total_loss) // 10))
    plt.tight_layout()
    plt.savefig(out_prefix + ".loss.pdf", bbox_inches='tight')
    plt.close(fig)

    # plot each loss separately (only validation losses are recorded). 
    # create subplots for each loss component   
    # TODO: fix x-axis tick marks            
    num_subplots = 6 if latent_layer_type == "GAUSS" else 4
    fig, axs = plt.subplots(num_subplots//2, 2, figsize=(11, 8), sharex=True)
    # axs = axs.flatten()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fill_in_loss_comp_fig(val_loss.epoch_total_loss, "combined", axs[0,0], starting_epoch)
    fill_in_loss_comp_fig(val_loss.epoch_phy_loss, "phy", axs[0,1], starting_epoch)
    fill_in_loss_comp_fig(val_loss.epoch_char_loss, "char", axs[1,1], starting_epoch)
    fill_in_loss_comp_fig(val_loss.epoch_aux_loss, "aux", axs[1,0], starting_epoch)
    if latent_layer_type == "GAUSS":
        fill_in_loss_comp_fig(val_loss.epoch_mmd_loss, "mmd", axs[2,0], starting_epoch)
        fill_in_loss_comp_fig(val_loss.epoch_vz_loss, "vz", axs[2,1], starting_epoch)
    plt.savefig(out_prefix + ".component_loss.pdf", bbox_inches='tight')
    plt.close(fig)

def fill_in_loss_comp_fig(val_losses, plot_label, ax, starting_epoch = 10):
    ax.plot(list(range(len(val_losses)))[starting_epoch:], 
            np.log10(val_losses[starting_epoch:]), label=plot_label, c="b")
            # val_losses[starting_epoch:], label=plot_label, c="b")
    ax.set_title(f"{plot_label} Loss")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log10 Loss')
    # ax.set_ylabel('Loss')
    ax.grid(True)
    ax.set_xticks(ticks=np.arange(0, len(val_losses), step=len(val_losses) // 10))


