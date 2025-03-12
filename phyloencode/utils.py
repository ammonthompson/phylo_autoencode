#!/usr/bin/env python3
import torch
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional, Union

def mmd_loss(latent_pred):

    device = latent_pred.device
    x = latent_pred
    y = torch.randn(x.shape).to(device)
    MMD2 = MMDLoss(device)(x, y)
            
    return MMD2

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

    def forward(self, X):
        L2_dists = torch.cdist(X, X) ** 2
        sf = (self.get_bw(L2_dists) * self.bw_multipliers)[:, None, None]

        return torch.exp(-L2_dists[None, ...] / sf).sum(dim=0)


class MMDLoss(nn.Module):
    ''' From: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py'''
    def __init__(self, device):
        super().__init__()
        self.kernel = RBF(device, bw = None)

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        # print(str(K.shape) + "  " + str(X.shape) + "  " + str(Y.shape))
        # print(K)
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        # print(str(XX) + "  " + str(XY) + "  " + str(YY))
        return XX - 2 * XY + YY
    

