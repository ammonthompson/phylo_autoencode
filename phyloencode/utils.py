#!/usr/bin/env python3
import torch
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional, Union

def get_vae_loss_function():
    """
    Variational autoencoder loss

    Returns:
        the loss function. duh.
    """

    def vae_loss_fx(x_hat, x, mu, log_var, kl_weight = 0):
        ''' 
           Args:
            x_hat: tree or auxiliary vector decoder output 
            x: tree or auxiliary data
            mu and log_var: MVN params of latent space
        '''

        # MSE loss
        decoded_loss = torch.nn.MSELoss(reduction = "mean")(x_hat, x)

        # KL Divergence: Encourages latent space to be standard normal
        var = log_var.exp()

        kl_loss = kl_weight * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - var))
        # print(kl_loss)

        # kl_loss = (0.5 * ((x - mu) ** 2 / var + log_var + torch.log(2 * torch.pi))).mean()
        
        return 0*decoded_loss, kl_loss
        

    return vae_loss_fx


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
