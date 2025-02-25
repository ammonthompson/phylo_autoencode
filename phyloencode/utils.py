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
        
        return decoded_loss, kl_loss
        

    return vae_loss_fx


def xxxget_mmd_loss_function():

    def get_mmd(pred, true, latent_pred, mmd_weight = 1,):
        """Emprical maximum mean discrepancy. The lower the result
        the more evidence that distributions are the same.
        Derived From: 
            https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html#references
        MMD^2 = E[k(xx, xx)] + E[k(yy, yy)] - 2E[k(xx,yy)]

        Args:
            pred: decoded data
            true: true data
            latent_pred: first sample, distribution P
            kernel: kernel type such as "multiscale" or "rbf"
        """

        device = latent_pred.device
        recon_loss = torch.nn.MSELoss(reduction = "mean")(pred, true)
        x = latent_pred
        y = torch.randn(x.shape).to(device)

        xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        # create a matrix with columns set to repeats of each diagonal element
        # [[1,2]  -> [[1,4]
        #  [3,4]] ->  [1,4]]
        rx = (xx.diag().unsqueeze(0).expand_as(xx)) 
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        # ||x - x'||^2
        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * xy # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))


        # bandwidth_range = [10, 15, 20, 50]
        # for a in bandwidth_range:
        #     XX += torch.exp(-0.5*dxx/a)
        #     YY += torch.exp(-0.5*dyy/a)
        #     XY += torch.exp(-0.5*dxy/a)
        gamma_multiplier = 100000.
        n0 = x.shape[0] // 2 - ((x.shape[0] // 2) & 1)
        gamma = gamma_multiplier * n0 * 2 /\
                (torch.sum(dxx) + 
                 torch.sum(dyy) +
                 2 * torch.sum(dxy))
        
        XX += torch.exp(-0.5*dxx * gamma)
        YY += torch.exp(-0.5*dyy * gamma)
        XY += torch.exp(-0.5*dxy * gamma)

        MMD2 = torch.mean(XX + YY - 2. * XY)
                
        return recon_loss, mmd_weight * MMD2

    return get_mmd


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




class RBF(nn.Module):
    ''' Derived From: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py'''
    def __init__(self, device, n_kernels=5, mul_factor=2., bandwidth=None):
        super().__init__()
        self.device = device
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(self.device)
        if bandwidth != None:
            self.bandwidth = torch.tensor(bandwidth).to(self.device)
        else:
            self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        # print()
        # print(X.shape)
        # print(L2_distances)
        # print(((self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0))
        # print(self.get_bandwidth(L2_distances))
        # print(self.bandwidth_multipliers)
        # print((-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * 
        #                                              self.bandwidth_multipliers)[:, None, None]).sum(dim=0))
        # print()
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * 
                                                     self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):
    ''' From: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py'''
    def __init__(self, device):
        super().__init__()
        self.kernel = RBF(device, bandwidth = None)

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
    

def get_mmd_loss_function():
    ''' 
        This uses the implementation of yiftachbeer on github.
        See classes above
    '''

    def get_mmd(pred, true, latent_pred, mmd_weight = 1,):

        device = latent_pred.device
        x = latent_pred
        y = torch.randn(x.shape).to(device)
        recon_loss = torch.nn.MSELoss(reduction = "mean")(pred, true)
        MMD2 = mmd_weight * MMDLoss(device)(x, y)
                
        return 0*recon_loss, MMD2

    return get_mmd




def xxxget_mmd_loss_function():

    def get_mmd(pred, true, latent_pred, mmd_weight = 1,):
        """Emprical maximum mean discrepancy. The lower the result
        the more evidence that distributions are the same.
        Derived From: 
            https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html#references
        MMD^2 = E[k(xx, xx)] + E[k(yy, yy)] - 2E[k(xx,yy)]

        Args:
            pred: decoded data
            true: true data
            latent_pred: first sample, distribution P
            kernel: kernel type such as "multiscale" or "rbf"
        """

        device = latent_pred.device
        recon_loss = torch.nn.MSELoss(reduction = "mean")(pred, true)
        x = latent_pred
        y = torch.randn(x.shape).to(device)

        # xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        # # create a matrix with columns set to repeats of each diagonal element
        # # [[1,2]  -> [[1,4]
        # #  [3,4]] ->  [1,4]]
        # rx = (xx.diag().unsqueeze(0).expand_as(xx)) 
        # ry = (yy.diag().unsqueeze(0).expand_as(yy))

        # # ||x - x'||^2
        # dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        # dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        # dxy = rx.t() + ry - 2. * xy # Used for C in (1)

        # ||x1 - x2||^2
        dxx = torch.cdist(x, x)**2
        dyy = torch.cdist(y, y)**2
        dxy = torch.cdist(x, y)**2

        XX, YY, XY = (torch.zeros(dxx.shape).to(device),
                      torch.zeros(dxx.shape).to(device),
                      torch.zeros(dxx.shape).to(device))


        # bandwidth_range = [10, 15, 20, 50]
        # for a in bandwidth_range:
        #     XX += torch.exp(-0.5*dxx/a)
        #     YY += torch.exp(-0.5*dyy/a)
        #     XY += torch.exp(-0.5*dxy/a)
        gamma_multiplier = 0.01
        n0 = x.shape[0] // 2 - ((x.shape[0] // 2) & 1)
        gamma = gamma_multiplier # * n0 * 2 / (torch.sum(dxx) + torch.sum(dyy) + 2 * torch.sum(dxy))
        # print(x.shape)
        
        XX += torch.exp(-0.5 * dxx * gamma)
        YY += torch.exp(-0.5 * dyy * gamma)
        XY += torch.exp(-0.5 * dxy * gamma)
        # print(2.*XY)

        MMD2 = torch.mean(XX + YY - 2. * XY)
        # print(MMD2)
                
        return recon_loss, mmd_weight * MMD2

    return get_mmd

