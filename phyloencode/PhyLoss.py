#!/usr/bin/env python3
import torch
import torch
import torch.nn as nn
import torch.nn.functional as fun
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import itertools



class PhyLoss(object):
    # holds component losses over epochs
    # to be called for an individual batch
    # methods:
        # compute, stores, and returns component and total loss for val and train sets
        # plots loss curves
    def __init__(self, weights : torch.Tensor, rand_matrix : torch.Tensor, char_type = None, latent_layer_Type = "GAUSS" ):
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
        # loss weights TODO: make a dictionary
        self.phy_w, self.char_w, self.aux_w, self.mmd_w, self.vz_w  = weights

        self.char_type = char_type
        self.latent_layer_type = latent_layer_Type
        
        self.rand_matrix = rand_matrix # K x D, where K is the number of latent dims and D is the data dimension
        

    def minibatch_loss(self, x : Tuple, y : Tuple, mask : Tuple):
        # appends to the self.losses and returns the current batch loss
        # x is a tuple (dictionary maybe?) of predictions for a batch
        # y is a tuple of the true values for a batch
        # both contain: 
            # phy, char, mask, aux, ntips

        phy_loss, char_loss, aux_loss, mmd_loss, vz_loss = self.losses(x, y, mask, self.char_type)

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


# Loss functions
    def losses(self, pred : torch.tensor, true : torch.tensor, mask : torch.tensor, char_type : str):
        # separate components
        phy_hat, char_hat, aux_hat, latent_hat = pred
        phy, char, aux, std_norm = true
        tree_mask, char_mask = mask

        device = phy_hat.device

        # recon loss
        phy_loss    = self.phy_recon_loss(phy_hat, phy, mask = tree_mask)
        char_loss   = self.char_recon_loss(char_hat, char, char_type, char_mask) if char is not None else torch.tensor(0.).to(device)
        aux_loss    = self.aux_recon_loss(aux_hat, aux)

        # latent loss
        latent_mmd_loss    = self.mmd_loss(latent_hat, std_norm) if latent_hat is not None else torch.tensor(0.).to(device)
        latent_vz_loss     = self.vz_loss(latent_hat, std_norm)  if latent_hat is not None else torch.tensor(0.).to(device)

        return phy_loss, char_loss, aux_loss, latent_mmd_loss, latent_vz_loss

    def mmd_loss(self, latent_pred, target = None):

        device = latent_pred.device
        x = latent_pred
        if target is None:
            y = torch.randn(x.shape).to(device)
        else:
            y = target[0:x.shape[0],:]
        MMD = MMDLoss(device)(x, y)
                
        return MMD

    def vz_loss(self, latent_pred, target = None):
        device = latent_pred.device
        x = latent_pred
        if target is None:
            y = torch.randn(x.shape).to(device)
        else:
            y = target[0:x.shape[0],:]
        VZ = VZLoss(device)(x, y)

        return VZ

    def rdp_loss(self, Z_batch: torch.Tensor, X_batch: torch.Tensor, rand_matrix: torch.Tensor, n_pairs: int):
        """
        Computes the average Random Distance Prediction (RDP) loss for a batch
        by sampling 2 * n_pairs distinct indices and splitting them into i and j.
        This guarantees i != j for each pair, and distinct indices across the sampled set.
        Described here: 10.48550/arXiv.1912.12186
        """
        batch_size = Z_batch.shape[0]
        X_batch = X_batch.permute(0, 2, 1).reshape(batch_size, -1)  # equivalent to np.reshape(X_batch, (batch_size, -1), order = 'F')

        max_possible_pairs = batch_size // 2
        num_pairs = min(n_pairs, max_possible_pairs)

        # Ensure n_pairs doesn't exceed half the batch size, I need 2*n_pairs distinct indices.
        if 2 * num_pairs > batch_size:
            raise ValueError(
                f"Cannot sample {num_pairs} distinct pairs from a batch of size {batch_size}. "
                f"Need at least {2 * num_pairs} unique indices. Reduce n_pairs or increase batch_size."
            )

        # Generate 2 * n_pairs unique indices without replacement from the batch
        # This ensures all indices selected for i and j are distinct.
        flat_rand_idxs = torch.randperm(batch_size, device=Z_batch.device)[:2 * num_pairs]

        # Split these distinct indices into two sets for i and j
        indices_i = flat_rand_idxs[:num_pairs]
        indices_j = flat_rand_idxs[num_pairs:]
        # Rest of the loss computation
        z_i = Z_batch[indices_i, ...]
        z_j = Z_batch[indices_j, ...]
        x_i = X_batch[indices_i, ...]
        x_j = X_batch[indices_j, ...]

        latent_inner_products = torch.sum(z_i * z_j, dim=1)

        eta_x_i = torch.matmul(x_i, rand_matrix.T)
        eta_x_j = torch.matmul(x_j, rand_matrix.T)
        projected_inner_products = torch.sum(eta_x_i * eta_x_j, dim=1)
        
        K_latent_dim = Z_batch.shape[1] # Assuming K is the second dimension of Z_batch
        
        # # Divide inner products by K
        scaled_latent_inner_products = latent_inner_products / K_latent_dim
        scaled_projected_inner_products = projected_inner_products / K_latent_dim

        # # Calculate loss using the scaled inner products
        loss = ((scaled_latent_inner_products - scaled_projected_inner_products)**2).mean()

        # EXPERIMENTAL: L^ad_aux = (||z_i - eta_x_i||^2 + ||z_j - eta_x_j||^2).mean()
        ad_aux_loss = ((torch.norm(z_i - eta_x_i, dim=1) ** 2 + torch.norm(z_j - eta_x_j, dim=1) ** 2)/K_latent_dim).mean()

        return 0.75 * loss + 0.25 * ad_aux_loss

    def phy_recon_loss(self, x, y, mask = None):
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

        tip1_loss = 0.1 * fun.mse_loss(x[:,0:2,0], y[:,0:2,0]) 

        return tree_loss + tip1_loss

    def char_recon_loss(self, x, y, char_type = "categorical", mask = None):
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

            # Convert one-hot to class indices
            y = y.argmax(dim=1)  # shape (N,)
            
            char_loss = fun.cross_entropy(x, y, reduction = 'none')
        else:
            char_loss = fun.mse_loss(x, y, reduction = 'none')

        if mask is not None:
            # Apply the mask to the loss
            mask = mask.permute(0, 2, 1).reshape(-1, mask.shape[1])
            char_loss = char_loss * mask[:,0] # match dims (all columns are the same in mask)

        return char_loss.mean()

    def aux_recon_loss(self, x, y):
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

    def set_weights(self, weights : torch.Tensor):
        """
        Set the weights for the loss components. 
        Args:
            weights (torch.Tensor): A tensor containing the new weights for phy, char, aux, mmd, and vz losses.
        """
        self.phy_w  = weights['phy_weight']  if 'phy_weight'  in weights else self.phy_w
        self.char_w = weights['char_weight'] if 'char_weight' in weights else self.char_w
        self.aux_w  = weights['aux_weight']  if 'aux_weight'  in weights else self.aux_w
        self.mmd_w  = weights['mmd_weight']  if 'mmd_weight'  in weights else self.mmd_w
        self.vz_w   = weights['vz_weight']   if 'vz_weight'   in weights else self.vz_w

# classes for MMD2 loss
# Encourage latent space to be be N(0,1) distributed
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
    
    
# Might not be using anymore
    def recon_loss(self, x, y, 
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
                tree_loss = self.phy_recon_loss(x, y, mask)
                char_loss = torch.tensor(0.0)
                char_weight = 0.
            else:
                tree_loss = self.phy_recon_loss(x[:,0:(x.shape[1]-num_chars)], 
                                        y[:,0:(x.shape[1]-num_chars)], 
                                        mask[:,0:(x.shape[1]-num_chars)])
                char_loss = self.char_recon_loss(x[:,(x.shape[1]-num_chars):], 
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