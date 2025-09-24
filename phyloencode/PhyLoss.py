#!/usr/bin/env python3
import torch
import torch
import torch.nn as nn
import torch.nn.functional as fun
from typing import List, Dict, Tuple, Optional, Union
# import matplotlib.pyplot as plt
import random
# import numpy as np

# TODO: implement abstract autoencoder loss class. PhyLoss should be a child of this class.
# in addition to forward, should perform: phy_recon_loss, aux_recon_loss, char_recon_loss, ntips_recon_loss, latent_loss
#   these _loss functions should take in pred, true, weight and
#   should return a tuple (weighted loss, unweighted loss)
# additionaly: abstract fields and methods for setting, getting component losses


class PhyLoss(nn.Module):
    """Statefull. performs loss calculation and holds batch and 
        epoch mean losses for all components.
    """
    # holds component losses over epochs
    # to be called for an individual batch
    # methods:
        # compute, stores, and returns component and total loss for val and train sets
        # plots loss curves
    def __init__(self, 
                 weights : dict[str, torch.Tensor], 
                 ntax_cidx : int,
                 char_type : str = None, 
                 latent_layer_Type = "GAUSS",
                 device = "auto",
                 validation  = False,
                 seed = None,
                 rand_matrix : torch.Tensor = None,) -> None:
        """Contains component losses and methods for computing component losses.

        Args:
            weights (dict[str, torch.Tensor]): weights [0, inf] for component losses
            ntax_cidx (int): column index of aux_data that contains \"num_taxa\"
            char_type (str, optional):["categorical", "continuous"]. Defaults to None.
            latent_layer_Type (str, optional): _description_. Defaults to "GAUSS".
            device (str, optional): _description_. Defaults to "cpu".
            validation (bool, optional): _description_. Defaults to False.
            rand_matrix (torch.Tensor, optional): _description_. Defaults to None.
        """

        
        super().__init__()

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # self.np_rng = None
        self.torch_g = None
        self.seed = seed 
        if self.seed is not None:
            self.torch_g = torch.Generator(self.device).manual_seed(self.seed)
        
        self.validation = validation
        self.ntax_cidx = ntax_cidx
        # weights for all components
        # initialize component loss vectors for train and validation losses

        # epoch losses are the average of the batch losses
        # epoch losses
        # TODO: put these in dictionaries (more generic)
        self.epoch_total_loss   = []
        self.epoch_phy_loss     = []
        self.epoch_char_loss    = []
        self.epoch_aux_loss     = []
        self.epoch_mmd_loss     = []
        self.epoch_vz_loss      = []
        # batch losses 
        self.batch_total_loss   = []
        self.batch_phy_loss     = []
        self.batch_char_loss    = []
        self.batch_aux_loss     = []
        self.batch_mmd_loss     = []
        self.batch_vz_loss      = []

        # loss weights TODO: make a dictionary
        # self.phy_w, self.char_w, self.aux_w, self.mmd_w, self.vz_w  = weights
        self.set_weights(weights)

        self.char_type = char_type
        self.latent_layer_type = latent_layer_Type
        
        # TODO: think this was part of rdp_loss experiments, prob should delete at some point
        self.rand_matrix = rand_matrix # K x D, where K is the number of latent dims and D is the data dimension

        # latent loss
        # note, these two losses make use of two different samples of y from N(0, I)
        self.mmd = MMDLoss(self.device)
        self.vz  = VZLoss(self.device)
        
    def set_weights(self, weights : dict[str, torch.Tensor]):
        """
        Set the weights for the loss components. 
        Args:
            weights (torch.Tensor): A tensor containing the new weights for phy, char, 
                                    aux, mmd, and vz losses.
        """
        try:
            self.phy_w  = weights['phy_loss_weight']
            self.char_w = weights['char_loss_weight']
            self.aux_w  = weights['aux_loss_weight']
            self.mmd_w  = weights['mmd_loss_weight']
            self.vz_w   = weights['vz_loss_weight']
        except KeyError as e:
            print(f"\nMissing loss weight parameter:\n {e}\n")
            raise

    def forward(self, pred : Tuple, true : Tuple, mask : Tuple):
        # appends to the self.losses and returns the current batch loss
        # pred is a tuple (dictionary maybe?) of predictions for a batch
        # true is a tuple of the true values for a batch
        # both contain: 
            # phy, char, mask, aux, ntips

        # separate components
        phy_hat, char_hat, aux_hat, latent_hat = pred
        phy, char, aux, std_norm = true
        tree_mask, char_mask = mask

        device = phy_hat.device

        # TODO: Note: the aux_loss also contains a num_taxa loss, so loss for numtips in 2 places. No big deal in my opinon.
        # recon loss
        phy_loss    = self._phy_recon_loss(phy_hat, phy, mask = tree_mask)
        char_loss   = self._char_recon_loss(char_hat, char, char_mask) \
            if char is not None and self.char_w > 0. else torch.tensor(0.).to(device)
        aux_loss    = self._aux_recon_loss(aux_hat, aux)
        ntips_loss  = self._num_tips_recon_loss(aux_hat[:, self.ntax_cidx], aux[:, self.ntax_cidx])

        # latent loss
        mmd_loss = self._mmd_loss(latent_hat, std_norm) \
            if latent_hat is not None and self.mmd_w > 0. else torch.tensor(0.).to(device)
        vz_loss  = self._vz_loss(latent_hat, std_norm)  \
            if latent_hat is not None and self.vz_w > 0. else torch.tensor(0.).to(device)

        # computed weighted total loss
        total_loss =    self.phy_w  * phy_loss  + \
                        self.char_w * char_loss + \
                        self.aux_w  * aux_loss  + \
                        self.phy_w  * ntips_loss + \
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
        self.batch_mmd_loss     = []
        self.batch_vz_loss      = []

    def print_epoch_losses(self, elapsed_time):
        if self.validation:
            loss_type = "Val loss:   "
        else:
            print(f"Epoch {len(self.epoch_total_loss)}")
            loss_type = "Train loss: "

        print(  f"\t {loss_type}{self.epoch_total_loss[-1]:.4f},  " +
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

    def _phy_recon_loss(self, x, y, mask = None):
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
        # TODO: instead of computing mask.sum() you can pass in num_tips 
        # TODO: 
        if mask is None:
            batch_mean_tree_loss = fun.mse_loss(x, y, reduction = "mean") 
        else:
            tree_mse = ((fun.mse_loss(x, y, reduction='none') * mask).sum(dim=(1,2)) / 
                         mask.sum(dim = (1,2))) 
            batch_mean_tree_loss = tree_mse.mean()

        tip1_loss = 0.1 * fun.mse_loss(x[:,0:2,0], y[:,0:2,0]) 

        return batch_mean_tree_loss + tip1_loss

    def _char_recon_loss(self, x, y, mask = None):
        """
        Compute the reconstruction loss for categorical character data.

        Args:
        # TODO: fix this doc string (num_chars)
            x (torch.Tensor): Reconstructed character data logits.
            y (torch.Tensor): Original character data.
            is_categorical (bool): Whether the character data is categorical.
            mask : tips that are just padding

        Returns:
            torch.Tensor: Reconstruction loss for character data.
        """
        # TODO: instead of computing mask.sum() you can pass in num_tips 

        if self.char_type == "categorical": 
            y_max_idx = y.argmax(dim=1)
            char_loss = fun.cross_entropy(x, y_max_idx, reduction = 'none') 
        else:
            char_loss = fun.mse_loss(x, y, reduction = 'none')

        if mask is not None:
            pb_char_loss = (char_loss * mask[:,0,:] ).sum(dim=1) / mask[:,0,:].sum(dim=1)# match dims (all columns are the same in mask)
        else:
            pb_char_loss = char_loss

        return pb_char_loss.mean()

    def _aux_recon_loss(self, x, y):
        """
        Compute the reconstruction loss for auxiliary data.       

        Args:
            x (torch.Tensor): Reconstructed auxiliary data.
            y (torch.Tensor): Original auxiliary data.

        Returns:
            torch.Tensor: Reconstruction loss for auxiliary data.
        """
        # Use mean squared error for auxiliary data
        aux_loss = fun.mse_loss(x, y) if x.shape[1] > 1 else torch.tensor(0.).to(x.device)
        # return fun.mse_loss(x, y)
        return aux_loss
    
    def _num_tips_recon_loss(self, x, y):
        # TODO: ordinal loss
        return fun.mse_loss(x, y)

    # TODO: implement _latent_loss
    def _latent_loss(self, latent_pred, std_norm, target = None):
        pass
        
    def _mmd_loss(self, latent_pred, target = None):

        x = latent_pred
        if target is None:
            y = torch.randn(x.shape, requires_grad=False, 
                            device=x.device, generator=self.torch_g)
        else:
            y = target[0:x.shape[0],:]
        MMD = self.mmd(x, y)
                
        return MMD

    def _vz_loss(self, latent_pred, target = None):
        x = latent_pred
        if target is None:
            y = torch.randn(x.shape, requires_grad=False, 
                            device=x.device, generator=self.torch_g)
        else:
            y = target[0:x.shape[0],:]
        VZ = self.vz(x,y)

        return VZ



    # Experimental
    def _rdp_loss(self, Z_batch: torch.Tensor, X_batch: torch.Tensor, rand_matrix: torch.Tensor, n_pairs: int):
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


# classes for MMD2 loss
# Encourage latent space to be be N(0,1) distributed
class RBF(nn.Module):
    ''' Derived From: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py'''
    def __init__(self, device, n_kernels=5, mul_factor=2., bw=None):
        super().__init__()
        self.device = device
        # Build a set of bandwidth multipliers: mul_factor^(k - floor(n_kernels/2)), 
        # k = 0..n_kernels-1
        # This centers the exponent range so you get symmetric scales around 1.0 
        # (e.g., for 5 and mul_factor=2: [1/4,1/2,1,2,4]).
        self.bw_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bw_multipliers = self.bw_multipliers.to(self.device)
        if bw != None:
            self.bw = torch.tensor(bw).to(self.device)
        else:
            self.bw = bw

    def get_bw(self, L2_dists):
        # returns the constant self.bw if set. Otherwise estimates from distances
        if self.bw is None:
            # Heuristic bandwidth: average pairwise squared distance across the matrix,
            # excluding the diagonal (n*(n-1) off-diagonal pairs if X=Y).
            n_samples = L2_dists.shape[0]

            # Note: uses .data to avoid autograd tracking of this statistic.
            avg_L2_dist = L2_dists.data.sum() / (n_samples * (n_samples - 1))
            return avg_L2_dist

        return self.bw

    def forward(self, X, Y = None):
        # X: (N_x, D), Y: (N_y, D) optional. If Y is None, 
        # compute all pairwise distances within X.
        # Output: (N_x, N_y) RBF kernel matrix (or (N_x, N_x) if Y is None), 
        # summed over a set of bandwidths.
        if Y is None:
             # Pairwise Euclidean distances; square to get squared L2 distances.
            L2_dists = torch.cdist(X, X) ** 2
        else:
            L2_dists = torch.cdist(X, Y) ** 2

        # Base bandwidth (scalar) times a vector of multipliers -> vector of bandwidths.
        # Shape after [:, None, None] is (K, 1, 1) 
        # so it can broadcast over the (N_x, N_y) distance matrix.

        # Compute exp(-||x - y||^2 / bw_k) for each k (adds a leading K axis via [None, ...]),
        # then sum over K to form a multi-kernel RBF (no normalization).

        bw = self.get_bw(L2_dists)

        sf = (bw * self.bw_multipliers)[:, None, None]

        # return torch.exp(-L2_dists[None, ...] / sf).sum(dim=0)
        return torch.exp(-L2_dists[None, ...] / sf).mean(dim=0)

class MMDLoss(nn.Module):
    ''' Derived from: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py'''
    def __init__(self, device):
        super().__init__()
        self.kernel = RBF(device, n_kernels=5)

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        k_xx = K[:X_size, :X_size]
        k_xy = K[:X_size, X_size:]
        k_yy = K[X_size:, X_size:]

        m = X_size
        n = K.shape[0] - m

        sum_xx = k_xx.fill_diagonal_(0).sum()
        sum_yy = k_yy.fill_diagonal_(0).sum()

        MMD2_loss = sum_xx / (m*(m-1)) + sum_yy / (n*(n-1)) - 2 * k_xy.mean()

        # MMD_loss = torch.sqrt(MMD2_loss.clamp_min(0.0) + 1e-12) 

        # return MMD_loss
        return MMD2_loss
        # return MMD2_loss.clamp_min(0.0)
    
class VZLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.kernel = RBF(device, n_kernels=5)
    
    def forward(self, X, Y):
        return self.kernel(X.T, Y.T).var()
    



# storage
class xxx_MMDLoss(nn.Module):
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

        m = X_size
        n = K.shape[0] - m
        sum_xx = (k_xx.sum() - k_xx.diag().sum())
        sum_yy = (k_yy.sum() - k_yy.diag().sum())

        MMD2_loss = (sum_xx / (m * (m - 1) + 1e-12)
              + sum_yy / (n * (n - 1) + 1e-12)
              - 2.0 * k_xy.mean())

        return MMD2_loss.clamp_min(0.0)
    
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