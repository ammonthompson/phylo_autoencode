#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as fun
from typing import Tuple
# import matplotlib.pyplot as plt
# import numpy as np

# TODO: implement abstract autoencoder loss class. PhyLoss should be a child of this class.
# in addition to forward, should perform: phy_recon_loss, aux_recon_loss, char_recon_loss, ntips_recon_loss, latent_loss
#   these _loss functions should take in pred, true, weight and
#   should return a tuple (weighted loss, unweighted loss)
# additionaly: abstract fields and methods for setting, getting component losses

# TODO: experiment with time-dependent loss weights. Annealing strategies.

class PhyLoss(nn.Module):
    """Stateful composite loss for training phylogenetic autoencoders.

    This loss object computes a weighted sum of several components and stores both
    per-batch and per-epoch summaries for logging:

    - Structured reconstruction loss (tree/phy channels).
    - Character reconstruction loss (optional; categorical via cross-entropy or continuous via MSE).
    - Auxiliary reconstruction loss (MSE; optionally skipped when aux has a single column).
    - MMD latent regularization when a latent representation is provided.

    The ``forward()`` method appends component losses to internal ``batch_*`` buffers. Call
    ``append_mean_batch_loss()`` once per epoch to compute epoch means and clear the batch
    buffers. Use ``print_epoch_losses()`` to print the latest epoch summary.
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
                 validation  = False) -> None:
        """Initialize a stateful loss accumulator.

        Args:
            weights (dict[str, torch.Tensor]): Mapping of component-weight names to scalar
                weights. Expected keys are:

                - ``"phy_loss_weight"``
                - ``"char_loss_weight"``
                - ``"aux_loss_weight"``
                - ``"mmd_loss_weight"``

            ntax_cidx (int): Column index in the auxiliary tensor corresponding to the
                number of taxa/tips (e.g. ``num_taxa``). Used for the separate num-tips loss.
            char_type (Optional[str]): Character type, typically ``"categorical"`` or
                ``"continuous"``. Defaults to None.
            latent_layer_Type (str): Latent layer type string (used for bookkeeping/logging).
                Defaults to ``"GAUSS"``.
            validation (bool): If True, ``print_epoch_losses()`` labels output as validation.
                Defaults to False.
        """

        
        super().__init__()

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
        # batch losses 
        self.batch_total_loss   = []
        self.batch_phy_loss     = []
        self.batch_char_loss    = []
        self.batch_aux_loss     = []
        self.batch_mmd_loss     = []

        # loss weights TODO: make a dictionary
        self.set_weights(weights)

        self.char_type = char_type
        self.latent_layer_type = latent_layer_Type
        
        # latent loss
        self.mmd = MMDLoss(n_kernels=3)
        
    def set_weights(self, weights : dict[str, torch.Tensor]):
        """Set loss weights from a mapping.

        Args:
            weights (dict[str, torch.Tensor]): Mapping containing the required weight keys
                (see ``__init__``). Values should be scalar tensors or floats.

        Raises:
            KeyError: If any required weight key is missing.
        """
        try:
            self.phy_w  = weights['phy_loss_weight']
            self.char_w = weights['char_loss_weight']
            self.aux_w  = weights['aux_loss_weight']
            self.mmd_w  = weights['mmd_loss_weight']
        except KeyError as e:
            print(f"\nMissing loss weight parameter:\n {e}\n")
            raise

    def forward(self, pred : Tuple, true : Tuple, mask : Tuple):
        """Compute the weighted loss for a batch and update internal buffers.

        Expected tuple formats (as used by ``phyloencode.PhyloAutoencoder.AETrainer``):

        - ``pred``: ``(phy_hat, char_hat, aux_hat, latent_hat)``
        - ``true``: ``(phy, char, aux)``
        - ``mask``: ``(tree_mask, char_mask)``

        Shapes:
            - ``phy_hat`` and ``phy``: ``(N, C_tree, W)``
            - ``char_hat`` and ``char``: ``(N, C_char, W)`` or None
            - ``aux_hat`` and ``aux``: ``(N, A)``
            - ``latent_hat``: ``(N, D)`` or None
            - masks: boolean tensors matching the corresponding structured tensors

        Notes:
            - A separate num-tips loss is computed from ``aux_hat[:, ntax_cidx]`` and
              ``aux[:, ntax_cidx]`` and is added to the total loss. It is not stored as a
              separate tracked component.
        Args:
            pred (Tuple): Model outputs for the batch.
            true (Tuple): Ground-truth targets for the batch.
            mask (Tuple): ``(tree_mask, char_mask)`` masks (each element may be None).

        Returns:
            torch.Tensor: Scalar loss tensor for backpropagation.
        """
        # appends to the self.losses and returns the current batch loss
        # pred is a tuple (dictionary maybe?) of predictions for a batch
        # true is a tuple of the true values for a batch
        # both contain: 
            # phy, char, mask, aux, ntips

        # separate components
        phy_hat, char_hat, aux_hat, latent_hat = pred
        phy, char, aux = true
        tree_mask, char_mask = mask

        device = phy_hat.device

        # TODO: Note: the aux_loss also contains a num_taxa loss, so loss for numtips in 2 places. 
        # dont think it matters.
        # recon loss
        phy_loss    = self._phy_recon_loss(phy_hat, phy, mask = tree_mask)
        char_loss   = self._char_recon_loss(char_hat, char, char_mask) \
            if char is not None and self.char_w > 0. else torch.tensor(0.).to(device)
        aux_loss    = self._aux_recon_loss(aux_hat, aux) \
            if aux_hat is not None and self.aux_w > 0. else torch.tensor(0.).to(device)
        ntips_loss  = self._num_tips_recon_loss(aux_hat[:, self.ntax_cidx], aux[:, self.ntax_cidx]) 

        # latent loss
        mmd_loss = self._mmd_loss(latent_hat) \
            if latent_hat is not None and self.mmd_w > 0. else torch.tensor(0.).to(device)

        # computed weighted total loss
        total_loss =    self.phy_w  * phy_loss  + \
                        self.char_w * char_loss + \
                        self.aux_w  * aux_loss  + \
                        self.phy_w  * ntips_loss + \
                        self.mmd_w  * mmd_loss

        
        self._append_minibatch_losses(total_loss, phy_loss, char_loss, aux_loss, mmd_loss)

        return total_loss

    def append_mean_batch_loss(self):
        """Aggregate the current batch buffers into epoch means and reset them."""
        # averages the batch loss arrays and return 
        mean_total_loss = torch.mean(torch.stack(self.batch_total_loss)).item()
        mean_phy_loss   = torch.mean(torch.stack(self.batch_phy_loss)).item()
        mean_char_loss  = torch.mean(torch.stack(self.batch_char_loss)).item()
        mean_aux_loss   = torch.mean(torch.stack(self.batch_aux_loss)).item()
        mean_mmd_loss   = torch.mean(torch.stack(self.batch_mmd_loss)).item()

        self._append_epoch_losses(mean_total_loss, mean_phy_loss, mean_char_loss, 
                                  mean_aux_loss, mean_mmd_loss)
        
        # reset batch losses
        self.batch_total_loss   = []
        self.batch_phy_loss     = []
        self.batch_char_loss    = []
        self.batch_aux_loss     = []
        self.batch_mmd_loss     = []

    def print_epoch_losses(self, elapsed_time):
        """Print the most recent epoch loss summary.

        Args:
            elapsed_time (float): Elapsed time for the epoch, in seconds.
        """
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
                f"Run time: {elapsed_time:.3f} sec" )

    # helpers
    def _append_minibatch_losses(self, total_loss, phy_loss, 
                                 char_loss, aux_loss, mmd_loss):
        self.batch_total_loss.append(total_loss)
        self.batch_phy_loss.append(phy_loss)
        self.batch_char_loss.append(char_loss)
        self.batch_aux_loss.append(aux_loss)
        self.batch_mmd_loss.append(mmd_loss)

    def _append_epoch_losses(self, total_loss, phy_loss, 
                             char_loss, aux_loss, mmd_loss):
        self.epoch_total_loss.append(total_loss)
        self.epoch_phy_loss.append(phy_loss)
        self.epoch_char_loss.append(char_loss)
        self.epoch_aux_loss.append(aux_loss)
        self.epoch_mmd_loss.append(mmd_loss)

    def _phy_recon_loss(self, x, y, mask = None):
        """Compute reconstruction loss for structured phylogenetic channels.

        Args:
            x (torch.Tensor): Reconstructed structured tensor, shape ``(N, C_tree, W)``.
            y (torch.Tensor): Ground-truth structured tensor, shape ``(N, C_tree, W)``.
            mask (Optional[torch.Tensor]): Optional boolean/0-1 mask with the same shape as
                ``x``/``y``. When provided, loss is averaged over unmasked elements per sample.

        Returns:
            torch.Tensor: Scalar batch-mean reconstruction loss.

        Notes:
            In addition to the masked/unmasked MSE, an extra penalty term is added on the
            first position of the first two channels (``0.1 * MSE(x[:, 0:2, 0], y[:, 0:2, 0])``).
        """
        # if cblv-like data, use the first two channels in the loss
        # else if augmented cblv-like data, use the first four channels in the data

        if mask is None:
            batch_mean_tree_loss = fun.mse_loss(x, y, reduction = "mean") 
        else:
            tree_mse = ((fun.mse_loss(x, y, reduction='none') * mask).sum(dim=(1,2)) / 
                         mask.sum(dim = (1,2))) 
            batch_mean_tree_loss = tree_mse.mean()

        # tip1_loss = 0.1 * fun.mse_loss(x[:,0:2,0], y[:,0:2,0]) 

        return batch_mean_tree_loss #+ tip1_loss

    def _char_recon_loss(self, x, y, mask = None):
        """Compute reconstruction loss for character channels.

        Args:
            x (torch.Tensor): Predicted character tensor, shape ``(N, C_char, W)``. For
                categorical characters this is interpreted as logits.
            y (torch.Tensor): Target character tensor, shape ``(N, C_char, W)``. For categorical
                characters this is typically one-hot (or a probability simplex) over channels.
            mask (Optional[torch.Tensor]): Optional boolean/0-1 mask with shape ``(N, C_char, W)``
                indicating which tips are present (padding tips are False). When provided, the
                mask from the first channel is used as a per-tip mask.

        Returns:
            torch.Tensor: Scalar batch-mean character reconstruction loss.
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
        """Compute reconstruction loss for auxiliary features.

        Args:
            x (torch.Tensor): Predicted auxiliary tensor, shape ``(N, A)``.
            y (torch.Tensor): Target auxiliary tensor, shape ``(N, A)``.

        Returns:
            torch.Tensor: Scalar MSE loss. If ``A == 1``, returns 0 (the single column is
            typically the num-tips field, which is handled separately).
        """
        # Use mean squared error for auxiliary data
        aux_loss = fun.mse_loss(x, y) if x.shape[1] > 1 else torch.tensor(0.).to(x.device)
        # return fun.mse_loss(x, y)
        return aux_loss
    
    def _num_tips_recon_loss(self, x, y):
        """Compute reconstruction loss for the num-tips auxiliary field.

        Args:
            x (torch.Tensor): Predicted num-tips values, shape ``(N,)`` or ``(N, 1)``.
            y (torch.Tensor): Target num-tips values, shape ``(N,)`` or ``(N, 1)``.

        Returns:
            torch.Tensor: Scalar MSE loss.
        """
        # TODO: ordinal loss
        return fun.mse_loss(x, y)

    def _mmd_loss(self, latent_pred):
        """Compute MMD-based latent regularization.

        Args:
            latent_pred (torch.Tensor): Latent tensor, shape ``(N, D)``.
        Returns:
            torch.Tensor: Scalar latent loss.
        """

        return self.mmd(latent_pred)

# this version uses deterministic kernel values for each term involving standard normal comparison
class MMDLoss(nn.Module):
    """Multi-kernel RBF MMD loss against a standard normal prior.

    Computes a Maximum Mean Discrepancy (MMD) between a batch ``X`` and ``N(0, I)`` using an
    RBF kernel ``k(x, y) = exp(-||x - y||^2 / bw)`` and closed-form expectations for the prior
    terms (no Monte Carlo sampling of the prior needed).

    The base bandwidth is either fixed or estimated from the batch, then expanded into a
    geometric grid of bandwidths (multi-kernel MMD).

    Notes:
        This implementation returns ``sqrt(MMD^2)`` (clipped to a small epsilon for numerical
        stability).

    References:
        Briol et al. (2025), ``10.48550/arXiv.2504.18830``.
    """
    def __init__(self,
                 n_kernels: int = 5,
                 mul_factor: float = 2.0,
                 bw: float | None = None,
                 bw_mode: str = "median"):
        super().__init__()
        self.bw_mode = bw_mode
        self.register_buffer(
            "bw_multipliers",
            (mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)).float()
        )
        self.fixed_bw = None if bw is None else torch.tensor(float(bw))

    @torch.no_grad()
    def _estimate_base_bw(self, X: torch.Tensor, L2_xx: torch.Tensor) -> torch.Tensor:
        """Estimate a base RBF bandwidth from the batch.

        Uses pairwise squared distances and either a median or mean heuristic (configured by
        ``bw_mode``). If all off-diagonal distances are zero, falls back to a variance-based
        estimate.

        Args:
            X (torch.Tensor): Input tensor with shape ``(m, d)``.
            L2_xx (torch.Tensor): Pairwise squared distances for ``X``, with shape
                ``(m, m)``.

        Returns:
            torch.Tensor: Scalar base bandwidth (same dtype/device as ``X``).
        """
        delta = 1e-8
        m = X.shape[0]

        # upper-triangular off-diagonal distances
        iu = torch.triu_indices(m, m, offset=1, device=X.device)
        # vals = (torch.cdist(X, X) ** 2)[iu[0], iu[1]]
        vals = L2_xx[iu[0], iu[1]]
        vals = vals[vals > 0]

        if vals.numel() > 0:
            if self.bw_mode == "mean":
                base = vals.mean()
            else:  # "median"
                base = torch.quantile(vals, 0.5)
        else:
            base = X.var(dim=0, unbiased=False).sum()

        return base.to(X.device, X.dtype).clamp(min=delta)


    @torch.no_grad()
    def _bws_from_x(self, X: torch.Tensor, L2_xx: torch.Tensor) -> torch.Tensor:
        """Create the per-kernel bandwidth vector from a batch.

        Args:
            X (torch.Tensor): Input tensor with shape ``(m, d)``.
            L2_xx (torch.Tensor): Pairwise squared distances for ``X``, with shape
                ``(m, m)``.

        Returns:
            torch.Tensor: Bandwidths with shape ``(K,)`` where ``K == n_kernels``.
        """
        base = self.fixed_bw.to(X.device, X.dtype) if self.fixed_bw is not None else self._estimate_base_bw(X, L2_xx=L2_xx)
        return base * self.bw_multipliers.to(X.device, X.dtype)  # (K,)

    def forward(self, X: torch.Tensor, Y: torch.Tensor | None = None) -> torch.Tensor:
        """Compute MMD between ``X`` and ``N(0, I)``.

        Args:
            X (torch.Tensor): Latent batch with shape ``(m, d)``.
            Y (Optional[torch.Tensor]): Ignored (kept for backward compatibility). Older
                versions accepted a sampled prior batch here.

        Returns:
            torch.Tensor: Scalar MMD value (``sqrt(MMD^2)``).
        """
        m, d = X.shape
        delta = 1e-6

        # Data–data term (unbiased) averaged over kernels
        #  E[k(x, x')] = 1/m(m-1) * sum_{i =\= j}^m(exp(-||x_i - x'_j||^2 / bw))
        L2_xx = torch.cdist(X, X) ** 2                       # (m, m)
        bws = self._bws_from_x(X, L2_xx)  # (K,)  bws = 2 * l_i^2 in Briol et al. 2025
        Kxx_k = torch.exp(-L2_xx[None, :, :] / bws[:, None, None])  # (K, m, m)
        Kxx = Kxx_k.mean(dim=0)                              # average over K -> (m, m)
        Kxx.fill_diagonal_(0.0)
        mean_Kxx = Kxx.sum() / (m * (m - 1))                # avgerage over all distances

        # see Briol et al. (2025) https://doi.org/10.48550/arXiv.2504.18830 equation 15
        # "A Dictionary of Closed-Form Kernel Mean Embeddings"

        # Prior–prior closed form: 
        #   E[k(z,z')] = (bw / (bw + 4))^(d/2) 
        # Note: this only depends on bws which is not optimized by SGD (in torch.no_grad context)
        # Note: since l_i^2 = bws/2;   l_i^2 / (l_i^2 + 2 * sigma^2) = bws/(bws + 4)
        mean_Kzz = (bws / (bws + 4.0)).pow(d / 2.0).mean()

        # Mixed closed form (semi-analytic): 
        #   E[k(x, z)] = (bw/(bw+2))^(d/2) * exp(-||x||^2 / (bw + 2)) 
        c1 = (bws / (bws + 2.0)).pow(d / 2.0)                # (K,)
        x2 = (X * X).sum(dim=1, keepdim=True)                # (m, 1)
        Ez_xz = c1 * torch.exp(-x2 / (bws + 2.0))            # (m, K) via broadcasting
        mean_Kxz = Ez_xz.mean()                               # mean over samples and kernels

        mmd2 = mean_Kxx + mean_Kzz - 2.0 * mean_Kxz
        # return mmd2
        return torch.sqrt(mmd2.clamp(min=delta))



    # DEVELOPMENT


class MMD_IMQ_Loss(nn.Module):
    """Multi-kernel IMQ MMD loss against a standard normal prior.

    Uses an inverse multiquadratic kernel
        k(x, y) = c / (c + ||x - y||^2)
    with a geometric grid of ``c`` values (multi-kernel MMD). The prior batch is sampled as
    ``z ~ N(0, I)`` with the same shape as ``X``.

    Returns ``sqrt(MMD^2)`` (clipped to a small epsilon for stability), matching the API and
    output behavior of ``MMDLoss`` above (``Y`` is accepted but ignored).
    """
    def __init__(self,
                 n_kernels: int = 5,
                 mul_factor: float = 2.0,
                 bw: float | None = None,
                 bw_mode: str = "median"):
        super().__init__()
        self.bw_mode = bw_mode
        self.register_buffer(
            "bw_multipliers",
            (mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)).float()
        )
        self.fixed_bw = None if bw is None else torch.tensor(float(bw))

    @staticmethod
    def _sq_cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x2 = (x * x).sum(dim=1, keepdim=True)
        y2 = (y * y).sum(dim=1, keepdim=True).T
        # numerical clamp avoids tiny negatives from fp roundoff
        return (x2 + y2 - 2.0 * (x @ y.T)).clamp_min(0.0)

    @torch.no_grad()
    def _estimate_base_bw_from_l2(self, L2_xx: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        delta = 1e-8
        m = X.shape[0]

        iu = torch.triu_indices(m, m, offset=1, device=X.device)
        vals = L2_xx[iu[0], iu[1]]
        vals = vals[vals > 0]

        if vals.numel() > 0:
            if self.bw_mode == "mean":
                base = vals.mean()
            else:  # "median"
                base = torch.quantile(vals, 0.5)
        else:
            base = X.var(dim=0, unbiased=False).sum()

        return base.to(device=X.device, dtype=X.dtype).clamp(min=delta)

    def forward(self, X: torch.Tensor, Y: torch.Tensor | None = None) -> torch.Tensor:
        m, d = X.shape
        delta = 1e-8

        if m < 2:
            return X.sum() * 0.0

        # Sample prior (no grad needed)
        Z = torch.randn_like(X)

        # Pairwise squared distances (avoid torch.cdist sqrt)
        L2_xx = self._sq_cdist(X, X)
        L2_zz = self._sq_cdist(Z, Z)
        L2_xz = self._sq_cdist(X, Z)
        L2_xx.fill_diagonal_(0.0)
        L2_zz.fill_diagonal_(0.0)

        # Kernel scales (treated as constants for stability/speed)
        with torch.no_grad():
            base = self.fixed_bw.to(X.device, X.dtype) if self.fixed_bw is not None \
                   else self._estimate_base_bw_from_l2(L2_xx.detach(), X)
            cs = base * self.bw_multipliers.to(X.device, X.dtype)  # (K,)

        # IMQ kernel: c / (c + ||x - y||^2)
        denom_xx = cs[:, None, None] + L2_xx[None, :, :]
        denom_zz = cs[:, None, None] + L2_zz[None, :, :]
        denom_xz = cs[:, None, None] + L2_xz[None, :, :]

        Kxx_k = cs[:, None, None] / denom_xx  # (K, m, m)
        Kzz_k = cs[:, None, None] / denom_zz  # (K, m, m)
        Kxz_k = cs[:, None, None] / denom_xz  # (K, m, m)

        # Unbiased within-sample terms (exclude diagonal); diag entries are 1 after L2 diag = 0
        sum_offdiag_xx = Kxx_k.sum(dim=(1, 2)) - float(m)
        sum_offdiag_zz = Kzz_k.sum(dim=(1, 2)) - float(m)
        mean_Kxx = sum_offdiag_xx.mean() / (m * (m - 1))
        mean_Kzz = sum_offdiag_zz.mean() / (m * (m - 1))

        # Mixed term uses all pairs
        mean_Kxz = Kxz_k.mean()

        mmd2 = mean_Kxx + mean_Kzz - 2.0 * mean_Kxz
        return torch.sqrt(mmd2.clamp(min=delta))
