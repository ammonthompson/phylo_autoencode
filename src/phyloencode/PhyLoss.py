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

# TODO: experiment with time-dependent loss weights. Annealing strategies.

class PhyLoss(nn.Module):
    """Stateful composite loss for training phylogenetic autoencoders.

    This loss object computes a weighted sum of several components and stores both
    per-batch and per-epoch summaries for logging:

    - Structured reconstruction loss (tree/phy channels).
    - Character reconstruction loss (optional; categorical via cross-entropy or continuous via MSE).
    - Auxiliary reconstruction loss (MSE; optionally skipped when aux has a single column).
    - Latent regularization losses (MMD / VZ) when a latent representation is provided.

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
                 device = "auto",
                 validation  = False,
                 seed = None,
                 rand_matrix : torch.Tensor = None,) -> None:
        """Initialize a stateful loss accumulator.

        Args:
            weights (dict[str, torch.Tensor]): Mapping of component-weight names to scalar
                weights. Expected keys are:

                - ``"phy_loss_weight"``
                - ``"char_loss_weight"``
                - ``"aux_loss_weight"``
                - ``"mmd_loss_weight"``
                - ``"vz_loss_weight"``

            ntax_cidx (int): Column index in the auxiliary tensor corresponding to the
                number of taxa/tips (e.g. ``num_taxa``). Used for the separate num-tips loss.
            char_type (Optional[str]): Character type, typically ``"categorical"`` or
                ``"continuous"``. Defaults to None.
            latent_layer_Type (str): Latent layer type string (used for bookkeeping/logging).
                Defaults to ``"GAUSS"``.
            device (str): ``"auto"``, ``"cpu"``, or ``"cuda"``. If ``"auto"``, selects CUDA
                when available. Defaults to ``"auto"``.
            validation (bool): If True, ``print_epoch_losses()`` labels output as validation.
                Defaults to False.
            seed (Optional[int]): Optional seed used to initialize an internal Torch generator.
                Defaults to None.
            rand_matrix (Optional[torch.Tensor]): Optional matrix used by experimental losses.
                Defaults to None.
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
        
        # latent loss
        self.mmd = MMDLoss()
        # TODO: delete all VZLoss implementation. Nolonger in use.
        self.vz  = VZLoss(generator=self.torch_g)
        
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
            self.vz_w   = weights['vz_loss_weight']
        except KeyError as e:
            print(f"\nMissing loss weight parameter:\n {e}\n")
            raise

    @staticmethod
    def _to_float(value) -> float:
        """Convert scalar tensor/number to plain float."""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    @classmethod
    def _to_float_list(cls, values: list) -> list[float]:
        """Convert a list of scalar tensors/numbers to plain floats."""
        return [cls._to_float(v) for v in values]

    def to_checkpoint_dict(self) -> dict:
        """Serialize loss config and tracked history to a plain dict."""
        return {
            "config": {
                "weights": {
                    "phy_loss_weight": self._to_float(self.phy_w),
                    "char_loss_weight": self._to_float(self.char_w),
                    "aux_loss_weight": self._to_float(self.aux_w),
                    "mmd_loss_weight": self._to_float(self.mmd_w),
                    "vz_loss_weight": self._to_float(self.vz_w),
                },
                "ntax_cidx": int(self.ntax_cidx),
                "char_type": self.char_type,
                "latent_layer_type": self.latent_layer_type,
                "device": str(self.device),
                "validation": bool(self.validation),
                "seed": self.seed,
            },
            "history": {
                "epoch_total_loss": self._to_float_list(self.epoch_total_loss),
                "epoch_phy_loss": self._to_float_list(self.epoch_phy_loss),
                "epoch_char_loss": self._to_float_list(self.epoch_char_loss),
                "epoch_aux_loss": self._to_float_list(self.epoch_aux_loss),
                "epoch_mmd_loss": self._to_float_list(self.epoch_mmd_loss),
                "epoch_vz_loss": self._to_float_list(self.epoch_vz_loss),
                "batch_total_loss": self._to_float_list(self.batch_total_loss),
                "batch_phy_loss": self._to_float_list(self.batch_phy_loss),
                "batch_char_loss": self._to_float_list(self.batch_char_loss),
                "batch_aux_loss": self._to_float_list(self.batch_aux_loss),
                "batch_mmd_loss": self._to_float_list(self.batch_mmd_loss),
                "batch_vz_loss": self._to_float_list(self.batch_vz_loss),
            },
        }

    @classmethod
    def from_checkpoint_dict(cls, state: dict, device: Optional[str] = None) -> "PhyLoss":
        """Reconstruct a PhyLoss from ``to_checkpoint_dict()`` output."""
        config = state["config"]
        hist = state["history"]
        loss = cls(
            weights=config["weights"],
            ntax_cidx=config["ntax_cidx"],
            char_type=config["char_type"],
            latent_layer_Type=config["latent_layer_type"],
            device=device if device is not None else config["device"],
            validation=config["validation"],
            seed=config["seed"],
        )
        loss.epoch_total_loss = list(hist["epoch_total_loss"])
        loss.epoch_phy_loss = list(hist["epoch_phy_loss"])
        loss.epoch_char_loss = list(hist["epoch_char_loss"])
        loss.epoch_aux_loss = list(hist["epoch_aux_loss"])
        loss.epoch_mmd_loss = list(hist["epoch_mmd_loss"])
        loss.epoch_vz_loss = list(hist["epoch_vz_loss"])
        loss.batch_total_loss = list(hist["batch_total_loss"])
        loss.batch_phy_loss = list(hist["batch_phy_loss"])
        loss.batch_char_loss = list(hist["batch_char_loss"])
        loss.batch_aux_loss = list(hist["batch_aux_loss"])
        loss.batch_mmd_loss = list(hist["batch_mmd_loss"])
        loss.batch_vz_loss = list(hist["batch_vz_loss"])
        return loss

    def forward(self, pred : Tuple, true : Tuple, mask : Tuple):
        """Compute the weighted loss for a batch and update internal buffers.

        Expected tuple formats (as used by ``phyloencode.PhyloAutoencoder.PhyloAutoencoder``):

        - ``pred``: ``(phy_hat, char_hat, aux_hat, latent_hat)``
        - ``true``: ``(phy, char, aux, std_norm)``
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
            - The ``std_norm``/``target`` argument is currently ignored by the latent losses
              in this implementation (closed-form prior terms are used instead).

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
        phy, char, aux, std_norm = true
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
        """Aggregate the current batch buffers into epoch means and reset them."""
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

    def _mmd_loss(self, latent_pred, target = None):
        """Compute MMD-based latent regularization.

        Args:
            latent_pred (torch.Tensor): Latent tensor, shape ``(N, D)``.
            target (Optional[torch.Tensor]): Ignored in the current implementation (kept for
                backward compatibility).

        Returns:
            torch.Tensor: Scalar latent loss.
        """

        x = latent_pred
        y = None
        MMD = self.mmd(x, y)
                
        return MMD

    def _vz_loss(self, latent_pred, target = None):
        """Compute VZ latent regularization.

        Args:
            latent_pred (torch.Tensor): Latent tensor, shape ``(N, D)``.
            target (Optional[torch.Tensor]): Ignored in the current implementation (kept for
                backward compatibility).

        Returns:
            torch.Tensor: Scalar latent loss.
        """
        x = latent_pred
        y = None
        VZ = self.vz(x,y)

        return VZ

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
    def _estimate_base_bw(self, X: torch.Tensor) -> torch.Tensor:
        """Estimate a base RBF bandwidth from the batch.

        Uses pairwise squared distances and either a median or mean heuristic (configured by
        ``bw_mode``). If all off-diagonal distances are zero, falls back to a variance-based
        estimate.

        Args:
            X (torch.Tensor): Input tensor with shape ``(m, d)``.

        Returns:
            torch.Tensor: Scalar base bandwidth (same dtype/device as ``X``).
        """
        delta = 1e-8
        m = X.shape[0]

        # upper-triangular off-diagonal distances
        iu = torch.triu_indices(m, m, offset=1, device=X.device)
        vals = (torch.cdist(X, X) ** 2)[iu[0], iu[1]]
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
    def _bws_from_x(self, X: torch.Tensor) -> torch.Tensor:
        """Create the per-kernel bandwidth vector from a batch.

        Args:
            X (torch.Tensor): Input tensor with shape ``(m, d)``.

        Returns:
            torch.Tensor: Bandwidths with shape ``(K,)`` where ``K == n_kernels``.
        """
        base = self.fixed_bw.to(X.device, X.dtype) if self.fixed_bw is not None else self._estimate_base_bw(X)
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
        bws = self._bws_from_x(X)  # (K,)  bws = 2 * l_i^2 in Briol et al. 2025
        delta = 1e-6

        # Data–data term (unbiased) averaged over kernels
        #  E[k(x, x')] = 1/m(m-1) * sum_{i =\= j}^m(exp(-||x_i - x'_j||^2 / bw))
        L2_xx = torch.cdist(X, X) ** 2                       # (m, m)
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



class VZLoss(nn.Module):
    """Latent regularizer combining per-dimension MMD variance and decorrelation.

    The loss is:

    - ``var_j MMD^2(z_j, N(0, 1))`` (encourages similar marginal distributions across latent dims),
    - plus a correlation penalty on the latent covariance matrix (encourages decorrelation).

    This is primarily experimental and is usually disabled in typical training runs.
    """

    def __init__(self,
                 n_kernels: int = 5,
                 mul_factor: float = 2.0,
                 bw: float | torch.Tensor | None = None,
                 corr_weight: float = 0.1,
                 shuffle_pairs: bool = True,
                 generator: torch.Generator | None = None):
        """Initialize the VZ latent regularizer.

        Args:
            n_kernels (int, optional): Number of bandwidths in the geometric grid. Defaults to 5.
            mul_factor (float, optional): Ratio between adjacent bandwidths. Defaults to 2.0.
            bw (Optional[float or torch.Tensor]): Optional fixed base bandwidth. If a scalar, the
                same base is used for all latent dims; if a tensor, it must have shape ``(d,)``.
                If None, the base bandwidth is estimated from the batch as ``2 * Var(z_j)`` per
                dimension. Defaults to None.
            corr_weight (float, optional): Weight for the decorrelation penalty. Defaults to 0.1.
            shuffle_pairs (bool, optional): If True, randomly pairs samples for the linear-time
                MMD estimator. Defaults to True.
            generator (Optional[torch.Generator]): Optional RNG used when ``shuffle_pairs`` is True.
                Defaults to None.
        """
        
        super().__init__()

        self.corr_weight = corr_weight
        self.shuffle_pairs = shuffle_pairs
        self.generator = generator

        # bw multipliers (K,)
        self.register_buffer(
            "bw_multipliers",
            (mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)).float()
        )
        # Optional fixed base bandwidth (scalar or per-dim vector)
        self.fixed_bw = None if bw is None else torch.as_tensor(bw).float()

    @torch.no_grad()
    def _per_dim_bandwidths(self, z: torch.Tensor) -> torch.Tensor:  # (d, K)
        """Compute per-dimension, per-kernel bandwidths.

        Args:
            z (torch.Tensor): Latent batch with shape ``(B, d)``.

        Returns:
            torch.Tensor: Bandwidth tensor with shape ``(d, K)``.
        """
        B, d = z.shape
        K = self.bw_multipliers.numel()
        dtype = z.dtype
        device = z.device

        if self.fixed_bw is not None:
            base = self.fixed_bw.to(device=device, dtype=dtype)
            if base.ndim == 0:
                base = base.expand(d)                 # same base for all dims
            elif base.shape != (d,):
                raise ValueError(f"fixed bw must be scalar or shape (d,), got {tuple(base.shape)}")
        else:
            # 2 * Var(z_j) per dim (cheap, stable)
            base = 2.0 * z.var(dim=0, unbiased=False)
            base = base.clamp_min(1e-8)

        bws = base[:, None] * self.bw_multipliers[None, :].to(device=device, dtype=dtype)
        return bws  # (d, K)

    def forward(self, z: torch.Tensor, z_prime: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the VZ regularizer value.

        Args:
            z (torch.Tensor): Latent batch with shape ``(B, d)``.
            z_prime (Optional[torch.Tensor]): Ignored (kept for backward compatibility).

        Returns:
            torch.Tensor: Scalar loss ``var_j MMD^2(z_j, N(0,1)) + corr_weight * decorrelation``.
        """
        B, d = z.shape
        device = z.device
        K = self.bw_multipliers.numel()
        
        if B < 2:
            return z.sum() * 0.0  # correct dtype/device zero

        # Pairing for linear-time estimator
        idx = torch.randperm(B, device=device, generator=self.generator) if self.shuffle_pairs \
              else torch.arange(B, device=device)
        M = (B // 2) * 2
        if M < 2:
            return z.sum() * 0.0

        z_used = z[idx[:M]]
        z0, z1 = z_used[0::2], z_used[1::2]                 # (m, d), (m, d)
        m = z0.shape[0]

        diffs2 = (z0 - z1).pow(2)                           # (m, d)

        # Per-dim per-kernel bandwidths
        with torch.no_grad():
            bws_dk = self._per_dim_bandwidths(z)    # (d, K)
        bws_kd = bws_dk.T                                   # (K, d)

        # Linear-time data–data term per dim -
        # kxx_kmd = exp( - (z0 - z1)^2 / bw_kd )
        kxx_kmd = torch.exp(-diffs2.unsqueeze(0) / bws_kd[:, None, :])   # (K, m, d)
        term_xx_d = kxx_kmd.mean(dim=1).mean(dim=0)                       # (d,)

        # Prior–prior closed form per dim (d=1): (bw / (bw + 4))^(1/2)
        term_zz_d = (bws_kd / (bws_kd + 4.0)).pow(0.5).mean(dim=0)        # (d,)

        # Mixed closed form per dim: (bw/(bw+2))^(1/2) * E[exp(-x^2/(bw+2))] ---
        x2_all = torch.cat([z0, z1], dim=0).pow(2)                         # (2m, d)
        exp_term = torch.exp(-x2_all.unsqueeze(1) / (bws_kd.unsqueeze(0) + 2.0))  # (2m, K, d)
        mean_over_samples = exp_term.mean(dim=0)                           # (K, d)
        c1_kd = (bws_kd / (bws_kd + 2.0)).pow(0.5)                         # (K, d)
        term_xz_d = (c1_kd * mean_over_samples).mean(dim=0)                # (d,)

        mmd2_per_dim = term_xx_d + term_zz_d - 2.0 * term_xz_d             # (d,)
        var_mmd = mmd2_per_dim.var(unbiased=True)

        # Decorrelation penalty (scale-invariant)
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(B - 1, 1)                                  # (d, d)
        std = cov.diag().clamp_min(1e-8).sqrt()
        corr = cov / (std[:, None] * std[None, :])
        offdiag = corr - torch.diag(torch.diag(corr))
        corr_loss = (offdiag.pow(2).sum()) / (d * (d - 1) + 1e-8)

        return var_mmd + self.corr_weight * corr_loss


class exprm_VZLoss(nn.Module):
    """Latent regularizer that matches mean and covariance to ``N(0, I)``.

    This explicitly penalizes *full-covariance* rank collapse (e.g. a near-zero eigenvalue)
    that can slip through marginal/IMQ/RBF regularizers.

    By default it uses the Gaussian moment KL:

        ``0.5 * (tr(C) - logdet(C) - d)``

    where ``C`` is the batch covariance of ``z``. Optionally divides by ``d`` so the scale is
    roughly independent of latent dimension.
    """

    def __init__(self,
                 cov_weight: float = 1.0,
                 mean_weight: float = 1.0,
                 normalize_by_dim: bool = True,
                 eps: float = 1e-6,
                 unbiased_cov: bool = True,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.cov_weight = float(cov_weight)
        self.mean_weight = float(mean_weight)
        self.normalize_by_dim = bool(normalize_by_dim)
        self.eps = float(eps)
        self.unbiased_cov = bool(unbiased_cov)
        self.generator = generator  # kept for drop-in compatibility (unused)

    def forward(self, z: torch.Tensor, z_prime: torch.Tensor | None = None) -> torch.Tensor:

        B, d = z.shape

        if B < 2:
            return z.sum() * 0.0  # correct dtype/device zero

        # Compute in fp32 for numerical stability (keep gradients).
        z_work = z.float() if z.dtype in (torch.float16, torch.bfloat16) else z

        mu = z_work.mean(dim=0)
        zc = z_work - mu

        denom = (B - 1) if self.unbiased_cov else B
        denom = max(int(denom), 1)
        cov = (zc.T @ zc) / denom
        cov = 0.5 * (cov + cov.T)  # enforce symmetry

        eye = torch.eye(d, device=z_work.device, dtype=z_work.dtype)
        cov_eps = cov + self.eps * eye

        # Stable logdet via Cholesky on the jittered covariance.
        # (Fallback to eigvals if Cholesky fails for numerical reasons.)
        try:
            chol = torch.linalg.cholesky(cov_eps)
            logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        except RuntimeError:
            eigvals = torch.linalg.eigvalsh(cov).clamp_min(self.eps)
            logdet = eigvals.log().sum()

        cov_kl = 0.5 * (cov.diag().sum() - logdet - float(d))
        mean_pen = 0.5 * mu.pow(2).sum()

        if self.normalize_by_dim:
            cov_kl = cov_kl / float(d)
            mean_pen = mean_pen / float(d)

        loss = self.cov_weight * cov_kl + self.mean_weight * mean_pen
        return loss




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
