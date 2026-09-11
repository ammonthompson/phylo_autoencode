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
    """Compute the composite objective for a phylogenetic autoencoder.

    Padding masks are constructed on the input device from raw per-sample tip
    counts. The returned objective remains connected to autograd, while the
    component values are returned separately for metric tracking:

    - Structured reconstruction loss (tree/phy channels).
    - Optional character reconstruction loss, using cross-entropy for
      categorical characters or MSE for continuous characters.
    - Auxiliary reconstruction loss (MSE).
    - A separate reconstruction loss for the normalized num-tips field.
    - MMD latent regularization when a latent representation is provided.
    """
    def __init__(self, 
                 weights : dict[str, float | torch.Tensor],
                 ntax_cidx : int,
                 char_type : str = None, 
                 mmd_num_kernels: int = 3) -> None:
        """Initialize the composite loss.

        Args:
            weights (dict[str, float | torch.Tensor]): Mapping of component
                names to scalar weights. Expected keys are:

                - ``"phy_loss_weight"``
                - ``"char_loss_weight"``
                - ``"aux_loss_weight"``
                - ``"mmd_loss_weight"``

            ntax_cidx (int): Column index of ``num_taxa`` in the normalized
                auxiliary target and prediction tensors.
            char_type (Optional[str]): ``"categorical"`` or ``"continuous"``.
                Defaults to None.
            mmd_num_kernels (int): Positive odd number of RBF bandwidths used by MMD.
                Defaults to 3.
        """

        
        super().__init__()

        self.ntax_cidx = ntax_cidx

        # loss weights TODO: make a dictionary
        self.set_weights(weights)

        self.char_type = char_type
        
        # latent loss
        self.mmd = MMDLoss(n_kernels=mmd_num_kernels)
        
    def set_weights(self, weights : dict[str, float | torch.Tensor]):
        """Set loss weights from a mapping.

        Args:
            weights (dict[str, float | torch.Tensor]): Mapping containing the
                required keys documented by ``__init__``. Values must be
                floats or scalar tensors.

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

    def forward(
         self,
         pred: Tuple,
         true: Tuple,
         num_tips: torch.Tensor
        ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted objective and its component metrics for a batch.

        Args:
            pred (Tuple): ``(phy_hat, char_hat, aux_hat, latent_hat)`` model
                outputs. Structured tensors have shape ``(N, C, W)``, auxiliary
                tensors have shape ``(N, A)``, and the optional latent tensor
                has shape ``(N, D)``.
            true (Tuple): ``(phy, char, aux)`` targets. ``char`` may be None.
            num_tips (torch.Tensor): Raw, unnormalized integer tip counts with
                shape ``(N,)``. This tensor must be on the same device as
                ``phy``.

        Returns:
            Tuple[torch.Tensor, dict[str, torch.Tensor]]: Scalar objective used
                for backpropagation and a mapping containing ``total``, ``phy``,
                ``char``, ``aux``, and ``mmd`` metric tensors.

        Notes:
            A compact mask with shape ``(N, 1, W)`` is created on the input
            device and broadcast across channels. Tree and continuous-character
            losses are averaged over valid tips and channels; categorical
            cross-entropy is averaged over valid tips. The separate num-tips
            loss uses the normalized auxiliary field and is included in the
            objective under the phylogenetic loss weight.
        """
        # separate components
        phy_hat, char_hat, aux_hat, latent_hat = pred
        phy, char, aux = true

        mask = self._mask_from_num_tips(num_tips=num_tips, phy=phy)
        # tree_mask, char_mask = mask

        zero = phy_hat.new_zeros(())

        # recon loss
        phy_loss    = self._phy_recon_loss(phy_hat, phy, mask, num_tips)
        char_loss   = self._char_recon_loss(char_hat, char, mask, num_tips) \
            if char is not None and self.char_w > 0. else zero
        aux_loss    = self._aux_recon_loss(aux_hat, aux) \
            if aux_hat is not None and self.aux_w > 0. else zero
        ntips_loss  = self._num_tips_recon_loss(aux_hat[:, self.ntax_cidx], aux[:, self.ntax_cidx]) 

        # latent loss
        mmd_loss = self._mmd_loss(latent_hat) \
            if latent_hat is not None and self.mmd_w > 0. else zero

        # computed weighted total loss
        total_loss = self.phy_w  * phy_loss  + \
                     self.char_w * char_loss + \
                     self.aux_w  * aux_loss  + \
                     self.phy_w  * ntips_loss + \
                     self.mmd_w  * mmd_loss

        
        metrics = {
            "total": total_loss,
            "phy": phy_loss,
            "char": char_loss,
            "aux": aux_loss,
            "mmd": mmd_loss,
        }
        return total_loss, metrics

    def _phy_recon_loss(self, x, y, mask, num_tips):
        """Compute reconstruction loss for structured phylogenetic channels.

        Args:
            x (torch.Tensor): Reconstructed tree tensor with shape
                ``(N, C_tree, W)``.
            y (torch.Tensor): Ground-truth tree tensor with the same shape.
            mask (Optional[torch.Tensor]): Boolean padding mask with shape
                ``(N, 1, W)``, broadcast across tree channels. If None, all
                elements contribute to the loss.
            num_tips (torch.Tensor): Raw tip counts with shape ``(N,)``, used
                with the channel count to normalize each sample.

        Returns:
            torch.Tensor: Scalar mean of the per-sample reconstruction losses.
        """
        # if cblv-like data, use the first two channels in the loss
        # else if augmented cblv-like data, use the first four channels in the data

        if mask is None:
            batch_mean_tree_loss = fun.mse_loss(x, y, reduction = "mean") 
        else:
            tree_mse = (
                (fun.mse_loss(x, y, reduction='none') * mask).sum(dim=(1, 2))
                / (num_tips * x.shape[1])
            )
            batch_mean_tree_loss = tree_mse.mean()

        # tip1_loss = 0.1 * fun.mse_loss(x[:,0:2,0], y[:,0:2,0]) 

        return batch_mean_tree_loss #+ tip1_loss

    def _char_recon_loss(self, x, y, mask, num_tips):
        """Compute reconstruction loss for character channels.

        Args:
            x (torch.Tensor): Predicted character tensor with shape
                ``(N, C_char, W)``. Categorical values are interpreted as
                logits.
            y (torch.Tensor): Target character tensor with the same shape.
                Categorical targets are represented across the channel axis.
            mask (Optional[torch.Tensor]): Boolean padding mask with shape
                ``(N, 1, W)``. If None, all positions contribute to the loss.
            num_tips (torch.Tensor): Raw tip counts with shape ``(N,)``.

        Returns:
            torch.Tensor: Scalar mean of the per-sample character losses.

        Notes:
            Continuous MSE is averaged over valid tips and character channels.
            Categorical cross-entropy has already reduced the channel axis and
            is therefore averaged only over valid tips.
        """
        if self.char_type == "categorical": 
            y_max_idx = y.argmax(dim=1)
            char_loss = fun.cross_entropy(x, y_max_idx, reduction = 'none') 
        else:
            char_loss = fun.mse_loss(x, y, reduction = 'none')

        if mask is not None:
            if char_loss.ndim == mask.ndim:
                loss_mask = mask
                reduce_dims = (1, 2)
                denominator = num_tips * x.shape[1]
            else:
                # Cross entropy has already reduced the categorical channel axis.
                loss_mask = mask[:, 0, :]
                reduce_dims = (1,)
                denominator = num_tips
            pb_char_loss = (
                (char_loss * loss_mask).sum(dim=reduce_dims)
                / denominator.clamp_min(1)
            )
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
        aux_loss = fun.mse_loss(x, y) if x.shape[1] > 1 else x.new_zeros(())
        # return fun.mse_loss(x, y)
        return aux_loss

    def _mask_from_num_tips(self, num_tips: torch.Tensor,
                            phy: torch.Tensor) -> torch.Tensor:
        """Create a compact padding mask on the structured tensor's device.

        Args:
            num_tips (torch.Tensor): Raw tip counts with shape ``(N,)``.
            phy (torch.Tensor): Structured reference tensor with shape
                ``(N, C, W)``. Only its width and device are used.

        Returns:
            torch.Tensor: Boolean mask with shape ``(N, 1, W)``. Valid tip
                positions are True and padded positions are False.
        """
        positions = torch.arange(phy.shape[-1], device=phy.device)
        return positions[None, None, :] < num_tips[:, None, None]
    
    def _num_tips_recon_loss(self, x, y):
        """Compute MSE for the normalized num-tips auxiliary field.

        Args:
            x (torch.Tensor): Predicted normalized values with shape ``(N,)``.
            y (torch.Tensor): Target normalized values with shape ``(N,)``.

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

    Args:
        n_kernels (int): Positive odd number of RBF kernels. Defaults to 5.
        mul_factor (float): Multiplicative spacing between adjacent bandwidths.
            Defaults to 2.
        bw (Optional[float]): Fixed base bandwidth. When omitted, the bandwidth
            is estimated from the current batch.
        bw_mode (str): ``"median"`` or ``"mean"`` bandwidth heuristic.
            Defaults to ``"median"``.
        max_bw_pairs (int): Maximum number of pairwise distances used to
            estimate the bandwidth. Smaller batches use every unique pair.
            Defaults to 131,072.

    Notes:
        This implementation returns ``sqrt(MMD^2)`` (clipped to a small epsilon for numerical
        stability).

    References:
        Briol et al. (2025), ``10.48550/arXiv.2504.18830``.
    """
    DEFAULT_MAX_BW_PAIRS = 131_072

    def __init__(self,
                 n_kernels: int = 5,
                 mul_factor: float = 2.0,
                 bw: float | None = None,
                 bw_mode: str = "median",
                 max_bw_pairs: int = DEFAULT_MAX_BW_PAIRS):
        super().__init__()
        if n_kernels < 1 or n_kernels % 2 == 0:
            raise ValueError("mmd_num_kernels must be a positive odd integer")
        if max_bw_pairs < 1:
            raise ValueError("max_bw_pairs must be positive")
        self.bw_mode = bw_mode
        self.max_bw_pairs = max_bw_pairs
        self.register_buffer(
            "bw_multipliers",
            (mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)).float()
        )
        self.fixed_bw = None if bw is None else torch.tensor(float(bw))

    @property
    def num_kernels(self) -> int:
        return self.bw_multipliers.numel()

    @torch.no_grad()
    def _estimate_base_bw(self, X: torch.Tensor, L2_xx: torch.Tensor) -> torch.Tensor:
        """Estimate a base RBF bandwidth from the batch.

        Uses pairwise squared distances and either a median or mean heuristic
        (configured by ``bw_mode``). All unique pairs are used for small
        batches. For large batches, at most ``max_bw_pairs`` uniformly sampled
        off-diagonal pairs are used. If all selected distances are zero, the
        estimate falls back to the total variance.

        Args:
            X (torch.Tensor): Input tensor with shape ``(m, d)``.
            L2_xx (torch.Tensor): Pairwise squared distances for ``X``, with shape
                ``(m, m)``.

        Returns:
            torch.Tensor: Scalar base bandwidth (same dtype/device as ``X``).
        """
        delta = 1e-8
        m = X.shape[0]

        num_unique_pairs = m * (m - 1) // 2
        max_bw_pairs = getattr(
            self, "max_bw_pairs", self.DEFAULT_MAX_BW_PAIRS
        )

        if num_unique_pairs <= max_bw_pairs:
            pair_idx = torch.triu_indices(m, m, offset=1, device=X.device)
            vals = L2_xx[pair_idx[0], pair_idx[1]]
        else:
            first = torch.randint(m, (max_bw_pairs,), device=X.device)
            second = torch.randint(m - 1, (max_bw_pairs,), device=X.device)
            second += second >= first
            vals = L2_xx[first, second]

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
