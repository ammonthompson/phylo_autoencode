#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
# from torch import optim
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from phyloencode import utils
# from phyloencode.PhyLoss import PhyLoss
from phyloencode.PhyLoss import PhyLoss
from phyloencode.DataProcessors import AEData
from phyloencode.PhyloAEModel import AECNN
import phyloencode.utils as utils
import time
import random
import importlib
# import os
from typing import List, Dict, Tuple, Optional, Union


class PhyloAutoencoder(object):
    """Train and evaluate a phylogenetic autoencoder.

    This class is a lightweight training loop around an ``phyloencode.PhyloAEModel.AECNN``
    (or compatible) autoencoder. It handles:

    - Device placement and optional global seeding for reproducibility.
    - Epoch-based training and validation over PyTorch DataLoader instances.
    - Tracking/printing losses via stateful loss objects (e.g. ``phyloencode.PhyLoss.PhyLoss``).

    Notes:
        The data loaders are expected to yield either ``(phy, aux)`` or ``(phy, aux, mask)``:

        - ``phy`` is a ``torch.Tensor`` shaped ``(batch, channels, width)``.
        - ``aux`` is a ``torch.Tensor`` shaped ``(batch, aux_dim)``.
        - ``mask`` (optional) is a ``torch.bool`` tensor with the same shape as ``phy``.

        If ``model.num_chars > 0``, the last ``num_chars`` channels of ``phy`` (and ``mask``, if
        present) are treated as character channels and are separated from the tree channels
        before loss computation.

    Attributes:
        device (str): ``"cuda"`` or ``"cpu"``.
        model (AECNN): Autoencoder model being optimized.
        optimizer: PyTorch optimizer used during training.
        lr_sched: Optional learning-rate scheduler stepped once per batch.
        train_loader (Optional[torch.utils.data.DataLoader]): Training data loader.
        val_loader (Optional[torch.utils.data.DataLoader]): Validation data loader.
        total_epochs (int): Total epochs trained so far (cumulative across ``train()`` calls).
        train_loss: Loss object used for training batches.
        val_loss: Loss object used for validation batches.
    """

    def __init__(self,
                 model: AECNN, 
                 optimizer : torch.optim.Optimizer, 
                 *, 
                 lr_scheduler : Optional[torch.optim.lr_scheduler.LRScheduler] = None, 
                 batch_size : Optional[int] = 128, 
                 train_loss : Optional[PhyLoss] = None, 
                 val_loss : Optional[PhyLoss] = None, 
                 seed : Optional[int] = None, 
                 device : Optional[str] = "auto",
                 track_grad : Optional[bool] = False,
                 checkpoints : Optional[list[int]] = None,
                 checkpt_file_prefix : Optional[str] = "train_out"):
        """Initialize the training loop.

        Args:
            model (AECNN): Autoencoder model to train.
            optimizer: Instantiated PyTorch optimizer (e.g. ``torch.optim.AdamW``) configured
                with ``model.parameters()``.
            lr_scheduler: Optional learning-rate scheduler with a ``.step()`` method.
                If provided, it is stepped once per training batch (not per epoch).
            batch_size (int, optional): Expected batch size. This is used for a few derived
                shapes (e.g. ``latent_shape``) and does not enforce DataLoader behavior.
                Defaults to 128.
            train_loss: Loss object/callable used during training. It must be callable as
                ``train_loss(pred, true, segmented_mask)`` and return a scalar tensor used for
                backprop. ``segmented_mask`` is a ``(tree_mask, char_mask)`` tuple (each element
                may be None). For logging, it is expected to provide
                ``.append_mean_batch_loss()`` and ``.print_epoch_losses(...)``.
            val_loss: Loss object/callable used during validation. It is called like
                ``val_loss(pred, true, segmented_mask)`` and is expected to track/print losses
                similarly to ``train_loss``.
            seed (int, optional): If provided, seeds Python, NumPy, and PyTorch RNGs and
                enables deterministic cuDNN behavior for reproducibility. Defaults to None.
            device (str, optional): ``"auto"``, ``"cuda"``, or ``"cpu"``. If ``"auto"``, selects
                CUDA when available. Defaults to ``"auto"``.
            track_grad (bool, optional): If True, record per-parameter gradient norms during
                training so they can be plotted after the run. Defaults to False.
            checkpoints (list[int], optional): Epoch numbers to save checkpoints. defaults to None.
        """
        
        # TODO: define the model object better (autoencoder ...)
        # TODO: run checks that the model has the expected attributes
        # TODO: add track_grad to parameters
        # TODO: add checks that the loss objects are correct (contain certain fields and methods)
        # TODO: fix checkpoints so starting w/pre-trained network can be used easilly.
        # TODO: Batch_size is a data loader attribute. Prob dont need for this constructor.
        # TODO: finish implementing checkpoint saves. 

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # This class sets module level rng behavior through the set_seed method. 
        # the Generator object is, as of 9/19/2025 only used for std_norm random variates in _mini_batch.

        self.torch_g = None
        self.seed = seed 
        if self.seed is not None:
            self.torch_g = torch.Generator(self.device).manual_seed(self.seed)
            self.set_seed(self.seed)

        self.batch_size   = batch_size
        self.epoch = 0
        self.train_loader = None
        self.val_loader   = None
        self.model        = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.lr_sched  = lr_scheduler
        self.checkpoints = checkpoints
        self.checkpt_file_prefix = checkpt_file_prefix

        # TODO: implement tracking model.state_dict for best validation score 
        self.best_model   = model.state_dict()


        # some data shape parameters
        self.nchars             = self.model.num_chars
        self.char_type          = self.model.char_type
        self.phy_channels       = self.model.num_structured_input_channel
        self.num_tree_chans     = self.phy_channels - self.nchars
        # TODO: # self.latent_shape is I think nolonger needed.
        self.latent_shape       = (self.batch_size * 2, self.model.latent_outwidth) 

        self.train_loss = train_loss
        self.val_loss   = val_loss

        self.track_grad = False
        self.batch_layer_grad_norm = None
        self.mean_layer_grad_norm = None
        self.set_track_grad(track_grad)


    def train(self, num_epochs, seed = None):
        """Train the model for a number of epochs.

        This method requires that ``train_loader`` and ``train_loss`` have been set
        (typically via ``set_data_loaders()`` and ``set_losses()``).

        If ``val_loader`` and ``val_loss`` are set, a validation pass is run after each epoch.

        Args:
            num_epochs (int): Number of epochs to train for.
            seed (int, optional): Optional seed to (re-)seed RNGs before training.
                If ``None``, the current RNG state is used. Defaults to None.

        Raises:
            ValueError: If training data has not been loaded.
            ValueError: If training loss has not been set.
        """
        

        if self.train_loader is None:
            raise ValueError("Must load training data.")
        if self.train_loss is None:
            raise ValueError("Must load loss layer.")

        # If both `seed` and `self.seed` are None, then the module-level RNG state governs
        # all random number generation.
        self.set_seed(seed if seed is not None else self.seed)

        for epoch in range(self.epoch + 1, num_epochs):
            self.epoch = epoch #bookeeping
            epoch_time = time.time()

            # target latent distribution sample
            # self.std_norm = torch.randn(self.latent_shape, device=self.device, generator=self.torch_g) \
            #     if self.model.latent_layer_type == "GAUSS" else None
            self.std_norm = None


            # perform all mini batch steps for the epoch for training data
            self._mini_batch(validation=False)
            self.train_loss.append_mean_batch_loss()
            # print training epoch mean losses to screen
            self.train_loss.print_epoch_losses(elapsed_time=time.time() - epoch_time)

            # perform mini batch on validation data
            if self.val_loader is not None and self.val_loss is not None:
                with torch.no_grad():
                    self._mini_batch(validation=True)             
                    self.val_loss.append_mean_batch_loss()
                    # print epoch mean component losses to screen       
                    self.val_loss.print_epoch_losses(elapsed_time = time.time() - epoch_time)

            # save checkpoints
            if self.checkpoints != None:
                if self.epoch in self.checkpoints:
                    self.save_checkpoint(self.checkpt_file_prefix + "_epoch_" + str(self.epoch) + ".ckpt.pt")

       
    def _mini_batch(self, validation = False):
        """Run one full pass over a data loader (train or validation).

        Iterates through ``train_loader`` or ``val_loader`` and calls the appropriate step
        function on each batch (``_train_step`` for training, ``evaluate`` for validation).

        Args:
            validation (bool, optional): If True, uses ``val_loader`` and ``evaluate()``.
                If False, uses ``train_loader`` and ``_train_step()``. Defaults to False.

        Returns:
            None: This method updates loss state as a side-effect.
        """
        # 

        # set data loader and step function
        if validation:
            data_loader   = self.val_loader
            step_function = self.evaluate
        else:
            data_loader   = self.train_loader
            step_function = self._train_step

        if data_loader == None:
            return None

        # perform step and return loss
        # loop through all batches of train or val data
        for batch in data_loader:
            if len(batch) == 3:
                phy_batch, aux_batch, mask_batch = batch
                mask_batch = mask_batch.to(self.device)
            else:
                phy_batch, aux_batch = batch
                mask_batch = None

            phy_batch = phy_batch.to(self.device)
            aux_batch = aux_batch.to(self.device)
               
            # target latent distribution sample
            # self.std_norm = torch.randn(self.latent_shape, device=self.device, generator=self.torch_g) \
            #     if self.model.latent_layer_type == "GAUSS" else None

            # perform SGD step for batch
            step_function(phy_batch, aux_batch, mask_batch, self.std_norm)

        # compute mean of batch grad norms per layer
        if self.track_grad and not validation:
            for k,v in self.batch_layer_grad_norm.items():
                self.mean_layer_grad_norm[k].append(np.mean(v))
                self.batch_layer_grad_norm[k].clear()

    
    def _train_step(self, phy: torch.Tensor, aux: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None, std_norm : Optional[torch.Tensor] = None):
        """Run a single gradient update on one batch.

        Args:
            phy (torch.Tensor): Structured input tensor shaped ``(batch, channels, width)``.
            aux (torch.Tensor): Unstructured/auxiliary input tensor shaped ``(batch, aux_dim)``.
            mask (torch.Tensor, optional): Optional boolean mask shaped like ``phy``. Defaults
                to None.
            std_norm (torch.Tensor, optional): Optional sample from a target latent
                distribution (e.g. standard normal) used by some latent losses. Defaults to
                None.
        """
        # batch train loss
        # set model to train mode
        self.model.train()

        # divide phy and mask into tree and character data
        tree, char, tree_mask, char_mask = self._split_tree_char(phy, mask)

        segmented_mask = (tree_mask, char_mask)          
        true = (tree, char, aux, std_norm)
        pred = self.model((phy, aux))

        # compute and update loss fields in train_loss
                
        loss = self.train_loss(pred, true, segmented_mask)

        # compute gradient
        loss.backward()

        # record gradient for assessments
        if self.track_grad:
            self._record_grad_norm()

        # update model paremeters with gradient
        self.optimizer.step()

        # Clear the gradient for the next iteration so it doesnt accumulate 
        self.optimizer.zero_grad()

        # update learning rate according to schedule
        if self.lr_sched != None and self.lr_sched._step_count <= self.lr_sched.state_dict()['total_steps']:
            # TODO delete commented code
            # print(self.lr_sched._step_count, self.lr_sched.state_dict()['total_steps']) 
            self.lr_sched.step()

        
    def evaluate(self, phy: torch.Tensor, aux: torch.Tensor,
                  mask: Optional[torch.Tensor] = None, std_norm: Optional[torch.Tensor] = None):
        """Evaluate the model on one batch and update validation loss state.

        This method does not disable gradients by itself; call it under
        ``torch.no_grad()`` during evaluation.

        Args:
            phy (torch.Tensor): Structured input tensor shaped ``(batch, channels, width)``.
            aux (torch.Tensor): Unstructured/auxiliary input tensor shaped ``(batch, aux_dim)``.
            mask (torch.Tensor, optional): Optional boolean mask shaped like ``phy``. Defaults
                to None.
            std_norm (torch.Tensor, optional): Optional sample from a target latent
                distribution used by some latent losses. Defaults to None.
        """
        
        # batch val loss        
        self.model.eval()

        # divide phy into tree and character data
        tree, char, tree_mask, char_mask = self._split_tree_char(phy, mask)

        segmented_mask = (tree_mask, char_mask)
        true = (tree, char, aux, std_norm)
        pred = self.model((phy, aux))

        # compute and update loss fields in val_loss
        self.val_loss(pred, true, segmented_mask)
        
    def predict(self, phy: torch.Tensor, aux: torch.Tensor, *,
                inference = False, detach = False) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct inputs with the current model.

        This is a thin wrapper around ``model.predict(...)``.

        Args:
            phy (torch.Tensor): Structured input tensor shaped ``(batch, channels, width)``.
            aux (torch.Tensor): Unstructured/auxiliary input tensor shaped ``(batch, aux_dim)``.
            inference (bool, optional): If True, runs in eval mode and disables gradients in
                the underlying model method. Defaults to False.
            detach (bool, optional): If True and ``inference`` is True, detaches outputs
                before returning. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(phy_pred, aux_pred)`` reconstructed outputs as
            NumPy arrays on CPU.
        """

        return self.model.predict(phy, aux, inference = inference, detach = detach)


    def to_device(self, device):
        """Move the model and this trainer to a new device.

        Args:
            device (str): ``"cpu"`` or ``"cuda"``.
        """
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            print(f"Didn't work, sending to {self.device} instead.")

    def set_data_loaders(self, train_loader : torch.utils.data.DataLoader, 
                               val_loader   : Optional[torch.utils.data.DataLoader] = None):
        """Set the training and validation data loaders.

        The loaders must yield either ``(phy, aux)`` or ``(phy, aux, mask)`` as described in
        the class-level docstring.

        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader.
            val_loader (torch.utils.data.DataLoader, optional): Validation data loader.
                Defaults to None.

        Raises:
            TypeError: If either loader is not a ``torch.utils.data.DataLoader``.
        """
        
        self.train_loader = train_loader
        self.val_loader   = val_loader

        if not isinstance(self.train_loader, torch.utils.data.DataLoader):
            raise TypeError(f"train_loader must be a DataLoader, got {type(self.train_loader).__name__}.")

        if self.val_loader is not None and not isinstance(self.val_loader, torch.utils.data.DataLoader):
            raise TypeError(f"val_loader must be a DataLoader, got {type(self.val_loader).__name__}.")


    def load_losses(self, train_loss, val_loss):
        """Set the loss objects used for training and validation.

        Args:
            train_loss: Loss object/callable used during training. It must be callable as
                ``train_loss(pred, true, segmented_mask)``, where ``segmented_mask`` is a
                ``(tree_mask, char_mask)`` tuple.
            val_loss: Loss object/callable used during validation, called like ``train_loss``.
        """
        self.train_loss = train_loss
        self.val_loss = val_loss

    def set_track_grad(self, track_grad: bool = False):
        """Enable or disable gradient-norm tracking for future training steps."""
        self.track_grad = bool(track_grad)
        if self.track_grad:
            self.batch_layer_grad_norm = {
                layer_name: [] for layer_name, _ in self.model.named_parameters()
            }
            self.mean_layer_grad_norm = {
                layer_name: [] for layer_name, _ in self.model.named_parameters()
            }
        else:
            self.batch_layer_grad_norm = None
            self.mean_layer_grad_norm = None

    def _record_grad_norm(self):
        """Record parameter gradient norms for the current batch.

        This is only used when ``self.track_grad`` is enabled.
        """
        with torch.no_grad():
            for layer_name, g in self.model.named_parameters():
                # gn = g.grad.data.norm().item()
                gn = g.grad.norm().item()
                self.batch_layer_grad_norm[layer_name].append(gn)

    def plot_gradient_norms(layer_grad_norms, out_file, plots_per_page = 4):

        laynorm = [z for z in layer_grad_norms.items()]
        n_plots = len(laynorm)
        n_pages = n_plots // 4 + ((n_plots % 4) > 0)
        with PdfPages(out_file) as pdf:
            for page in range(n_pages):
                fig, axes = plt.subplots(plots_per_page // 2, 2)
                axes = axes.flatten()
                for plot_i in range(plots_per_page):
                    idx = page * plots_per_page + plot_i
                    if idx >= n_plots:
                        axes.axis('off')
                        continue
                    axes[plot_i].plot(laynorm[idx][1])
                    axes[plot_i].set_title(laynorm[idx][0], size = 6.)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


    def make_graph(self):
        """Add the model graph to a TensorBoard writer (if configured).

        Notes:
            This method expects ``self.writer`` to be set externally to a TensorBoard
            ``SummaryWriter``-like object.
        """
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

    def set_seed(self, seed = None):
        """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

        Notes:
            This mutates global RNG state (``random``, ``numpy.random``, and ``torch``) and sets
            cuDNN to deterministic mode.

        Args:
            seed (int, optional): Seed value. If None, this is a no-op. Defaults to None.
        """
        if seed is None:
            return  # use module-level RNGs as-is

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def save_checkpoint(self, filename):
        """Save a training checkpoint to disk.

        The checkpoint includes model/optimizer state and basic training metadata.

        Args:
            filename (str): Output path for ``torch.save(...)``.
        """
        if not isinstance(self.train_loss, PhyLoss):
            raise TypeError("save_checkpoint requires `train_loss` to be a PhyLoss instance.")
        if not isinstance(self.val_loss, PhyLoss):
            raise TypeError("save_checkpoint requires `val_loss` to be a PhyLoss instance.")

        sched_module = None
        sched_class = None
        sched_init_kwargs = None
        sched_state = None
        if self.lr_sched is not None:
            sched_module = self.lr_sched.__class__.__module__
            sched_class = self.lr_sched.__class__.__name__
            sched_state = self.lr_sched.state_dict()
            if isinstance(self.lr_sched, torch.optim.lr_scheduler.OneCycleLR):
                phases = sched_state['_schedule_phases']
                total_steps = int(sched_state['total_steps'])
                pct_start = (float(phases[0]['end_step']) + 1.0) / float(total_steps)
                param_groups = self.optimizer.param_groups
                max_lrs = [float(pg['max_lr']) for pg in param_groups]
                initial_lrs = [float(pg['initial_lr']) for pg in param_groups]
                min_lrs = [float(pg['min_lr']) for pg in param_groups]
                sched_init_kwargs = {
                    'max_lr': max_lrs,
                    'total_steps': total_steps,
                    'pct_start': pct_start,
                    'anneal_strategy': sched_state['_anneal_func_type'],
                    'cycle_momentum': bool(sched_state['cycle_momentum']),
                    'div_factor': max_lrs[0] / initial_lrs[0],
                    'final_div_factor': initial_lrs[0] / min_lrs[0],
                    'three_phase': len(phases) == 3,
                }
                if sched_state['cycle_momentum']:
                    sched_init_kwargs['base_momentum'] = [float(pg['base_momentum']) for pg in param_groups]
                    sched_init_kwargs['max_momentum'] = [float(pg['max_momentum']) for pg in param_groups]
            else:
                raise TypeError(
                    f"save_checkpoint does not support scheduler type "
                    f"{self.lr_sched.__class__.__module__}.{self.lr_sched.__class__.__name__}."
                    f"Currently only supports torch.optim.lr_scheduler.OneCycleLR."
                )

        # necessary if want to create new model objects with same state
        checkpoint = {'model_config' : self.model.get_config_dict(),
                      'seed': self.seed,
                      'epoch': self.epoch,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_module': self.optimizer.__class__.__module__,
                      'optimizer_class': self.optimizer.__class__.__name__,
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'scheduler_module': sched_module,
                      'scheduler_class': sched_class,
                      'scheduler_init_kwargs': sched_init_kwargs,
                      'scheduler_state_dict': sched_state,
                      'train_loss_state': self.train_loss.to_checkpoint_dict(),
                      'val_loss_state': self.val_loss.to_checkpoint_dict()}
        torch.save(checkpoint, filename)

    @classmethod
    def load_checkpoint(cls, filename, map_location : Optional[str] = "cpu",
                        track_grad: bool = False) -> "PhyloAutoencoder":
        """
        Assumes the file is a checkpoint file output by save_checkpoint() above. 
        """
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)

        model = AECNN(**checkpoint['model_config'])
        model_state_dict = model.repair_legacy_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(model_state_dict)

        opt_state = checkpoint['optimizer_state_dict']
        opt_module_name = checkpoint['optimizer_module']
        opt_class_name = checkpoint['optimizer_class']
        opt_module = importlib.import_module(opt_module_name)
        opt_cls = getattr(opt_module, opt_class_name)

        saved_groups = opt_state['param_groups']
        if len(saved_groups) == 2:
            wd = max(float(group['weight_decay']) for group in saved_groups)
            opt = opt_cls(utils.split_params_by_wd(model, wd))
        else:
            opt = opt_cls(model.parameters())

        lr_sched = None
        sched_state = checkpoint['scheduler_state_dict']
        if sched_state is not None:
            sched_module_name = checkpoint['scheduler_module']
            sched_class_name  = checkpoint['scheduler_class']
            sched_init_kwargs = checkpoint['scheduler_init_kwargs']
            sched_module = importlib.import_module(sched_module_name)
            sched_cls = getattr(sched_module, sched_class_name)
            lr_sched = sched_cls(opt, **sched_init_kwargs)

        load_device = None if map_location is None else str(map_location)
        train_loss = PhyLoss.from_checkpoint_dict(checkpoint['train_loss_state'], device=load_device)
        val_loss   = PhyLoss.from_checkpoint_dict(checkpoint['val_loss_state'], device=load_device)

        self = cls(
            model=model,
            optimizer=opt,
            lr_scheduler=lr_sched,
            train_loss=train_loss,
            val_loss=val_loss,
            seed=checkpoint['seed'],
            device="auto" if load_device is None else load_device,
            track_grad=track_grad,
        )

        self.optimizer.load_state_dict(opt_state)
        if sched_state is not None and self.lr_sched is not None:
            self.lr_sched.load_state_dict(sched_state)
        self.epoch = checkpoint['epoch']
        self.model.train()
        return self



    def tree_encode(self, phy: torch.Tensor, aux: torch.Tensor, *,
               inference = False, detach = False ):
        """Encode inputs into the latent representation.

        This is a thin wrapper around ``model.encode(...)``.

        Args:
            phy (torch.Tensor): Structured input tensor shaped ``(batch, channels, width)``.
            aux (torch.Tensor): Unstructured/auxiliary input tensor shaped ``(batch, aux_dim)``.
            inference (bool, optional): If True, runs in eval mode and disables gradients in
                the underlying model method. Defaults to False.
            detach (bool, optional): If True and ``inference`` is True, detaches outputs.
                Defaults to False.

        Returns:
            torch.Tensor: Flattened latent representation shaped ``(batch, latent_dim)``.
        """
        # same defaults as model.encode
        return self.model.encode(phy, aux, inference = inference, detach = detach)

    def latent_decode(self, encoded_tree: torch.Tensor, *,
               inference = False, detach = False):
        """Decode a latent representation back to ``(phy, aux)`` outputs.

        This is a thin wrapper around ``model.decode(...)``.

        Args:
            encoded_tree (torch.Tensor): Latent tensor shaped ``(batch, latent_dim)``.
            inference (bool, optional): If True, runs in eval mode and disables gradients in
                the underlying model method. Defaults to False.
            detach (bool, optional): If True and ``inference`` is True, detaches outputs.
                Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(phy_decoded, aux_decoded)`` tensors.
        """
        # same defaults as model.decode
        return self.model.decode(encoded_tree, inference = inference, detach = detach)

    def get_latent_shape(self):
        """Return the structured latent shape used by the model.

        Returns:
            Tuple[int, int]: ``(num_structured_latent_channels, reshaped_shared_latent_width)``.
        """
        return self.model.num_structured_latent_channels, self.model.reshaped_shared_latent_width
       
    def plot_losses(self, out_prefix = "AElossplot", log = True, starting_epoch = 10):
        """Plot training and validation loss curves.

        Args:
            out_prefix (str, optional): Output filename prefix. Defaults to ``"AElossplot"``.
            log (bool, optional): If True, use a log y-scale. Defaults to True.
            starting_epoch (int, optional): First epoch to include in plot. Defaults to 10.
        """

        utils.make_loss_plots(self.train_loss, self.val_loss, 
                                latent_layer_type=self.model.latent_layer_type,
                                out_prefix=out_prefix, log=log, starting_epoch=starting_epoch)

    def _split_tree_char(self, phy : torch.Tensor, mask : Optional[torch.Tensor]) -> Tuple[torch.Tensor, 
                                                                                 torch.Tensor,
                                                                                 torch.Tensor,       
                                                                                 torch.Tensor, ]: 
        """Split structured input and mask into tree and character channels.

        Args:
            phy (torch.Tensor): Structured input tensor shaped ``(batch, channels, width)``.
            mask (torch.Tensor): Optional boolean mask shaped like ``phy``.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            ``(tree, char, tree_mask, char_mask)``. If ``model.num_chars == 0``, ``char`` and
            ``char_mask`` are returned as None.
        """
        # divide phy into tree and character data
        if self.nchars > 0:
            tree = phy[:, :self.num_tree_chans, :]
            char = phy[:, self.num_tree_chans:, :]
            tree_mask = mask[:, :self.num_tree_chans, :] if mask is not None else None
            char_mask = mask[:, self.num_tree_chans:, :] if mask is not None else None
        else:
            tree = phy
            char = None
            tree_mask = mask
            char_mask = None

        return tree, char, tree_mask, char_mask
