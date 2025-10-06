#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch
# from torch import optim
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from phyloencode import utils
# from phyloencode.PhyLoss import PhyLoss
from phyloencode.DataProcessors import AEData
from phyloencode.PhyloAEModel import AECNN
import phyloencode.utils as utils
import time
import random
# import os
from typing import List, Dict, Tuple, Optional, Union


class PhyloAutoencoder(object):

    def __init__(self,
                 model: AECNN, 
                 optimizer, 
                 *, 
                 lr_scheduler = None, 
                 batch_size=128, 
                 train_loss = None, 
                 val_loss = None, 
                 seed = None, 
                 device = "auto"):
        """Performs training and valdiation on an autoencoder model.

        Args:
            model (AECNN): _description_
            optimizer (_type_): _description_
            lr_scheduler (_type_, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 128.
            train_loss (torch.nn.Module, optional): _description_. Defaults to None.
            val_loss (torch.nn.Module optional): _description_. Defaults to None.
            seed (int): seed. Defaults to None.
            device ({"cuda", "cpu"}) : device. Defaults to "auto"
        """
        
        # TODO: define the model object better (autoencoder ...)
        # TODO: run checks that the model has the expected attributes
        # TODO: add track_grad to parameters
        # TODO: add checks that the loss objects are correct (contain certain fields and methods)
        # TODO: fix checkpoints so starting w/pre-trained network can be used easilly.

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
        self.total_epochs = 0
        self.train_loader = None
        self.val_loader   = None
        self.model        = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.lr_sched  = lr_scheduler
        # self.aux_ntax_cidx = aux_ntax_cidx

        # some data shape parameters
        self.nchars             = self.model.num_chars
        self.char_type          = self.model.char_type
        self.phy_channels       = self.model.num_structured_input_channel
        self.num_tree_chans     = self.phy_channels - self.nchars
        self.latent_shape       = (self.batch_size * 2, self.model.latent_outwidth)

        self.train_loss = train_loss
        self.val_loss   = val_loss

        self.track_grad = False
        if self.track_grad:
            self.batch_layer_grad_norm = { layer_name : param.grad.norm().item() if param.grad is not None else []
                               for layer_name, param in self.model.named_parameters() }
            self.mean_layer_grad_norm = { layer_name : param.grad.norm().item() if param.grad is not None else []
                               for layer_name, param in self.model.named_parameters() }
        else:
            self.batch_layer_grad_norm = None
            self.mean_layer_grad_norm = None


    def train(self, num_epochs, seed = None):

        if self.train_loader is None:
            raise ValueError("Must load training data.")
        if self.train_loss is None:
            raise ValueError("Must load loss layer.")

        # If both local and instance seed are None, 
        # then module level rng outside class governs random number generation
        if seed is not None:
            # set seed to instance variable, self.seed (which might also be None)
            self.set_seed(self.seed)
        else:
            # set seed to parameter seed
            self.set_seed(seed)

        for epoch in range(num_epochs):
            self.total_epochs += 1
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
            with torch.no_grad():
                self._mini_batch(validation=True)             
                self.val_loss.append_mean_batch_loss()
                # print epoch mean component losses to screen       
                self.val_loss.print_epoch_losses(elapsed_time = time.time() - epoch_time)

       
    def _mini_batch(self, validation = False):
        """Performs a mini batch step for the model.
        loops through the data loader and performs a step for each batch
        Uses step_function (self._train_step or self.evaluate) to perform SGD step for each batch
        updates the mean loss from the mini batch steps

        Args:
            validation (bool, optional): use val_loader and evaluate if True,
            If False use train_loader and _train_step. Defaults to False.

        Returns:
            None
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
                   mask: torch.Tensor = None, std_norm : torch.Tensor = None):
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
        if self.lr_sched != None:
            self.lr_sched.step()

        
    def evaluate(self, phy: torch.Tensor, aux: torch.Tensor,
                  mask: torch.Tensor = None, std_norm: torch.Tensor = None):
        
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

        return self.model.predict(phy, aux, inference = inference, detach = detach)


    def to_device(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            print(f"Didn't work, sending to {self.device} instead.")

    def set_data_loaders(self, train_loader : torch.utils.data.DataLoader, 
                               val_loader   : torch.utils.data.DataLoader = None):
        
        self.train_loader = train_loader
        self.val_loader   = val_loader

        if not isinstance(self.train_loader, torch.utils.data.DataLoader):
            raise TypeError(f"train_loader must be a DataLoader, got {type(self.train_loader).__name__}.")

        if not isinstance(self.val_loader, torch.utils.data.DataLoader):
            raise TypeError(f"val_loader must be a DataLoader, got {type(self.train_loader).__name__}.")


    def set_losses(self, train_loss, val_loss):
        self.train_loss = train_loss
        self.val_loss = val_loss

    # def set_data(self, data : AEData, num_workers : int):
    #     self.data = data
    #     self.train_loader, self.val_loader = self.data.get_dataloaders(self.batch_size, shuffle = True, 
    #                                                                     num_workers = num_workers)

    def _record_grad_norm(self):
        with torch.no_grad():
            for layer_name, g in self.model.named_parameters():
                # gn = g.grad.data.norm().item()
                gn = g.grad.norm().item()
                self.batch_layer_grad_norm[layer_name].append(gn)

    def make_graph(self):
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

    def set_seed(self, seed = None):
        if seed is None:
                return  # use module-level RNGs as-is

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def save_checkpoint(self, filename):
        checkpoint = {'epoch':self.total_epochs,
                      'model_state_dict':self.model.state_dict(),
                      'optimizer_state_dict':self.optimizer.state_dict(),
                      'loss':self.losses,
                      'val_loss':self.val_losses}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        self.model.train()

    # TODO: instead save the model.state_dict along with epoch, optimizer state. Will require instantiating model object
    # in downstream applications that use trained model.
    def save_model(self, filename):
        torch.save(self.model, filename)

    def tree_encode(self, phy: torch.Tensor, aux: torch.Tensor, *,
               inference = False, detach = False ):
        # same defaults as model.encode
        return self.model.encode(phy, aux, inference = inference, detach = detach)

    def latent_decode(self, encoded_tree: torch.Tensor, *,
               inference = False, detach = False):
        # same defaults as model.decode
        return self.model.decode(encoded_tree, inference = inference, detach = detach)

    def get_latent_shape(self):
        return self.model.num_structured_latent_channels, self.model.reshaped_shared_latent_width
       
    def plot_losses(self, out_prefix = "AElossplot", log = True, starting_epoch = 10):

        utils.make_loss_plots(self.train_loss, self.val_loss, 
                                latent_layer_type=self.model.latent_layer_type,
                                out_prefix=out_prefix, log=log, starting_epoch=starting_epoch)

    def _split_tree_char(self, phy : torch.Tensor, mask : torch.Tensor) -> Tuple[torch.Tensor, 
                                                                                 torch.Tensor,
                                                                                 torch.Tensor,       
                                                                                 torch.Tensor, ]: 
        """used by train_step and evaluate

        Args:
            phy (torch.Tensor): _description_
            mask (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ]: _description_
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

