#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch
# from torch import optim
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from phyloencode import utils
from phyloencode.PhyLoss import PhyLoss
import time
# import os
from typing import List, Dict, Tuple, Optional, Union


class PhyloAutoencoder(object):
    def __init__(self, model, optimizer, lr_scheduler = None,
                 batch_size=128, phy_loss_weight=0.5, char_weight=0.5,
                 mmd_lambda=None, vz_lambda=None):
        """ Performs training and valdiation on an autoencoder object.

        Args:
            model (nn.Module): Autoencoder model
            optimizer (torch.optim): optimizer for sgd updating.
            batch_size (int, optional): Minibatch size. Defaults to 128.
            phy_loss_weight (float, optional): _description_. Defaults to 0.5.
            char_weight (float, optional): _description_. Defaults to 0.5.
            mmd_lambda (_type_, optional): _description_. Defaults to None.
            vz_lambda (_type_, optional): _description_. Defaults to None.
        """
        
        # TODO: define the model object better (autoencoder ...)
        # TODO: the Loss object should maybe be passed in as a parameter
        # TODO: run checks that the model has the expected attributes
        # TODO: Update seed functionality
        # TODO: create an aux_weight, change everything to "lambda" or "weight" for consistency
        # TODO: add track_grad to parameters

        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size   = batch_size
        self.total_epochs = 0
        self.train_loader = None
        self.val_loader   = None
        self.model        = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.lr_sched  = lr_scheduler


        # some data shape parameters
        self.nchars             = self.model.num_chars
        self.char_weight        = char_weight if self.nchars > 0 else torch.tensor(0.)
        self.char_type          = self.model.char_type
        self.phy_channels       = self.model.num_structured_input_channel
        self.num_tree_chans     = self.phy_channels - self.nchars
        self.latent_shape       = (self.batch_size, self.model.latent_outwidth)

        self.weights = (phy_loss_weight, self.char_weight, 1-phy_loss_weight, mmd_lambda, vz_lambda)

        self.train_loss  = PhyLoss(self.weights, self.char_type, self.model.latent_layer_type, self.device)
        self.val_loss    = PhyLoss(self.weights, self.char_type, self.model.latent_layer_type, self.device)

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

        # TODO: Would returning the trained model object make sense here for downstream readability?

        if seed is not None:
            self.set_seed(seed)

        for epoch in range(num_epochs):
            self.total_epochs += 1
            epoch_time = time.time()

            # perform all mini batch steps for the epoch
            self._mini_batch(validation=False)
            self.train_loss.append_mean_batch_loss()
            # print training epoch mean losses to screen
            self.train_loss.print_epoch_losses(elapsed_time=time.time() - epoch_time)

            with torch.no_grad():
                self._mini_batch(validation=True)             
                self.val_loss.append_mean_batch_loss()
                # print epoch mean component losses to screen       
                self.val_loss.print_epoch_losses(elapsed_time = time.time() - epoch_time)
                print("")


        
    def _mini_batch(self, validation = False):
        """Performs a mini batch step for the model.
        loops through the data loader and performs a step for each batch
        Uses self.step_function to perform SGD step for each batch
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
            self.std_norm = torch.randn(self.latent_shape).to(self.device) if self.model.latent_layer_type == "GAUSS" else None

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
        loss = self.train_loss.minibatch_loss(pred, true, segmented_mask)

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
                  mask: torch.Tensor = None, std_norm : torch.Tensor = None):
        
        # batch val loss        
        self.model.eval()

        # divide phy into tree and character data
        tree, char, tree_mask, char_mask = self._split_tree_char(phy, mask)

        segmented_mask = (tree_mask, char_mask)
        true = (tree, char, aux, std_norm)
        pred = self.model((phy, aux))

        # compute and update loss fields in val_loss
        self.val_loss.minibatch_loss(pred, true, segmented_mask)

        
    def predict(self, phy: torch.Tensor, aux: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval() 
        phy = phy.to(self.device)
        aux = aux.to(self.device)
        tree_pred, char_pred, aux_pred, latent = self.model((phy, aux))

        if char_pred is not None:
            if self.char_type == "categorical":
                char_sftmx = torch.softmax(char_pred, dim = 1)
                phy_pred = torch.cat((tree_pred, char_sftmx), dim = 1)
            else:
                phy_pred =  torch.cat((tree_pred, char_pred), dim = 1)
        else:
            phy_pred = tree_pred

        self.model.train()
        return phy_pred.detach().cpu().numpy(), aux_pred.detach().cpu().numpy()


    def to_device(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            print(f"Didn't work, sending to {self.device} instead.")

    # TODO: should mask_loader be split to train and val loaders?
    def set_data_loaders(self, train_loader : torch.utils.data.DataLoader, 
                               val_loader   : torch.utils.data.DataLoader = None,
                               mask_loader  : torch.utils.data.DataLoader = None):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.mask_loader  = mask_loader

    def _record_grad_norm(self):
        with torch.no_grad():
            for layer_name, g in self.model.named_parameters():
                # gn = g.grad.data.norm().item()
                gn = g.grad.norm().item()
                self.batch_layer_grad_norm[layer_name].append(gn)

    # TODO: many of these methods should probably be handled by the model object
    def make_graph(self):
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

    def set_seed(self, seed = 1):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

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

    def save_model(self, filename):
        torch.save(self.model, filename)

    def tree_encode(self, phy: torch.Tensor, aux: torch.Tensor):
        self.model.eval() 
        phy = phy.to(self.device)
        aux = aux.to(self.device)

        # get latent unstrcutured and structured embeddings
        structured_encoded_x   = self.model.structured_encoder(phy)    # (1, nchannels, out_width)
        unstructured_encoded_x = self.model.unstructured_encoder(aux)  # (1, out_width)

        # reshape structured embeddings
        flat_structured_encoded_x = structured_encoded_x.flatten(start_dim=1)
        combined_latent           = torch.cat((flat_structured_encoded_x, 
                                               unstructured_encoded_x), dim=1)
        
        
        # get combined latent output
        if self.model.latent_layer_type   == "CNN":
            reshaped_shared_latent = combined_latent.view(-1, self.model.num_structured_latent_channels, 
                                                              self.model.reshaped_shared_latent_width)
            shared_latent_out = self.model.latent_layer(reshaped_shared_latent)

        elif self.model.latent_layer_type == "GAUSS":
            shared_latent_out = self.model.latent_layer(combined_latent)

        elif self.model.latent_layer_type == "DENSE":
            shared_latent_out = self.model.latent_layer(combined_latent)

        self.model.train()

        return(shared_latent_out.flatten(start_dim=1))

    def get_latent_shape(self):
        return self.model.num_structured_latent_channels, self.model.reshaped_shared_latent_width

    def latent_decode(self, encoded_tree: torch.Tensor):
        self.model.eval()
        encoded_tree = encoded_tree.to(self.device)
        decoded_latent_layer = self.model.latent_layer_decoder(encoded_tree)
        # reshape
        reshaped_decoded_latent_layer = decoded_latent_layer.view(-1, self.model.num_structured_latent_channels, 
                                                                        self.model.reshaped_shared_latent_width) 
        decoded_tree = self.model.structured_decoder(reshaped_decoded_latent_layer)
        decoded_aux  = self.model.unstructured_decoder(decoded_latent_layer.flatten(start_dim=1))
        if self.char_type == "categorical":
            # softmax the char data
            char_start_idx = decoded_tree.shape[1] - self.nchars
            decoded_tree[:,char_start_idx:,:] = torch.softmax(decoded_tree[:,char_start_idx:,:], dim = 1)
        self.model.train()
        return decoded_tree, decoded_aux
        
    def plot_losses(self, out_prefix = "AElossplot", log = True, starting_epoch = 10):
        # plot total losses
        fig = plt.figure(figsize=(11, 8))
        plt.plot(list(range(len(self.train_loss.epoch_total_loss)))[:starting_epoch],
                 np.log10(self.train_loss.epoch_total_loss[:starting_epoch]), label='Training Loss', c="b")
        if self.val_loader:
            plt.plot(np.log10(self.val_loss.epoch_total_loss), label='Validation Loss', c='r')
        plt.xlabel('Epochs')
        plt.ylabel('log10 Loss')
        plt.legend()        
        plt.grid(True)
        range_y = [min(np.log10(np.concat((self.train_loss.epoch_total_loss, 
                                           self.val_loss.epoch_total_loss)))), 
                   max(np.log10(np.concat((self.train_loss.epoch_total_loss, 
                                           self.val_loss.epoch_total_loss))))]
        plt.yticks(ticks = np.linspace(range_y[0], range_y[1], num = 20))
        plt.xticks(ticks=np.arange(0, len(self.train_loss.epoch_total_loss), 
                                   step=len(self.train_loss.epoch_total_loss) // 10))
        plt.tight_layout()
        plt.savefig(out_prefix + ".loss.pdf", bbox_inches='tight')
        plt.close(fig)

        # plot each loss separately (only validation losses are recorded). 
        # create subplots for each loss component   
        # # TODO: fix x-axis tick marks            
        num_subplots = 6 if self.model.latent_layer_type == "GAUSS" else 4
        fig, axs = plt.subplots(num_subplots//2, 2, figsize=(11, 8), sharex=True)
        # axs = axs.flatten()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        self._fill_in_loss_comp_fig(self.val_loss.epoch_total_loss, "combined", axs[0,0], starting_epoch)
        self._fill_in_loss_comp_fig(self.val_loss.epoch_phy_loss, "phy", axs[0,1], starting_epoch)
        self._fill_in_loss_comp_fig(self.val_loss.epoch_char_loss, "char", axs[1,1], starting_epoch)
        self._fill_in_loss_comp_fig(self.val_loss.epoch_aux_loss, "aux", axs[1,0], starting_epoch)
        if self.model.latent_layer_type == "GAUSS":
            self._fill_in_loss_comp_fig(self.val_loss.epoch_mmd_loss, "mmd", axs[2,0], starting_epoch)
            self._fill_in_loss_comp_fig(self.val_loss.epoch_vz_loss, "vz", axs[2,1], starting_epoch)
        plt.savefig(out_prefix + ".component_loss.pdf", bbox_inches='tight')
        plt.close(fig)

    def _fill_in_loss_comp_fig(self, val_losses, plot_label, ax, starting_epoch = 10):
        ax.plot(list(range(len(val_losses)))[starting_epoch:], 
                np.log10(val_losses[starting_epoch:]), label=plot_label, c="b")
                # val_losses[starting_epoch:], label=plot_label, c="b")
        ax.set_title(f"{plot_label} Loss")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Log10 Loss')
        # ax.set_ylabel('Loss')
        ax.grid(True)
        ax.set_xticks(ticks=np.arange(0, len(val_losses), step=len(val_losses) // 10))
        
    def _split_tree_char(self, phy : torch.Tensor, mask : torch.Tensor) -> Tuple[torch.Tensor, 
                                                                                 torch.Tensor,
                                                                                 torch.Tensor, 
                                                                                 torch.Tensor, ]: 
        """_summary_

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
