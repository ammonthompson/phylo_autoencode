#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from phyloencode import utils
# from sklearn.linear_model import LinearRegression
import time
import os
from typing import List, Dict, Tuple, Optional, Union


class PhyloAutoencoder(object):
    def __init__(self, model, optimizer = optim.Adam, 
                 batch_size=128, phy_loss_weight=0.5, char_weight=0.5,
                 mmd_lambda=None, vz_lambda=None):
        '''
            model is an object from torch.model
            optimizer is an object from torch.optim
            loss_func is an object from torch.nn
            phy_loss_weight is a float between 0 and 1
            mmd_lambda is a float >= 0
            vz_lambda is a float >=0
        '''

        self.device             = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size         = batch_size
        self.optimizer          = optimizer

        self.total_epochs       = 0
        self.train_loader       = None
        self.val_loader         = None

        self.model              = model
        self.model.to(self.device)

        self.nchars             = self.model.num_chars
        self.char_type          = self.model.char_type
        
        # TODO: do a proper error handing at some point 
        # (should be passed as a parameter)

        self.weights = (phy_loss_weight, char_weight, mmd_lambda, vz_lambda)

        self.train_loss  = utils.PhyLoss(self.weights, self.char_type, self.model.latent_layer_type)
        self.val_loss    = utils.PhyLoss(self.weights, self.char_type, self.model.latent_layer_type)


        # self.writer = None

    def train(self, num_epochs, seed = 1):
        self.set_seed(seed)

        for epoch in range(num_epochs):
            self.total_epochs += 1
            epoch_time = time.time()

            self._mini_batch()
            self.train_loss.append_mean_batch_loss()

            with torch.no_grad():
                self._mini_batch(validation=True)             
                self.val_loss.append_mean_batch_loss()       
                elapsed_time = time.time() - epoch_time
                self.val_loss.print_epoch_losses(elapsed_time)

            # TODO
        #     if self.writer:
        #         scalars = {'training':epoch_train_loss}
        #         if epoch_validation_loss != None:
        #             scalars.update({'validation':epoch_validation_loss})

        #         self.writer.add_scalars(main_tag='loss', tag_scaler_dict = scalars, 
        #                                 global_step = epoch)                
         
        # if self.writer:
        #     self.writer.flush()

        
    def _mini_batch(self, validation = False):
        # Performs a mini batch step for the model.
        # loops through the data loader and performs a step for each batch
        # Uses self.step_function to perform SGD step for each batch
        # returns the mean loss from the mini batch steps
        # If validation = True, then returns phy, aux, and mmd loss.

        # set data loader and step function
        if validation:
            data_loader   = self.val_loader
            step_function = self.evaluate
        else:
            data_loader   = self.train_loader
            step_function = self.train_step

        if data_loader == None:
            return None

        if self.model.latent_layer_type == "GAUSS":
            std_norm =  torch.randn(self.model.latent_outwidth).to(self.device)
        else:
            std_norm = None

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

            # perform SGD step for batch
            step_function(phy_batch, aux_batch, mask_batch, std_norm)


    
    def train_step(self, phy: torch.Tensor, aux: torch.Tensor, 
                   mask: torch.Tensor = None, std_norm : torch.Tensor = None):
        # batch train loss
        # set model to train mode
        self.model.train()

        phy_hat, aux_hat, latent = self.model((phy, aux))
           
        loss = self.train_loss.minibatch_loss((phy_hat, None, aux_hat, latent), 
                                              (phy, None, aux, std_norm), mask)

        # compute gradient
        loss.backward()

        # update model paremeters with gradient
        self.optimizer.step()

        # Clear the gradient for the next iteration, 
        # preventing accumulation from previous steps.
        self.optimizer.zero_grad()

        
    def evaluate(self, phy: torch.Tensor, aux: torch.Tensor,
                  mask: torch.Tensor = None, std_norm : torch.Tensor = None):
        
        # batch val loss
        
        self.model.eval()

        phy_hat, aux_hat, latent = self.model((phy, aux))

        self.val_loss.minibatch_loss((phy_hat, None, aux_hat, latent),
                                     (phy, None, aux, std_norm), mask)


    def plot_losses(self, out_prefix = "AElossplot", log = True):
        # plot total losses
        fig = plt.figure(figsize=(11, 8))
        plt.plot(np.log10(self.train_loss.epoch_total_loss), label='Training Loss', c="b")
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
        num_subplots = 4 if self.model.latent_layer_type == "GAUSS" else 2
        fig, (axs1, axs2) = plt.subplots(num_subplots//2, 2, figsize=(11, 8), sharex=True)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        self.fill_in_loss_comp_fig(self.train_loss.epoch_phy_loss, "phy", axs1[0])
        self.fill_in_loss_comp_fig(self.train_loss.epoch_aux_loss, "aux", axs1[1])
        if self.model.latent_layer_type == "GAUSS":
            self.fill_in_loss_comp_fig(self.train_loss.epoch_mmd_loss, "mmd", axs2[0])
            self.fill_in_loss_comp_fig(self.train_loss.epoch_vz_loss, "vz", axs2[1])
        plt.savefig(out_prefix + ".component_loss.pdf", bbox_inches='tight')
        plt.close(fig)

    def fill_in_loss_comp_fig(self, val_losses, plot_label, ax):
        ax.plot(np.log10(val_losses), label=plot_label, c="b")
        ax.set_title(f"{plot_label} Loss")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Log10 Loss')
        ax.grid(True)
        ax.set_xticks(ticks=np.arange(0, len(val_losses), step=len(val_losses) // 10))
        
    def predict(self, phy: torch.Tensor, aux: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval() 
        phy = phy.to(self.device)
        aux = aux.to(self.device)
        phy_pred, aux_pred, latent = self.model((phy, aux))

        if self.char_type == "categorical":
            # softmax the char data
            char_start_idx = phy_pred.shape[1] - self.nchars
            phy_pred[:,char_start_idx:,:] = torch.softmax(phy_pred[:,char_start_idx:,:], dim = 1)

        self.model.train()
        return phy_pred.detach().cpu().numpy(), aux_pred.detach().cpu().numpy()

    def to_device(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            print(f"Didn't work, sending to {self.device} instead.")

    def set_data_loaders(self, train_loader : torch.utils.data.DataLoader, 
                               val_loader   : torch.utils.data.DataLoader = None,
                               mask_loader  : torch.utils.data.DataLoader = None):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.mask_loader  = mask_loader

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
        decoded_tree = self.model.structured_decoder(encoded_tree)
        decoded_aux  = self.model.unstructured_decoder(encoded_tree.flatten(start_dim=1))
        if self.char_type == "categorical":
            # softmax the char data
            char_start_idx = decoded_tree.shape[1] - self.nchars
            decoded_tree[:,char_start_idx:,:] = torch.softmax(decoded_tree[:,char_start_idx:,:], dim = 1)
        self.model.train()
        return decoded_tree, decoded_aux

        
