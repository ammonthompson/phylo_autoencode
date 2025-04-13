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


class PhyloAutoencoder(object):
    def __init__(self, model, optimizer, loss_func, batch_size=128,
                 phy_loss_weight=0.5, char_weight=0.5,
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
        self.loss_func          = loss_func
        self.phy_loss_weight    = phy_loss_weight
        self.char_weight        = char_weight
        self.total_epochs       = 0
        self.train_loader       = None
        self.val_loader         = None

        self.model              = model
        self.model.to(self.device)

        # TODO: do a proper error handing at some point 
        # (should be passed as a parameter)

        self.losses     = []
        self.val_losses = []
        self.phy_losses = []
        self.aux_losses = []

        if model.latent_layer_type == "GAUSS":
            self.mmd_lambda = mmd_lambda
            self.vz_lambda = vz_lambda
            self.latent_layer_shape = (self.batch_size, model.combined_latent_width)
            self.std_norm = torch.randn(self.latent_layer_shape).to(self.device)
            self.mmd_losses = []
            self.vz_losses = []


        # step functions. These perform a single gradient descent step:
        # (model(x), then backward) given a batch
        # updates internal state. returns the loss for the batch
        # self.train_step = self._make_train_step_func()
        # self.val_step   = self._make_val_func()
        # self.train_step = self.train_step
        # self.val_step   = self.evaluate


        # self.writer = None

    def train(self, num_epochs, seed = 1):
        self.set_seed(seed)

        for epoch in range(num_epochs):
            self.total_epochs += 1
            epoch_time = time.time()
            # new std_norm for each epoch (shared by all batches)
            # if self.model.latent_layer_type == "GAUSS":
            #     self.std_norm = torch.randn(self.latent_layer_shape).to(self.device)
            epoch_train_loss = self._mini_batch()
            elapsed_time = time.time() - epoch_time
            with torch.no_grad():
                if self.model.latent_layer_type == "GAUSS":
                    epoch_validation_loss, phy_loss, aux_loss, mmd_loss, vz_loss = self._mini_batch(validation=True)
                    print(f"Epoch {epoch},  " +
                            f"Loss: {epoch_validation_loss:.4f},  " +
                            f"phy L: {phy_loss:.4f},  " +
                            f"aux L: {aux_loss:.4f},  " +
                            f"MMD L: {mmd_loss:.4f},  " +
                            f"VZ L: {vz_loss:.4f},  " +
                            f"Run time: {elapsed_time:.3f} sec"
                            )
                else:
                    epoch_validation_loss, phy_loss, aux_loss = self._mini_batch(validation=True)
                    print(f"Epoch {epoch},  " +
                            f"Loss: {epoch_validation_loss:.4f},  " +
                            f"phy L: {phy_loss:.4f},  " +
                            f"aux L: {aux_loss:.4f}, " + 
                            f"Run time: {elapsed_time:.3f} sec"
                            )

            # append losses to list for plotting
            self.losses.append(epoch_train_loss)
            self.val_losses.append(epoch_validation_loss)

            # append phy, aux, mmd, vz losses to list for component loss plots
            self.phy_losses.append(phy_loss)
            self.aux_losses.append(aux_loss)
            if self.model.latent_layer_type == "GAUSS":
                self.mmd_losses.append(mmd_loss)    
                self.vz_losses.append(vz_loss)
               


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

        # perform step and return loss
        # loop through all batches of train or val data
        mini_batch_losses       = []
        phy_mini_batch_losses   = []
        aux_mini_batch_losses   = []
        mmd_mini_batch_losses   = []
        vz_mini_batch_losses    = []
        for phy_batch, aux_batch in data_loader:
            phy_batch = phy_batch.to(self.device)
            aux_batch = aux_batch.to(self.device)
            if not validation:
                # one step of SGD
                loss = step_function(phy_batch, aux_batch)
            elif self.model.latent_layer_type == "GAUSS" and validation:
                # one step of SGD
                loss, phy_loss, aux_loss, mmd_loss, vz_loss = step_function(phy_batch, aux_batch)
                phy_mini_batch_losses.append(phy_loss)
                aux_mini_batch_losses.append(aux_loss)
                mmd_mini_batch_losses.append(mmd_loss)
                vz_mini_batch_losses.append(vz_loss)
            else:
                # one step of SGD
                loss, phy_loss, aux_loss = step_function(phy_batch, aux_batch)
                phy_mini_batch_losses.append(phy_loss)
                aux_mini_batch_losses.append(aux_loss)

            mini_batch_losses.append(loss)

        # return the mean loss over all batches
        mean_loss = torch.mean(torch.stack(mini_batch_losses))
        if self.model.latent_layer_type == "GAUSS" and validation:
            return (
                mean_loss.item(),
                torch.mean(torch.stack(phy_mini_batch_losses)).item(),
                torch.mean(torch.stack(aux_mini_batch_losses)).item(),
                torch.mean(torch.stack(mmd_mini_batch_losses)).item(),
                torch.mean(torch.stack(vz_mini_batch_losses)).item(),
            )
        elif validation:
            return (
                mean_loss.item(),
                torch.mean(torch.stack(phy_mini_batch_losses)).item(),
                torch.mean(torch.stack(aux_mini_batch_losses)).item(),
            )
        else:
            return mean_loss.item()
    

    # def _make_train_step_func(self):

    # def train_step(phy: torch.Tensor, aux: torch.Tensor):
    def train_step(self, phy: torch.Tensor, aux: torch.Tensor):
        # set model to train mode
        # only returns total loss. validation also returns each component's loss
        self.model.train()

        # get model predictions and losses
        if self.model.latent_layer_type == "GAUSS":
            # get model predictions
            phy_hat, aux_hat, latent = self.model((phy, aux))
            # compute recon loss
            phy_loss = self.loss_func(phy_hat, phy, self.char_weight, tip1_weight = 1.0)
            aux_loss = self.loss_func(aux_hat, aux)
            recon_loss = self.phy_loss_weight * phy_loss + (1-self.phy_loss_weight) * aux_loss
            # compute latent loss
            # mmd_loss = utils.mmd_loss(latent, self.std_norm)
            # vz_loss  = utils.vz_loss(latent, self.std_norm)
            std_norm = torch.randn(latent.shape).to(self.device)
            mmd_loss = utils.mmd_loss(latent, std_norm)
            if self.vz_lambda > 0:
                vz_loss  = utils.vz_loss(latent, std_norm)
            else:
                vz_loss = torch.tensor(0.0).to(self.device)
            
            latent_loss = self.mmd_lambda * mmd_loss + self.vz_lambda * vz_loss
            # compute total loss
            loss = recon_loss + latent_loss                        
        else:
            phy_hat, aux_hat = self.model((phy, aux))
            phy_loss = self.loss_func(phy_hat, phy, self.char_weight, tip1_weight = 1.0)
            aux_loss = self.loss_func(aux_hat, aux)
            loss = self.phy_loss_weight * phy_loss + (1-self.phy_loss_weight) * aux_loss
            
        # compute gradient
        loss.backward()

        # update model paremeters with gradient
        self.optimizer.step()

        # Clear the gradient for the next iteration, 
        # preventing accumulation from previous steps.
        self.optimizer.zero_grad()
        # return loss.item()
        return loss
        
        # return train_step

    # def _make_val_func(self):

    # def evaluate(phy: torch.Tensor, aux: torch.Tensor):
    def evaluate(self, phy: torch.Tensor, aux: torch.Tensor):
        self.model.eval()
        if self.model.latent_layer_type == "GAUSS":
            # get model predictions
            phy_hat, aux_hat, latent = self.model((phy, aux))
            # compute recon loss
            phy_loss    = self.loss_func(phy_hat, phy, self.char_weight, tip1_weight = 1.0)
            aux_loss    = self.loss_func(aux_hat, aux)
            recon_loss  = self.phy_loss_weight * phy_loss + (1-self.phy_loss_weight) * aux_loss
            
            # compute latent loss
            # mmd_loss = utils.mmd_loss(latent, self.std_norm)
            # vz_loss  = utils.vz_loss(latent, self.std_norm)
            std_norm = torch.randn(latent.shape).to(self.device)
            mmd_loss = utils.mmd_loss(latent, std_norm)
            if self.vz_lambda > 0.0:
                vz_loss  = utils.vz_loss(latent, std_norm)
            else:
                vz_loss = torch.tensor(0.0).to(self.device)
            
            latent_loss = self.mmd_lambda * mmd_loss + self.vz_lambda * vz_loss
            # compute total loss
            loss = recon_loss + latent_loss
            # return loss.item(), phy_loss.item(), aux_loss.item(), mmd_loss.item(), vz_loss.item()
            return loss, phy_loss, aux_loss, mmd_loss, vz_loss
        else:
            phy_hat, aux_hat = self.model((phy, aux))
            phy_loss = self.loss_func(phy_hat, phy, self.char_weight, tip1_weight = 1.0)
            aux_loss = self.loss_func(aux_hat, aux)                
            loss = self.phy_loss_weight * phy_loss + (1-self.phy_loss_weight) * aux_loss
            # return loss.item(), phy_loss.item(), aux_loss.item()
            return loss, phy_loss, aux_loss

        # return evaluate

        
    def plot_losses(self, out_prefix = "AElossplot"):
        fig = plt.figure(figsize=(11, 8))
        plt.plot(np.log10(self.losses), label='Training Loss', c="b")
        if self.val_loader:
            plt.plot(np.log10(self.val_losses), label='Validation Loss', c='r')
        plt.xlabel('Epochs')
        plt.ylabel('log10 Loss')
        plt.legend()        
        plt.grid(True)
        range_y = [min(np.log10(np.concat((self.losses, self.val_losses)))), 
                   max(np.log10(np.concat((self.losses, self.val_losses))))]
        plt.yticks(ticks = np.linspace(range_y[0], range_y[1], num = 20))
        plt.xticks(ticks=np.arange(0, len(self.losses), step=len(self.losses) // 10))
        plt.tight_layout()
        plt.savefig(out_prefix + ".loss.pdf", bbox_inches='tight')
        plt.close(fig)

        # plot each loss separately (only validation losses are recorded). 
        # create subplots for each loss component               
        num_subplots = 4 if self.model.latent_layer_type == "GAUSS" else 2
        fig, (axs1, axs2) = plt.subplots(num_subplots//2, 2, figsize=(11, 8), sharex=True)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        self.fill_in_loss_comp_fig(self.phy_losses, "phy", axs1[0])
        self.fill_in_loss_comp_fig(self.aux_losses, "aux", axs1[1])
        if self.model.latent_layer_type == "GAUSS":
            self.fill_in_loss_comp_fig(self.mmd_losses, "mmd", axs2[0])
            self.fill_in_loss_comp_fig(self.vz_losses, "vz", axs2[1])
        plt.savefig(out_prefix + ".component_loss.pdf", bbox_inches='tight')
        plt.close(fig)


                
    def fill_in_loss_comp_fig(self, val_losses, plot_label, ax):
        ax.plot(np.log10(val_losses), label=plot_label, c="b")
        ax.set_title(f"{plot_label} Loss")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Log10 Loss')
        ax.grid(True)
        ax.set_xticks(ticks=np.arange(0, len(val_losses), step=len(val_losses) // 10))
         
    
    
    def predict(self, phy: torch.Tensor, aux: torch.Tensor):
        self.model.eval() 
        phy = phy.to(self.device)
        aux = aux.to(self.device)
        if self.model.latent_layer_type == "GAUSS":
            phy_pred, aux_pred, latent = self.model((phy, aux))
        else:
            phy_pred, aux_pred = self.model((phy, aux))
        self.model.train()
        return phy_pred.detach().cpu().numpy(), aux_pred.detach().cpu().numpy()

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
        self.model.train()
        return decoded_tree, decoded_aux

        
