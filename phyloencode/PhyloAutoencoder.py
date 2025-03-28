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


class PhyloAutoencoder(object):
    def __init__(self, model, optimizer, loss_func, phy_loss_weight=0.5, mmd_weight=None):
        '''
            model is an object from torch.model
            optimizer is an object from torch.optim
            loss_func is an object from torch.nn
            phy_loss_weight is a float between 0 and 1
            mmd_weight is a float >= 0
        '''

        self.device    = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model     = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.phy_loss_weight = phy_loss_weight

        # TODO: do a proper error handing at some point 
        # (should be passed as a parameter)
        if model.latent_layer_type == "GAUSS":
            self.mmd_weight = mmd_weight
            self.mmd_weight_max = mmd_weight


        self.train_loader = None
        self.val_loader   = None

        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # step functions. These perform a single gradient descent step:
        # (model(x), then backward) given a batch
        # updates internal state. returns the loss for the batch
        self.train_step = self._make_train_step_func()
        self.val_step   = self._make_val_func()
        # self.writer = None

    def train(self, num_epochs, seed = 1):
        self.set_seed(seed)
        for epoch in range(num_epochs):
            self.total_epochs += 1
            epoch_train_loss = self._mini_batch()

            with torch.no_grad():
                if self.model.latent_layer_type == "GAUSS":
                    epoch_validation_loss, phy_loss, aux_loss, mmd_loss = self._mini_batch(validation=True)
                    self.mmd_weight = min(self.mmd_weight_max, 1.5 * self.mmd_weight + 0.1)

                    print(f"epoch {epoch},  " +
                            f"loss: {epoch_validation_loss:.4f},  " +
                            f"phy loss: {phy_loss:.4f},  " +
                            f"aux loss: {aux_loss:.4f},  " +
                            f"MMD2 loss: {mmd_loss:.4f}")
                else:
                    epoch_validation_loss, phy_loss, aux_loss = self._mini_batch(validation=True)
                    print(f"epoch {epoch},  " +
                            f"loss: {epoch_validation_loss:.4f},  " +
                            f"phy loss: {phy_loss:.4f},  " +
                            f"aux loss: {aux_loss:.4f}")


            self.losses.append(epoch_train_loss)
            self.val_losses.append(epoch_validation_loss)
               


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
        # Uses self.step_function to perform SGD step for each batch
        # returns the mean loss from the mini batch steps
        # If validation = True, then returns phy, aux, and mmd loss.

        # set data loader and step function
        if validation:
            data_loader   = self.val_loader
            step_function = self.val_step
        else:
            data_loader   = self.train_loader
            step_function = self.train_step

        if data_loader == None:
            return None

        # perform step and return loss
        # loop through all batches of train or val data
        mini_batch_losses = []
        phy_mini_batch_losses = []
        aux_mini_batch_losses = []
        mmd_mini_batch_losses = []
        for phy_batch, aux_batch in data_loader:
            phy_batch = phy_batch.to(self.device)
            aux_batch = aux_batch.to(self.device)
            if not validation:
                loss = step_function(phy_batch, aux_batch)
            elif self.model.latent_layer_type == "GAUSS" and validation:
                loss, phy_loss, aux_loss, mmd_loss = step_function(phy_batch, aux_batch)
                phy_mini_batch_losses.append(phy_loss)
                aux_mini_batch_losses.append(aux_loss)
                mmd_mini_batch_losses.append(mmd_loss)
            else:
                loss, phy_loss, aux_loss = step_function(phy_batch, aux_batch)
                phy_mini_batch_losses.append(phy_loss)
                aux_mini_batch_losses.append(aux_loss)

            mini_batch_losses.append(loss)

        # return the mean loss over all batches
        if self.model.latent_layer_type == "GAUSS" and validation:
            return  np.mean(mini_batch_losses), \
                    np.mean(phy_mini_batch_losses), \
                    np.mean(aux_mini_batch_losses), \
                    np.mean(mmd_mini_batch_losses)
        elif validation:    
            return np.mean(mini_batch_losses), np.mean(phy_mini_batch_losses), np.mean(aux_mini_batch_losses)
        else:
            return np.mean(mini_batch_losses)

    def _make_train_step_func(self):

        def train_step(phy: torch.Tensor, aux: torch.Tensor):
            # set model to train mode
            # only returns total loss. validation also returns each component's loss
            self.model.train()

            # get model predictions and losses
            if self.model.latent_layer_type == "GAUSS":
                phy_hat, aux_hat, latent = self.model((phy, aux))
                mmd_loss = utils.mmd_loss(latent)
                phy_loss = self.loss_func(phy_hat, phy)
                aux_loss = self.loss_func(aux_hat, aux)
                loss = self.phy_loss_weight * phy_loss + \
                    (1 - self.phy_loss_weight) * aux_loss + \
                        self.mmd_weight * mmd_loss
            else:
                phy_hat, aux_hat = self.model((phy, aux))
                phy_loss = self.loss_func(phy_hat, phy)
                aux_loss = self.loss_func(aux_hat, aux)
                # get weighted average of losses (torch.Tensor object)
                loss = self.phy_loss_weight * phy_loss + \
                    (1 - self.phy_loss_weight) * aux_loss
                
            # compute gradient
            loss.backward()

            # update model paremeters with gradient
            self.optimizer.step()

            # Clear the gradient for the next iteration, 
            # preventing accumulation from previous steps.
            self.optimizer.zero_grad()
            return loss.item()
        
        return train_step

    def _make_val_func(self):

        def evaluate(phy: torch.Tensor, aux: torch.Tensor):
            self.model.eval()
            if self.model.latent_layer_type == "GAUSS":
                phy_hat, aux_hat, latent = self.model((phy, aux))
                mmd_loss = utils.mmd_loss(latent)
                phy_loss = self.loss_func(phy_hat, phy)
                aux_loss = self.loss_func(aux_hat, aux)                
                loss = self.phy_loss_weight * phy_loss + \
                    (1 - self.phy_loss_weight) * aux_loss + \
                        self.mmd_weight * mmd_loss
                return loss.item(), phy_loss.item(), aux_loss.item(), mmd_loss.item()
            else:
                phy_hat, aux_hat = self.model((phy, aux))
                phy_loss = self.loss_func(phy_hat, phy)
                aux_loss = self.loss_func(aux_hat, aux)                
                loss = self.phy_loss_weight * phy_loss + \
                    (1 - self.phy_loss_weight) * aux_loss
                return loss.item(), phy_loss.item(), aux_loss.item()

        return evaluate

    def predict(self, phy: torch.Tensor, aux: torch.Tensor):
        self.model.eval() 
        phy = phy.to(self.device)
        aux = aux.to(self.device)

        if self.model.latent_layer_type == "GAUSS":
            # phy_pred, aux_pred, mu_pred, logv_pred = self.model((phy, aux))
            phy_pred, aux_pred, latent = self.model((phy, aux))
        else:
            phy_pred, aux_pred = self.model((phy, aux))

        # x_tensor = torch.as_tensor(x).float().to(self.device)
        self.model.train()
        return phy_pred.detach().cpu().numpy(), aux_pred.detach().cpu().numpy()

    def to_device(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            print(f"Didn't work, sending to {self.device} instead.")

    def set_data_loaders(self, train_loader, val_loader = None):
        self.train_loader = train_loader
        self.val_loader   = val_loader

    def plot_losses(self):
        fig = plt.figure(figsize=(11, 8))
        plt.plot(np.log10(self.losses), label='Training Loss', c="b")
        if self.val_loader:
            plt.plot(np.log10(self.val_losses), label='Validation Loss', c='r')
        plt.xlabel('Epochs')
        plt.ylabel('log Loss')
        plt.legend()        
        plt.grid(True) 
        plt.xticks(ticks=np.arange(0, len(self.losses), step=len(self.losses) // 10))
        plt.tight_layout()
        return fig

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
        
        reshaped_shared_latent = combined_latent.view(-1, self.model.num_structured_latent_channels, 
                                                    self.model.reshaped_shared_latent_width)
        
        # get combined latent output
        if self.model.latent_layer_type   == "CNN":
            shared_latent_out = self.model.latent_layer(reshaped_shared_latent)

        elif self.model.latent_layer_type == "GAUSS":
            shared_latent_out = self.model.latent_layer(combined_latent)

        elif self.model.latent_layer_type == "DENSE":
            shared_latent_out = self.model.latent_layer(combined_latent)

        self.model.train()

        return(shared_latent_out.flatten(start_dim=1))

    def get_latent_shape(self):
        return self.model.num_structured_latent_channels, self.model.reshaped_shared_latent_width

    def latent_decode(self, encoded_tree):
        self.model.eval()
        encoded_tree = encoded_tree.to(self.device)
        decoded_tree = self.model.structured_decoder(encoded_tree)
        decoded_aux  = self.model.unstructured_decoder(encoded_tree.flatten(start_dim=1))
        self.model.train()
        return decoded_tree, decoded_aux

        
