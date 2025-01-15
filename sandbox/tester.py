#!/usr/bin/env python3

import torch
import torch.utils.data as td
import numpy as np
import pandas as pd
import sklearn as sk
import phyloencode as ph
import sys
import h5py
import multiprocessing
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import phyloencode as ph
from phyloencode.PhyloAutoencoder import PhyloAutoencoder

with h5py.File("test_data/peak_time.train.hdf5", "r") as f:
    phy_data = torch.tensor(f['phy_data'][0:4500,...], dtype = torch.float32)
    aux_data = torch.tensor(f['aux_data'][0:4500,...], dtype = torch.float32)
    test_phy_data = torch.tensor(f['phy_data'][45000:45100,...], dtype = torch.float32)
    test_aux_data = torch.tensor(f['aux_data'][45000:45100,...], dtype = torch.float32)


# checking how much aux_data is helping encode tree patterns
# rand_idx = torch.randperm(aux_data.shape[0])
# aux_data = aux_data[rand_idx]

num_cpus = multiprocessing.cpu_count()

# create Data container
ae_data = ph.DataProcessors.AEData(data = (phy_data, aux_data), 
                                   prop_train = 0.8,  
                                   nchannels  = 7)

# create data loaders
trn_loader, val_loader = ae_data.get_dataloaders(batch_size  = 32, 
                                                 shuffle     = True, 
                                                 num_workers = 0)

# create model
ae_model  = ph.PhyloAEModel.AECNN(ae_data,
                                  stride       = [2,10,10],
                                  kernel       = [3,11,11],
                                  out_channels = [16,18,20])

# create Trainer
tree_autoencoder = PhyloAutoencoder(model     = ae_model, 
                                    optimizer = torch.optim.Adam(ae_model.parameters()), 
                                    loss_func = torch.nn.MSELoss(), 
                                    phy_loss_weight = 1.0)

# Train model
rand_seed = np.random.randint(0,10000)
tree_autoencoder.set_data_loaders(train_loader=trn_loader, val_loader=val_loader)
tree_autoencoder.train(num_epochs = 4, seed = rand_seed)

# plot
epoch_loss_figure = tree_autoencoder.plot_losses().savefig("AElossplot.pdf")


# Test data
phy_normalizer, aux_normalizer = ae_data.get_normalizers()
phydat = phy_normalizer.transform(test_phy_data)
auxdat = aux_normalizer.transform(test_aux_data)
phydat = phydat.reshape((phydat.shape[0], ae_data.nchannels, int(phydat.shape[1]/ae_data.nchannels)), order = "F")
phydat = torch.Tensor(phydat)
auxdat = torch.Tensor(auxdat)
phy_pred, auxpred = tree_autoencoder.predict(phydat, auxdat)
phy_pred = phy_normalizer.inverse_transform(phy_pred.reshape((phy_pred.shape[0], -1), order = "F"))

# print out comparison of a part of an input tree and it passed through the filter
for i in range(50,53):
    print(test_phy_data.numpy()[i,18:24])
    print(np.array(phy_pred[i,18:24]))
    print("    ")

phy_true_df = pd.DataFrame(test_phy_data.numpy())
phy_true_df.to_csv("phy_true.cblv", header = False)

phy_pred_df = pd.DataFrame(phy_pred)
phy_pred_df.to_csv("phy_pred.cblv", header = False)


# tree latent space check on test trees
encoded_trees = tree_autoencoder.tree_encode(phydat, auxdat)

# for PCA analysis of a sample of training trees
rand_idx = np.random.randint(0, ae_data.prop_train * 4500, size = 500)

latent_dat = tree_autoencoder.tree_encode(torch.Tensor(ae_data.norm_train_phy_data[rand_idx,...]), 
                                          torch.Tensor(ae_data.norm_train_aux_data[rand_idx,...]))

latent_dat_df = pd.DataFrame(latent_dat.detach().to('cpu').numpy(), columns = None, index = None)
latent_dat_df.to_csv("train_latent_for_pca.csv")