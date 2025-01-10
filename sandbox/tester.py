#!/usr/bin/env python3

import torch
import torch.utils.data as td
import numpy as np
import pandas as pd
import sklearn as sk
import phyloencode as ph
import sys
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import phyloencode as ph
# from phyloencode.DataProcessors import AEData
# from phyloencode.PhyloAEModel
from phyloencode import PhyloAutoencoder as pa
from phyloencode import utils

with h5py.File("test_data/peak_time.train.hdf5", "r") as f:
    phy_data = torch.tensor(f['phy_data'][0:15000,...], dtype = torch.float32)
    aux_data = torch.tensor(f['aux_data'][0:15000,...], dtype = torch.float32)
    test_phy_data = torch.tensor(f['phy_data'][45000:45100,...], dtype = torch.float32)
    test_aux_data = torch.tensor(f['aux_data'][45000:45100,...], dtype = torch.float32)


# checking how much aux_data is helping encode tree patterns
# rand_idx = torch.randperm(aux_data.shape[0])
# aux_data = aux_data[rand_idx]


# Data container
ae_data = ph.DataProcessors.AEData(data = (phy_data, aux_data), prop_train = 0.8,  nchannels  = 7)


# data loaders
train_dataloader, val_dataloader = ae_data.get_dataloaders(batch_size=32, 
                                                           shuffle=True, 
                                                           num_workers=0)

# model
ae_model  = ph.PhyloAEModel.AECNN(ae_data.nchannels, 
                                ae_data.phy_width, 
                                ae_data.aux_width, 
                                stride = [4,8,8],
                                kernel = [5,9,9],
                                out_channels=[16,24,32])

# Trainer
loss_fx   = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ae_model.parameters())
tree_autoencoder = pa.PhyloAutoencoder(model=ae_model, 
                                       optimizer=optimizer, 
                                       loss_func=loss_fx, 
                                       phy_loss_weight=1.0)


# Train
rand_seed = np.random.randint(0,10000)
tree_autoencoder.set_data_loaders(train_loader=train_dataloader, val_loader=val_dataloader)
tree_autoencoder.train(num_epochs = 20, seed = rand_seed)

epoch_loss_figure = tree_autoencoder.plot_losses().savefig("AElossplot.pdf")

print("Training Finished")


# Test data
phy_normalizer, aux_normalizer = ae_data.get_normalizers()
phydat   = phy_normalizer.transform(test_phy_data)
auxdat   = aux_normalizer.transform(test_aux_data)
phydat = phydat.reshape((phydat.shape[0], ae_data.nchannels, int(phydat.shape[1]/ae_data.nchannels)), order = "F")
phydat = torch.Tensor(phydat)
auxdat = torch.Tensor(auxdat)
phy_pred, auxpred = tree_autoencoder.predict(phydat, auxdat)
phy_pred = phy_normalizer.inverse_transform(phy_pred.reshape((phy_pred.shape[0], -1), order = "F"))

# print out comparison of a part of an input tree and it passed through the filter
for i in range(28,38):
    print(test_phy_data.numpy()[i,18:24])
    print(np.array(phy_pred[i,18:24]))
    print("    ")

