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

import phyloencode as ph
from phyloencode import DataProcessors as dataproc
from phyloencode import PhyloAEModel as aem
from phyloencode import PhyloAutoencoder as pa

with h5py.File("test_data/peak_time.train.hdf5", "r") as f:
    phy_data = torch.tensor(f['phy_data'][0:10000,...], dtype=torch.float32)
    aux_data = torch.tensor(f['aux_data'][0:10000,...], dtype=torch.float32)


phy_width = phy_data.shape[1]
aux_width = aux_data.shape[1]

num_train = int(0.8 * phy_data.shape[0])

data = np.hstack((phy_data, aux_data))

# split data 
train_data, val_data = train_test_split(data, 
                                                                  train_size = num_train, 
                                                                  shuffle=True)
train_phy_data = train_data[:,:phy_width]
train_aux_data = train_data[:,phy_width:]
val_phy_data   = val_data[:,:phy_width]
val_aux_data   = val_data[:,phy_width:]



# standardize train data
phy_ss = StandardScaler()
aux_ss = StandardScaler()
phy_normalizer = phy_ss.fit(train_phy_data)
aux_normalizer = aux_ss.fit(train_aux_data)
norm_train_phy_data = phy_normalizer.transform(train_phy_data)
norm_train_aux_data = aux_normalizer.transform(train_aux_data)
norm_val_phy_data   = phy_normalizer.transform(val_phy_data)
norm_val_aux_data   = aux_normalizer.transform(val_aux_data)

# reshape phy data (in this case channels are 3 locs and 4 cblv dimensions)
nchannels = 7
assert(train_phy_data.shape[1] % nchannels == 0)
norm_train_phy_data = norm_train_phy_data.reshape((norm_train_phy_data.shape[0], 
                                                   nchannels, 
                                                   int(norm_train_phy_data.shape[1]/nchannels)),
                                                 order = "F")
norm_val_phy_data   = norm_val_phy_data.reshape((norm_val_phy_data.shape[0], 
                                                 nchannels, 
                                                 int(norm_val_phy_data.shape[1]/nchannels)),
                                                 order = "F")
phy_width = norm_train_phy_data.shape[2]
aux_width = norm_train_aux_data.shape[1]

# create Datasets. __getitem__() returns a tuple (phy, aux)
train_dataset = dataproc.TreeDataSet(norm_train_phy_data, norm_train_aux_data)
val_dataset   = dataproc.TreeDataSet(norm_val_phy_data,   norm_val_aux_data)

# create model
ae_model  = aem.PhyloAEModelCNN(phy_width, nchannels, aux_width, 10, 10)
loss_fx   = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ae_model.parameters())

# data loaders
train_dataloader = td.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_dataloader   = td.DataLoader(val_dataset)

# Train
tree_autoencoder = pa.PhyloAutoencoder(model=ae_model, optimizer=optimizer, loss_func=loss_fx)
tree_autoencoder.set_data_loaders(train_loader=train_dataloader, val_loader=val_dataloader)
tree_autoencoder.train(num_epochs=20, seed = 1)
# tree_autoencoder.plot_losses()

print("Finished")
