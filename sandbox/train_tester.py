#!/usr/bin/env python3

import torch
import torch.utils.data as td
import numpy as np
import pandas as pd
import sklearn as sk
import phyloencode as ph
import sys
import h5py

import phyloencode as ph
from phyloencode.PhyloAutoencoder import PhyloAutoencoder

data_fn = sys.argv[1]

# not used. dataset too small
# num_cpus = multiprocessing.cpu_count()
# num_workers = 0 if (num_cpus - 4) < 0 else num_cpus - 4
nworkers = 0
rand_seed = np.random.randint(0,10000)

# get formated tree data
with h5py.File(data_fn, "r") as f:
    phy_data = torch.tensor(f['phy_data'][0:45000,...], dtype = torch.float32)
    aux_data = torch.tensor(f['aux_data'][0:45000,...], dtype = torch.float32)
    test_phy_data = torch.tensor(f['phy_data'][45000:45100,...], dtype = torch.float32)
    test_aux_data = torch.tensor(f['aux_data'][45000:45100,...], dtype = torch.float32)


# checking how much aux_data is helping encode tree patterns
# rand_idx = torch.randperm(aux_data.shape[0])
# aux_data = aux_data[rand_idx]


# create Data container
ae_data = ph.DataProcessors.AEData(data = (phy_data, aux_data), 
                                   prop_train = 0.8,  
                                   nchannels  = 7)

# create data loaders
trn_loader, val_loader = ae_data.get_dataloaders(batch_size  = 32, 
                                                 shuffle     = True, 
                                                 num_workers = nworkers)

# create model
ae_model  = ph.PhyloAEModel.AECNN(num_structured_input_channel = ae_data.nchannels, 
                                  structured_input_width   = ae_data.phy_width,  # Input width for structured data
                                  unstructured_input_width = ae_data.aux_width,
                                  stride        = [2,2,5,5],
                                  kernel        = [3,3,6,6],
                                  out_channels  = [32,32,32,32])

# create Trainer
tree_autoencoder = PhyloAutoencoder(model     = ae_model, 
                                    optimizer = torch.optim.Adam(ae_model.parameters()), 
                                    loss_func = torch.nn.MSELoss(), 
                                    phy_loss_weight = 1.0)

# Train model
tree_autoencoder.set_data_loaders(train_loader=trn_loader, val_loader=val_loader)
tree_autoencoder.train(num_epochs = 25, seed = rand_seed)

# plot
epoch_loss_figure = tree_autoencoder.plot_losses().savefig("AElossplot.pdf")

# save trained model
tree_autoencoder.save_model("ae_trained.pt")
ae_data.save_normalizers("ae_test")


# Test data
phy_normalizer, aux_normalizer = ae_data.get_normalizers()
phydat = phy_normalizer.transform(test_phy_data)
auxdat = aux_normalizer.transform(test_aux_data)
phydat = phydat.reshape((phydat.shape[0], ae_data.nchannels, 
                         int(phydat.shape[1]/ae_data.nchannels)), order = "F")
phydat = torch.Tensor(phydat)
auxdat = torch.Tensor(auxdat)
phy_pred, auxpred = tree_autoencoder.predict(phydat, auxdat)
phy_pred = phy_normalizer.inverse_transform(phy_pred.reshape((phy_pred.shape[0], -1), order = "F"))

phy_true_df = pd.DataFrame(test_phy_data.numpy())
phy_true_df.to_csv("phy_true.cblv", header = False)

phy_pred_df = pd.DataFrame(phy_pred)
phy_pred_df.to_csv("phy_pred.cblv", header = False)


# tree latent space check on test trees
encoded_test_trees = tree_autoencoder.tree_encode(phydat, auxdat)

# for PCA analysis of a sample of training trees
# make encoded tree file
rand_idx = np.random.randint(0, ae_data.prop_train * 15000, size = 5000)

latent_dat = tree_autoencoder.tree_encode(torch.Tensor(ae_data.norm_train_phy_data[rand_idx,...]), 
                                          torch.Tensor(ae_data.norm_train_aux_data[rand_idx,...]))

latent_dat_df = pd.DataFrame(latent_dat.detach().to('cpu').numpy(), columns = None, index = None)
latent_dat_df.to_csv("traindat_latent_for_pca.csv", header = False, index = False)