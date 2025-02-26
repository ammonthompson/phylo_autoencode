#!/usr/bin/env python3

import torch
import torch.utils.data as td
import numpy as np
import pandas as pd
import sklearn as sk
import phyloencode as ph
import sys
import h5py
import argparse

import phyloencode as ph
from phyloencode import utils
from phyloencode.PhyloAutoencoder import PhyloAutoencoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--trn_data", required = True, help = "Training data in hdf5 format.")
    parser.add_argument("-o", "--out_prefix", required = True, help = "Output prefix.")

    args = parser.parse_args()
    data_fn = args.trn_data
    out_prefix = args.out_prefix


    # not used. dataset too small
    # num_cpus = multiprocessing.cpu_count()
    # num_workers = 0 if (num_cpus - 4) < 0 else num_cpus - 4
    num_subset = 48000
    nworkers = 0
    rand_seed = np.random.randint(0,10000)

    # get formated tree data
    with h5py.File(data_fn, "r") as f:
        phy_data = torch.tensor(f['phy_data'][0:num_subset,...], dtype = torch.float32)
        aux_data = torch.tensor(f['aux_data'][0:num_subset,...], dtype = torch.float32)
        test_phy_data = torch.tensor(f['phy_data'][num_subset:(num_subset + 500),...], dtype = torch.float32)
        test_aux_data = torch.tensor(f['aux_data'][num_subset:(num_subset + 500),...], dtype = torch.float32)


    # checking how much aux_data is helping encode tree patterns
    # rand_idx = torch.randperm(aux_data.shape[0])
    # aux_data = aux_data[rand_idx]


    # create Data container
    ae_data = ph.DataProcessors.AEData(data = (phy_data, aux_data), 
                                        prop_train = 0.8,  
                                        nchannels  = 7)
    ae_data.save_normalizers(out_prefix)

    # create data loaders
    trn_loader, val_loader = ae_data.get_dataloaders(batch_size  = 64, 
                                                     shuffle     = True, 
                                                     num_workers = nworkers)

    # create model
    ae_model  = ph.PhyloAEModel.AECNN(num_structured_input_channel = ae_data.nchannels, 
                                      structured_input_width   = ae_data.phy_width,  # Input width for structured data
                                      unstructured_input_width = ae_data.aux_width,
                                      stride        = [2,2,7,9],
                                      kernel        = [3,3,8,10],
                                      out_channels  = [8,16,32,32],
                                      latent_layer_type = "GAUSS")

    # create Trainer
    tree_autoencoder = PhyloAutoencoder(
                                        model     = ae_model, 
                                        optimizer = torch.optim.Adam(ae_model.parameters()), 
                                        loss_func = torch.nn.MSELoss(), 
                                        # loss_func = utils.get_vae_loss_function(), 
                                        phy_loss_weight = 1.0,
                                        mmd_weight = 1,
                                        )

    # Load data loaders and Train model and plot
    tree_autoencoder.set_data_loaders(train_loader=trn_loader, val_loader=val_loader)
    tree_autoencoder.train(num_epochs = 150, seed = rand_seed)
    tree_autoencoder.plot_losses().savefig("AElossplot.pdf")
    tree_autoencoder.save_model(out_prefix + ".ae_trained.pt")


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
    phy_true_df.to_csv(out_prefix + ".phy_true.cblv", header = False)

    phy_pred_df = pd.DataFrame(phy_pred)
    phy_pred_df.to_csv(out_prefix + ".phy_pred.cblv", header = False)


    # for PCA analysis of a sample of training trees
    # make encoded tree file
    rand_idx = np.random.randint(0, ae_data.prop_train * num_subset, size = min(5000, num_subset))
    latent_dat = tree_autoencoder.tree_encode(torch.Tensor(ae_data.norm_train_phy_data[rand_idx,...]), 
                                              torch.Tensor(ae_data.norm_train_aux_data[rand_idx,...]))

    latent_dat_df = pd.DataFrame(latent_dat.detach().to('cpu').numpy(), columns = None, index = None)
    latent_dat_df.to_csv(out_prefix + ".traindat_latent.csv", header = False, index = False)

if __name__ == "__main__":
    main()
