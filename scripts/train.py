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
    num_subset = 100000
    nworkers = 0
    rand_seed = np.random.randint(0,10000)
    num_epochs = 500
    batch_size = 1024

    nchannels = 9
    max_tips = 1000

    # mmd_weight = 5. # ppp - zzz
    # mmd_weight = 20. # aaa
    mmd_weight = 5. # xxx5
    phy_loss_weight = 0.85

    # get formated tree data
    with h5py.File(data_fn, "r") as f:
        # idx = np.where(f['aux_data_names'][...][0] == b'num_taxa')[0][0]

        phy_data = torch.tensor(f['phy_data'][0:num_subset,...], dtype = torch.float32)
        aux_data = torch.tensor(f['aux_data'][0:num_subset,...], dtype = torch.float32)
        # phy_data = np.array(f['phy_data'][0:num_subset,...], dtype = np.float32)
        # phy_data = phy_data.reshape((phy_data.shape[0],9,max_tips), order = "F")
        # phy_data = phy_data[:,0:nchannels,:]
        # phy_data = phy_data.reshape((phy_data.shape[0],-1), order = "F")
        # phy_data = torch.tensor(phy_data, dtype = torch.float32)
        # aux_data = torch.tensor(f['aux_data'][0:num_subset,idx], dtype = torch.float32).view(-1,1)

        test_phy_data = torch.tensor(f['phy_data'][num_subset:(num_subset + 500),...], dtype = torch.float32)
        test_aux_data = torch.tensor(f['aux_data'][num_subset:(num_subset + 500),...], dtype = torch.float32)
        # test_phy_data = np.array(f['phy_data'][num_subset:(num_subset + 500),...], dtype = np.float32)
        # test_phy_data = test_phy_data.reshape((test_phy_data.shape[0],9,max_tips), order = "F")
        # test_phy_data = test_phy_data[:,0:nchannels,:]
        # test_phy_data = test_phy_data.reshape((test_phy_data.shape[0],-1), order = "F")
        # test_phy_data = torch.tensor(test_phy_data, dtype = torch.float32)
        # test_aux_data = torch.tensor(f['aux_data'][num_subset:(num_subset + 500),idx], dtype = torch.float32).view(-1,1)

    
    # checking how much aux_data is helping encode tree patterns
    # rand_idx = torch.randperm(aux_data.shape[0])
    # aux_data = aux_data[rand_idx]

    # create Data container
    ae_data = ph.DataProcessors.AEData(data = (phy_data, aux_data), 
                                        prop_train = 0.8,  
                                        nchannels  = nchannels)
    ae_data.save_normalizers(out_prefix)


    ###############################################################################
    # TODO: CHECKING CBLV impact of normalization on terminal vs internal branches
    ###############################################################################

    # create data loaders
    trn_loader, val_loader = ae_data.get_dataloaders(batch_size  = batch_size, 
                                                     shuffle     = True, 
                                                     num_workers = nworkers)

    # create model
    ae_model  = ph.PhyloAEModel.AECNN(num_structured_input_channel  = ae_data.nchannels, 
                                      structured_input_width        = ae_data.phy_width,
                                      unstructured_input_width      = ae_data.aux_width,
                                      stride                        = [2,8],
                                      kernel                        = [3,9],
                                      out_channels                  = [16,64],
                                      latent_layer_type             = "GAUSS",
                                      )

    # create Trainer
    tree_autoencoder = PhyloAutoencoder(model     = ae_model, 
                                        optimizer = torch.optim.Adam(ae_model.parameters()), 
                                        loss_func = torch.nn.MSELoss(), 
                                        phy_loss_weight = phy_loss_weight,
                                        mmd_weight = mmd_weight,
                                        )
    
    # Load data loaders and Train model and plot
    tree_autoencoder.set_data_loaders(train_loader=trn_loader, val_loader=val_loader)
    tree_autoencoder.train(num_epochs = num_epochs, seed = rand_seed)
    tree_autoencoder.plot_losses().savefig("AElossplot.pdf")
    tree_autoencoder.save_model(out_prefix + ".ae_trained.pt")


    # Test data
    # save true data and predictions pushed through the autoencoder
    # save true data
    phy_true_df = pd.DataFrame(test_phy_data.numpy())
    phy_true_df.to_csv(out_prefix + ".phy_true.cblv", header = False, index = False)
    aux_true_df = pd.DataFrame(test_aux_data.numpy())
    aux_true_df.to_csv(out_prefix + ".aux_true.csv", header = False, index = False)

    # make predictions of test data with trained model
    phy_normalizer, aux_normalizer = ae_data.get_normalizers()
    phydat = phy_normalizer.transform(test_phy_data)
    auxdat = aux_normalizer.transform(test_aux_data)

    phydat = phydat.reshape((phydat.shape[0], ae_data.nchannels, ae_data.phy_width), order = "F")
    phydat = torch.Tensor(phydat)
    auxdat = torch.Tensor(auxdat)
    phy_pred, auxpred = tree_autoencoder.predict(phydat, auxdat)
    # transform and flatten data
    phy_pred = phy_normalizer.inverse_transform(phy_pred.reshape((phy_pred.shape[0], -1), order = "F"))
    auxpred  = aux_normalizer.inverse_transform(auxpred)

    # save predictions
    phy_pred_df = pd.DataFrame(phy_pred)
    phy_pred_df.to_csv(out_prefix + ".phy_pred.cblv", header = False, index = False)
    aux_pred_df = pd.DataFrame(auxpred)
    aux_pred_df.to_csv(out_prefix + ".aux_pred.csv", header  = False, index = False)


    # for PCA analysis of a sample of training trees
    # make encoded tree file
    rand_idx = np.random.randint(0, ae_data.prop_train * num_subset, size = min(5000, num_subset))
    latent_dat = tree_autoencoder.tree_encode(torch.Tensor(ae_data.norm_train_phy_data[rand_idx,...]), 
                                              torch.Tensor(ae_data.norm_train_aux_data[rand_idx,...]))
    latent_dat_df = pd.DataFrame(latent_dat.detach().to('cpu').numpy(), columns = None, index = None)
    latent_dat_df.to_csv(out_prefix + ".traindat_latent.csv", header = False, index = False)


if __name__ == "__main__":
    main()
