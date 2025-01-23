#!/usr/bin/env python3
# test a pretrained model
import phyloencode as ph
from phyloencode.PhyloAutoencoder import PhyloAutoencoder
import torch
import joblib
import h5py
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import argparse
import re

def file_exists(fname):
    "Return error if no file"
    if not os.path.isfile(fname):
        raise argparse.ArgumentTypeError(f"File '{fname}' does not exist.")
    return fname

def main ():
    cmd = argparse.ArgumentParser(description="Encode phylogenetic trees and auxiliary data with trained autodencoder.")
    cmd.add_argument("-m", "--model", required=True, help="Path to the model file")
    cmd.add_argument("-p", "--phy-normalizer", required=True, help="Path to the phy_normalizer.pkl file")
    cmd.add_argument("-a", "--aux-normalizer", required=True, help="Path to the aux_normalizer.pkl file")
    cmd.add_argument("-t", "--tree-data", required=True, help="Path to the phyddle formated tree h5py file")
    cmd.add_argument("-o", "--out-prefix", required=False, help="Path to out file prefix")

    args = cmd.parse_args()

    ae_model_fn       = file_exists(args.model)
    phy_normalizer_fn = file_exists(args.phy_normalizer)
    aux_normalizer_fn = file_exists(args.aux_normalizer)
    tree_data_fn      = file_exists(args.tree_data)
    
    if args.out_prefix is None:
        out_file_prefix = tree_data_fn.split('/')[-1].split('.')[0]
    else:
        out_file_prefix = args.out_prefix

    # load trained model and normalizers and create PhyloAutoencoder object
    ae_model       = torch.load(ae_model_fn)
    phy_normalizer = joblib.load(phy_normalizer_fn)
    aux_normalizer = joblib.load(aux_normalizer_fn)

    tree_autoencoder = PhyloAutoencoder(model     = ae_model, 
                                        optimizer = torch.optim.Adam(ae_model.parameters()), 
                                        loss_func = torch.nn.MSELoss(), 
                                        phy_loss_weight = 1.0)

    # import test data
    with h5py.File(tree_data_fn, "r") as f:
        test_phy_data = torch.tensor(f['phy_data'][...], dtype = torch.float32)
        test_aux_data = torch.tensor(f['aux_data'][...], dtype = torch.float32)


    # make predictions with trained model
    phydat = phy_normalizer.transform(test_phy_data)
    auxdat = aux_normalizer.transform(test_aux_data)
    phydat = phydat.reshape((phydat.shape[0], ae_model.num_structured_input_channel, 
                            int(phydat.shape[1]/ae_model.num_structured_input_channel)), order = "F")
    phydat = torch.Tensor(phydat)
    auxdat = torch.Tensor(auxdat)
    phy_pred, auxpred = tree_autoencoder.predict(phydat, auxdat)
    phy_pred = phy_normalizer.inverse_transform(phy_pred.reshape((phy_pred.shape[0], -1), order = "F"))

    # make encoded tree file
    latent_dat = tree_autoencoder.tree_encode(phydat, auxdat)
    latent_dat_df = pd.DataFrame(latent_dat.detach().to('cpu').numpy(), columns = None, index = None)
    latent_dat_df.to_csv(out_file_prefix + ".ae_encoded.csv", header = False, index = False)

    print("Wrote to: " + out_file_prefix + ".ae_encoded.csv")

if __name__ == "__main__":
    main()