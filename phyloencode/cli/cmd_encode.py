#!/usr/bin/env python3
# test a pretrained model
from phyloencode.PhyloAutoencoder import PhyloAutoencoder
import torch
import joblib
import h5py
import pandas as pd
import os
import argparse
import numpy as np
import os

def file_exists(fname):
    "Return error if no file"
    if not os.path.isfile(fname):
        raise argparse.ArgumentTypeError(f"File '{fname}' does not exist.")
    return fname

def main ():
    cmd = argparse.ArgumentParser(description="Encode phylogenetic trees and auxiliary data with trained autodencoder.")
    cmd.add_argument("-m", "--model", required=True, help="Path to the trained model.pt file")
    cmd.add_argument("-p", "--phy-normalizer", required=True, help="Path to the phy_normalizer.pkl file")
    cmd.add_argument("-a", "--aux-normalizer", required=True, help="Path to the aux_normalizer.pkl file")
    cmd.add_argument("-t", "--tree-data", required=True, help="Path to the phyddle formated tree CBLV(S). If using a phyddle -s F output hdf5, " \
    "use the key 'phy_data' for the tree data and 'aux_data' for the auxiliary data. If using csv, for the cblv tree data, then" \
        " use the -s [--aux-data] flag to specify the auxiliary data file.")
    cmd.add_argument("-s", "--aux-data", required=False, help="Path to the auxiliary data if using csv format. " )
    cmd.add_argument("-nc", "--num-channels", required=False, type=int, help="Number of channels in the data. Default is 2.")
    cmd.add_argument("-nt", "--num-tips", required=False, type=int,help="Max number of tips in the data. Default is 500.")
    cmd.add_argument("-o", "--out-prefix", required=False, help="Path to out file prefix")

    args = cmd.parse_args()

    ae_model_fn       = file_exists(args.model)
    phy_normalizer_fn = file_exists(args.phy_normalizer)
    aux_normalizer_fn = file_exists(args.aux_normalizer)
    tree_data_fn      = file_exists(args.tree_data)
    aux_data_fn       = file_exists(args.aux_data) if args.aux_data is not None else None
    num_channels      = args.num_channels if args.num_channels is not None else None
    num_tips          = args.num_tips if args.num_tips is not None else None

    if (num_tips is not None and num_channels is None) or (num_channels is not None and num_tips is None):
        raise ValueError("If using num_tips, then num_channels must also be specified. " \
        "If using num_channels, then num_tips must also be specified.")



    
    if args.out_prefix is None:
        out_file_prefix = tree_data_fn.split('/')[-1].split('.')[0]
    else:
        out_file_prefix = args.out_prefix

    # load trained model and normalizers and create PhyloAutoencoder object
    ae_model       = torch.load(ae_model_fn, weights_only=False)
    phy_normalizer = joblib.load(phy_normalizer_fn)
    aux_normalizer = joblib.load(aux_normalizer_fn)

    tree_autoencoder = PhyloAutoencoder(model     = ae_model, 
                                        optimizer = torch.optim.Adam(ae_model.parameters()))

    # import test data
    # test if file type is hdf5
    # if not, raise error
    if tree_data_fn.endswith('.hdf5'):
        with h5py.File(tree_data_fn, "r") as f:
            test_phy_data = torch.tensor(f['phy_data'][...], dtype = torch.float32)
            test_aux_data = torch.tensor(f['aux_data'][...], dtype = torch.float32)
            if len(test_aux_data.shape) == 1:
                test_aux_data = test_aux_data.reshape((test_aux_data.shape[0], 1))
            
            if 'labels' in f.keys() and 'label_names' in f.keys():
                # if labels are present, save them to a csv file
                # check if labels are in the correct format
                labels = f['labels'][...]
                label_names = [lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl for lbl in f['label_names'][...][0]]
                df = pd.DataFrame(labels, columns=label_names)
                df.to_csv(out_file_prefix + ".labels.csv", index = False)
            else:
                print("No labels found in the hdf5 file. Continuing without making labels file.")
                # raise ValueError("No labels found in the hdf5 file.")

    elif tree_data_fn.endswith('.csv'):
        test_phy_data = pd.read_csv(tree_data_fn, header = None, index_col = None).to_numpy(dtype=np.float32)
        test_aux_data = pd.read_csv(aux_data_fn, header = None, index_col = None).to_numpy(dtype=np.float32)
    else:
        raise ValueError("Input file must be in hdf5 or csv format.")
    
    if num_channels is not None:
        # print(test_phy_data[0,0:28])   
        nc = test_phy_data.shape[1] // num_tips
        if test_phy_data.shape[1] % num_tips != 0:
            raise ValueError("Number of tips does not match the number of tips in the trained model.")  
        # reshape, and then reduce the tensor to the number of channels desired (might be fewer than the original)
        test_phy_data = test_phy_data.reshape(-1, num_tips, nc)[:, :, :num_channels].flatten(start_dim=1)
        # print(test_phy_data[0,0:8])

    # make predictions with trained model
    phydat = phy_normalizer.transform(test_phy_data)
    auxdat = aux_normalizer.transform(test_aux_data)
    phydat = phydat.reshape((phydat.shape[0], ae_model.num_structured_input_channel, 
                            int(phydat.shape[1]/ae_model.num_structured_input_channel)), order = "F")
    phydat = torch.Tensor(phydat)
    auxdat = torch.Tensor(auxdat)

    # make encoded tree file
    latent_dat = tree_autoencoder.tree_encode(phydat, auxdat)
    latent_dat_df = pd.DataFrame(latent_dat.detach().to('cpu').numpy(), columns = None, index = None)
    latent_dat_df.to_csv(out_file_prefix + ".ae_encoded.csv", header = None, index = None)

    print("Wrote to: " + out_file_prefix + ".ae_encoded.csv")

if __name__ == "__main__":
    main()