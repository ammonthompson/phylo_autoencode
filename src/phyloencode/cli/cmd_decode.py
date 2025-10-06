#!/usr/bin/env python3
# test a pretrained model
# from phyloencode.PhyloAutoencoder import PhyloAutoencoder
import phyloencode.utils as utils
import torch
# import joblib
# import h5py
import numpy as np
import pandas as pd
# import os
import argparse

# takes in an encoded tree (encoded by the same trained model as the decoder)
# outputs a cblv file

# TODO: Output a nwk string too?

def main ():
    cmd = argparse.ArgumentParser(description="Encode phylogenetic trees and auxiliary data with trained autodencoder.")
    cmd.add_argument("-m", "--model", required=True, help="Path to the trained model.pt file")
    cmd.add_argument("-e", "--encoded-data", required=True, help="Path to the model encoded data file")
    cmd.add_argument("-o", "--out-prefix", required=False, help="Path to out file prefix")

    args = cmd.parse_args()

    ae_model_fn       = utils.file_exists(args.model)
    encoded_data_fn   = utils.file_exists(args.encoded_data)
    
    if args.out_prefix is None:
        out_file_prefix = encoded_data_fn.split('/')[-1].split('.')[0]
    else:
        out_file_prefix = args.out_prefix

    # load trained model and normalizers and create PhyloAutoencoder object
    ae_model = torch.load(ae_model_fn, weights_only=False)

    # import test data
    encoded_data = pd.read_csv(encoded_data_fn, header = None, index_col = None).to_numpy(dtype=np.float32)

    test_phy_data, test_aux_data = ae_model.decode_and_denorm(encoded_data)

    test_phy_data = utils.set_pred_pad_to_zero(test_phy_data,  test_aux_data[:, ae_model.aux_numtips_idx])    
    test_phy_data = test_phy_data.reshape((test_phy_data.shape[0], -1), order = "F")

 
    cblv_df = pd.DataFrame(test_phy_data)
    aux_df  = pd.DataFrame(test_aux_data)
    cblv_df.to_csv(out_file_prefix + ".ae_decoded.cblv.csv", header = None, index = None)
    aux_df.to_csv(out_file_prefix + ".ae_decoded.aux.csv", header = None, index = None)

    print("Wrote to: " + out_file_prefix + ".ae_decoded.cblv.csv and " + out_file_prefix + ".ae_decoded.aux.csv")

if __name__ == "__main__":
    main()