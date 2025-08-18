#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import sys
import time 
import argparse
import torch    
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import scipy
import phyloencode as ph
from phyloencode import utils
from phyloencode.PhyloAutoencoder import PhyloAutoencoder

'''
    This is meant to work with the files from PhyloAutoecnoder
'''
# TODO: am I using the estimated num tips in the recon aux data at all?

def file_exists(fname):
    "Return error if no file"
    if not os.path.isfile(fname):
        raise argparse.ArgumentTypeError(f"File '{fname}' does not exist.")
    return fname

def main():
    cmd_args = argparse.ArgumentParser(description='Measure reconstruction error for test data')
    # data can be either a single hdf5 file containing a 'phy_data' and 'aux_data' element
    # or two csv files containing cblv and aux encoded trees
    cmd_args.add_argument('-d', '--data', type=str, required=False, default='test', help='dataset to use. hdf5 file')
    cmd_args.add_argument('-cblv', '--cblv', type=int, required=False, default=0, help='csv file containging cblv encoded trees')
    cmd_args.add_argument('-aux', '--aux', type=int, required=False, default=0, help='csv file containging aux encoded trees')
    cmd_args.add_argument('-max_tips', '--max-tips', type=int, required=True, help='max number of tips in the data. Default is 500.')
    # trained model and normalizers prefix
    cmd_args.add_argument('-m', '--model-prefix', type = str, required=True, help='prefix of the model and normalizers.')
    cmd_args.add_argument('-o', '--out-prefix', type=str, required=False, default=None, help='Error output prefix for results files')


    args = cmd_args.parse_args()

    # output file
    if args.out_prefix is None:
        output = args.model_prefix
    else:
        output = args.out_prefix

    # import the model and normalizers
    model_fn            = args.model_prefix + ".ae_trained.pt"
    phy_normalizer_fn   = args.model_prefix + ".phy_normalizer.pkl"
    aux_normalizer_fn   = args.model_prefix + ".aux_normalizer.pkl"
    model               = torch.load(file_exists(model_fn), weights_only=False)
    phy_normilzer       = joblib.load(file_exists(phy_normalizer_fn))
    aux_normilzer       = joblib.load(file_exists(aux_normalizer_fn))

    # Model may only use a subset of the channels (e.g. 2 channels for phylogenetic data and ignore character data)
    num_channels    = model.num_structured_input_channel
    num_chars       = model.num_chars
    max_tips        = int(args.max_tips)

    # check if data is provided
    # if data is not provided, check if cblv and aux are provided
    if args.data is None:
        if args.cblv is not None and args.aux is not None:
            file_exists(args.cblv)
            file_exists(args.aux)
            # cblv_data   = pd.read_csv(args.cblv, header=None).values
            aux_data    = pd.read_csv(args.aux, header=None).values
            # labels      = np.array([0] * len(cblv_data))
            # label_names = np.array([0] * len(cblv_data))
            ntips_idx   = np.where(aux_data.keys() == 'num_taxa')[0]
            ntips       = aux_data.iloc['num_taxa'].to_numpy()
        else:
            print('Please provide at least one of the following: -d, or both -cblv and -aux')
            sys.exit(1)
    else: # if all data are in a single hdf5 file
        # give warning if both -d and -cblv/-aux are provided
        if args.cblv is not None or args.aux is not None:
            print('Warning: -d will override -cblv and -aux. Proceeding with -d only.')

        file_exists(args.data)
        with h5py.File(args.data, 'r') as f:
            phy_data    = f['phy_data'][:]
            aux_data    = f['aux_data'][:]
            # aux_data_names = f['aux_data_names'][:][0]
            # labels      = f['labels'][:]
            # label_names = f['label_names'][:][0]
            # get the num tips
            ntips_idx = np.where(f['aux_data_names'][...][0] == b'num_taxa')[0][0]
            ntips = aux_data[:, ntips_idx]

    # subset phy_data to include only channels specified in the model
    phy_data = phy_data.reshape((phy_data.shape[0], int(phy_data.shape[1]/max_tips), max_tips), order='F')
    phy_data = phy_data[:, :num_channels,:]

    # flatten before normalizing
    phy_data = phy_data.reshape((phy_data.shape[0], -1), order='F')
    norm_phy_data = phy_normilzer.transform(phy_data)
    norm_aux_data = aux_normilzer.transform(aux_data)

    # reshape subsetted phy data to (num_samples, num_channels, num_tips)
    norm_phy_data = norm_phy_data.reshape((norm_phy_data.shape[0], num_channels, max_tips), order='F')

    # convert to torch tensors
    norm_phy_data = torch.tensor(norm_phy_data, dtype=torch.float32)
    norm_aux_data = torch.tensor(norm_aux_data, dtype=torch.float32)
    

    # normalize test data and make predictions
    # normalize -> predict -> denormalize
    tree_autoencoder = PhyloAutoencoder(model     = model, 
                                        optimizer = torch.optim.Adam(model.parameters()))

    # this outputs a tensor of shape (num_smaple, num_channels, num_tips)
    pred_phy_data, pred_aux_data = tree_autoencoder.predict(norm_phy_data, norm_aux_data)
    
    # flatten    
    pred_phy_data = pred_phy_data.reshape((pred_phy_data.shape[0], -1), order='F')
    pred_aux_data = pred_aux_data
    pred_phy_data = phy_normilzer.inverse_transform(pred_phy_data)
    pred_aux_data = aux_normilzer.inverse_transform(pred_aux_data)

    mask = np.zeros(phy_data.shape, dtype=bool)
    # phy_flat_width = max_tips * num_channels
    num_unmask = ntips * num_channels

    for t in range(pred_phy_data.shape[0]):
        mask[t,0:int(num_unmask[t])] = True

    # calculate errors    
    phy_abs_diff    = np.abs(phy_data - pred_phy_data) * mask
    phy_rmse        = np.sqrt(np.sum(phy_abs_diff ** 2, axis=1) / num_unmask)
    phy_mae         = np.sum(phy_abs_diff, axis=1) / num_unmask
    phy_mse         = np.sum(phy_abs_diff ** 2, axis=1) / num_unmask

    # concatenate phy errors (one mean error per tree)
    phy_error = np.concatenate((phy_rmse.reshape(-1, 1), phy_mae.reshape(-1, 1), phy_mse.reshape(-1, 1)), axis=1)

    # save phy and aux errors to dataframes
    phy_df = pd.DataFrame(phy_data, columns=None)
    phy_error_df = pd.DataFrame(phy_error, columns=['phy_rmse', 'phy_mae', 'phy_mse'])

    # save phy error results to single file
    phy_df.to_csv(output + '_phy_data.cblv.csv', index=True, header = None)
    phy_error_df.to_csv(output + '_phy_error.csv', index_label = "tree_number", index=True)

    # create scatter plots true v pred for cblv
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output + "_scatter_plots.pdf") as f:
        # 9 plots per page
        ndat = min(100, pred_phy_data.shape[1])
        for i in range(pred_phy_data.shape[1] // 9):
            fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
            fig.tight_layout(pad=2., h_pad=2., w_pad = 2.)
            for j in range(9):
                d = 9 * i + j
                row = j // 3
                col = j % 3
                min_val = min([np.min(phy_data[0:ndat,d]), np.min(pred_phy_data[0:ndat,d])])
                max_val = max([np.max(phy_data[0:ndat,d]), np.max(pred_phy_data[0:ndat,d])])
                ax[row][col].scatter(phy_data[0:ndat,d], pred_phy_data[0:ndat,d], s = 5)
                ax[row][col].set_xlabel("True", fontsize = 8)
                ax[row][col].set_ylabel("Pred", fontsize = 8)
                ax[row][col].label_outer()
                ax[row][col].set_title("dim " + str(d), fontsize = 8)
                ax[row][col].plot([min_val, max_val], [min_val, max_val],
                                   color = "r", linewidth=1)
            f.savefig(fig)
            plt.close()

    # print average errors
    print('avg. phy_rmse: {}'.format(phy_rmse.mean()))
    print('avg. phy_mae: {}'.format(phy_mae.mean()))    


if __name__ == '__main__':
    main()  