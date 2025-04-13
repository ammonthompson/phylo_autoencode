#!/usr/bin/env python3

import torch
# import torch.utils.data as td
import numpy as np
import pandas as pd
# import sklearn as sk
import phyloencode as ph
# import sys
import h5py
import argparse

import phyloencode as ph
from phyloencode import utils
from phyloencode.PhyloAutoencoder import PhyloAutoencoder
import cProfile
import pstats
def main():

    cProfile.run('''
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--trn_data",         required = True,  help = "Training data in hdf5 format.")
parser.add_argument("-o", "--out_prefix",       required = True,  help = "Output prefix.")
parser.add_argument("-cfg", "--config",          required = False, help = "Configuration file. Settings dictionary. Default None.")
parser.add_argument("-s", "--seed",             required = False, help = "Random seed. Default random.")
parser.add_argument("-nw", "--num_workers",      required = False, help = "Number of workers. Default 0")
parser.add_argument("-ne", "--num_epochs",       required = False, help = "Number of training epochs. Default 100")
parser.add_argument("-b", "--batch_size",       required = False, help = "Batch size. Default 128")
parser.add_argument("-mmd", "--mmd_lambda",       required = False, help = "MMD lambda (>= 0). Default 1.0")
parser.add_argument("-vz", "--vz_lambda",        required = False, help = "VZ lambda (>= 0). Default 1.0")
parser.add_argument("-mt", "--max_tips",         required = False, help = "maximum number of tips.   Default 1000")
parser.add_argument("-w", "--phy_loss_weight",  required = False, help = "Phylogenetic loss weight (in [0,1]) . Default 0.9")
parser.add_argument("-ns", "--num_subset",       required = False, help = "subset of data used for training/testing. Default 10000")
parser.add_argument("-nc", "--num_channels",     required = False, help = "number of data channels. Default 9")
parser.add_argument("-l", "--latent_model_type", required = False, help = "latent model type (GAUSS, DENSE, or CNN). Default GAUSS")
parser.add_argument("-k", "--kernel",           required = False, help = "kernel size. Default 3,5,5")
parser.add_argument("-r", "--stride",           required = False, help = "stride size. Default 2,4,4")
parser.add_argument("-oc", "--out_channels",    required = False, help = "output channels. Default 32,32,128")

args = parser.parse_args()
data_fn = args.trn_data
out_prefix = args.out_prefix

# not used. dataset too small
# num_cpus = multiprocessing.cpu_count()
# num_workers = 0 if (num_cpus - 4) < 0 else num_cpus - 4

# Default settings
num_subset  = 10000
nworkers    = 0
rand_seed   = np.random.randint(0,10000)
num_epochs  = 100
batch_size  = 128
nchannels   = 9
max_tips    = 1000
mmd_lambda  = 1. # xxx5
vz_lambda   = 1. # xxx5
phy_loss_weight = 0.9
latent_model_type = "GAUSS"
stride          = [2,2,4]
kernel          = [3,3,5]
out_channels    = [32,64,128]


# user optional settings
if args.num_subset is not None:
    num_subset = int(args.num_subset)
if args.num_epochs is not None:
    num_epochs = int(args.num_epochs)
if args.batch_size is not None:
    batch_size = int(args.batch_size)
if args.num_channels is not None:
    nchannels = int(args.num_channels)
if args.max_tips is not None:
    max_tips = int(args.max_tips)
if args.mmd_lambda is not None:
    mmd_lambda = float(args.mmd_lambda)
if args.vz_lambda is not None:
    vz_lambda = float(args.vz_lambda)
if args.phy_loss_weight is not None:
    phy_loss_weight = float(args.phy_loss_weight)
if args.latent_model_type is not None:
    latent_model_type = args.latent_model_type
if args.kernel is not None:
    kernel = [int(i) for i in args.kernel.split(",")]
if args.stride is not None:
    stride = [int(i) for i in args.stride.split(",")]
if args.out_channels is not None:
    out_channels = [int(i) for i in args.out_channels.split(",")]

# if config file provided, read in settings and override user settings
if args.config is not None:
    config_fn = args.config
    config = ph.utils.read_config(config_fn)
    config = pd.read_json(config_fn, orient = "index").to_dict(orient = "records")[0]
    if "num_subset" in config:
        num_subset = config["num_subset"]
    if "num_epochs" in config:
        num_epochs = config["num_epochs"]
    if "batch_size" in config:
        batch_size = config["batch_size"]               
    if "num_channels" in config:
        nchannels = config["num_channels"]
    if "max_tips" in config:
        max_tips = config["max_tips"]
    if "mmd_lambda" in config:
        mmd_lambda = config["mmd_lambda"]
    if "vz_lambda" in config:
        vz_lambda = config["vz_lambda"]
    if "phy_loss_weight" in config:
        phy_loss_weight = config["phy_loss_weight"]
    if "latent_model_type" in config:
        latent_model_type = config["latent_model_type"]
    if "kernel" in config:
        kernel = config["kernel"]
    if "stride" in config:      
        stride = config["stride"]
    if "out_channels" in config:
        out_channels = config["out_channels"]
    if "num_workers" in config:
        nworkers = config["num_workers"]
    if "seed" in config:
        rand_seed = config["seed"]

# get formated tree data
with h5py.File(data_fn, "r") as f:
    # idx = np.where(f['aux_data_names'][...][0] == b'num_taxa')[0][0]

    # phy_data = torch.tensor(f['phy_data'][0:num_subset,...], dtype = torch.float32)
    aux_data = torch.tensor(f['aux_data'][0:num_subset,...], dtype = torch.float32)
    phy_data = np.array(f['phy_data'][0:num_subset,...], dtype = np.float32)
    phy_data = phy_data.reshape((phy_data.shape[0], phy_data.shape[1]//max_tips, max_tips), order = "F")
    phy_data = phy_data[:,0:nchannels,:]
    phy_data = phy_data.reshape((phy_data.shape[0],-1), order = "F")
    phy_data = torch.tensor(phy_data, dtype = torch.float32)
    # aux_data = torch.tensor(f['aux_data'][0:num_subset,idx], dtype = torch.float32).view(-1,1)

    # test_phy_data = torch.tensor(f['phy_data'][num_subset:(num_subset + 500),...], dtype = torch.float32)
    test_aux_data = torch.tensor(f['aux_data'][num_subset:(num_subset + 500),...], dtype = torch.float32)
    test_phy_data = np.array(f['phy_data'][num_subset:(num_subset + 500),...], dtype = np.float32)
    test_phy_data = test_phy_data.reshape((test_phy_data.shape[0], test_phy_data.shape[1]//max_tips, max_tips), order = "F")
    test_phy_data = test_phy_data[:,0:nchannels,:]
    test_phy_data = test_phy_data.reshape((test_phy_data.shape[0],-1), order = "F")
    test_phy_data = torch.tensor(test_phy_data, dtype = torch.float32)
    # test_aux_data = torch.tensor(f['aux_data'][num_subset:(num_subset + 500),idx], dtype = torch.float32).view(-1,1)


# checking how much aux_data is helping encode tree patterns
# rand_idx = torch.randperm(aux_data.shape[0])
# aux_data = aux_data[rand_idx]

# create Data container
ae_data = ph.DataProcessors.AEData(data = (phy_data, aux_data), 
                                    prop_train = 0.85,  
                                    nchannels  = nchannels)
ae_data.save_normalizers(out_prefix)

# create data loaders
trn_loader, val_loader = ae_data.get_dataloaders(batch_size  = batch_size, 
                                                    shuffle     = True, 
                                                    num_workers = nworkers)

# create model
ae_model  = ph.PhyloAEModel.AECNN(num_structured_input_channel  = ae_data.nchannels, 
                                    structured_input_width        = ae_data.phy_width,
                                    unstructured_input_width      = ae_data.aux_width,
                                    stride                        = stride,
                                    kernel                        = kernel,
                                    out_channels                  = out_channels,
                                    latent_layer_type             = latent_model_type,
                                    )

# create Trainer
tree_autoencoder = PhyloAutoencoder(model           = ae_model, 
                                    optimizer       = torch.optim.Adam(ae_model.parameters()), 
                                    loss_func       = ph.utils.recon_loss, 
                                    batch_size      = batch_size,
                                    phy_loss_weight = phy_loss_weight,
                                    mmd_lambda      = mmd_lambda,
                                    vz_lambda       = vz_lambda
                                    )

# Load data loaders and Train model and plot
tree_autoencoder.set_data_loaders(train_loader=trn_loader, val_loader=val_loader)
tree_autoencoder.train(num_epochs = num_epochs, seed = rand_seed)''', "profile_output") 
# end cProfile.run

    # Save the profiling results to a file
    with open("profile_results.txt", "w") as f:
        stats = pstats.Stats("profile_output", stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats()

if __name__ == "__main__":
    main()
