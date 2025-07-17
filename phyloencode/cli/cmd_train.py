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

def main():

    # Training settings: Architecture, num epochs, batch size, etc.
    # override default settings provided as command line arguments
    # Override all settings provided in config file if provided
    settings = get_default_settings()
    args = parse_arguments()
    update_settings_from_command_line(settings = settings, args = args)
    if args.config:
        config = ph.utils.read_config(args.config)
        update_settings_from_config(settings = settings, config = config)

    # required command line argument
    data_fn = args.trn_data

    # get formated tree data
    with h5py.File(data_fn, "r") as f:
        aux_data_names = f['aux_data_names'][...][0] 
        num_tips_idx = np.where(aux_data_names == b'num_taxa')[0][0]
        aux_data = torch.tensor(f['aux_data'][0:settings["num_subset"],...], dtype = torch.float32)
        if len(aux_data.shape) != 2: # i.e. is an array but should be a matrix with 1 column
            aux_data = aux_data.reshape((aux_data.shape[0], 1))
        phy_data = np.array(f['phy_data'][0:settings["num_subset"],...], dtype = np.float32)
        phy_data = phy_data.reshape((phy_data.shape[0], phy_data.shape[1] // settings["max_tips"], settings["max_tips"]), order = "F")
        phy_data = phy_data[:,0:settings["num_channels"],:] 
        phy_data = phy_data.reshape((phy_data.shape[0],-1), order = "F")
        phy_data = torch.tensor(phy_data, dtype = torch.float32)
        # aux_data = torch.tensor(f['aux_data'][0:settings["num_subset"], num_tips_idx], dtype = torch.float32).view(-1,1)
        num_tips = torch.tensor(f['aux_data'][0:settings["num_subset"],num_tips_idx], dtype = torch.float32).view(-1,1)
        
        test_aux_data = torch.tensor(f['aux_data'][settings["num_subset"]:(settings["num_subset"] + 500),...], dtype = torch.float32)
        if len(test_aux_data.shape) != 2: # i.e. is an array but should be a matrix with 1 column
            test_aux_data = test_aux_data.reshape((test_aux_data.shape[0], 1))
        test_phy_data = np.array(f['phy_data'][settings["num_subset"]:(settings["num_subset"] + 500),...], dtype = np.float32)
        test_phy_data = test_phy_data.reshape((test_phy_data.shape[0], test_phy_data.shape[1] // settings["max_tips"], settings["max_tips"]), order = "F")
        test_phy_data = test_phy_data[:,0:settings["num_channels"],:]
        test_phy_data = test_phy_data.reshape((test_phy_data.shape[0],-1), order = "F")
        test_phy_data = torch.tensor(test_phy_data, dtype = torch.float32)
        # test_aux_data = torch.tensor(f['aux_data'][settings["num_subset"]:(settings["num_subset"] + 500), num_tips_idx], dtype = torch.float32).view(-1,1)


    # create Data container
    ae_data = ph.DataProcessors.AEData( 
                                       phy_data         = phy_data,
                                       aux_data         = aux_data,
                                       prop_train       = 0.85,  
                                       num_channels     = settings["num_channels"], 
                                       num_chars        = settings["num_chars"],
                                       num_tips         = num_tips)
        
    # create model
    ae_model  = ph.PhyloAEModel.AECNN(
                                      num_structured_input_channel  = ae_data.num_channels, 
                                      structured_input_width        = ae_data.phy_width,
                                      unstructured_input_width      = ae_data.aux_width,
                                      stride                        = settings["stride"],
                                      kernel                        = settings["kernel"],
                                      out_channels                  = settings["out_channels"],
                                      latent_output_dim             = settings["latent_output_dim"],
                                      latent_layer_type             = settings["latent_model_type"],
                                      num_chars                     = settings["num_chars"],
                                      char_type                     = settings["char_type"],
                                      out_prefix                    = settings["out_prefix"]
                                      )

    # create Trainer
    tree_autoencoder = PhyloAutoencoder(
                                        model           = ae_model, 
                                        optimizer       = torch.optim.Adam, 
                                        batch_size      = settings["batch_size"],
                                        phy_loss_weight = settings["phy_loss_weight"],
                                        char_weight     = settings["char_weight"],
                                        mmd_lambda      = settings["mmd_lambda"],
                                        vz_lambda       = settings["vz_lambda"],
                                        )
    

    #######################################
    # Train model with data from ae_data ##
    #######################################
    # create and add data loaders to trainer and train model
    trn_loader, val_loader = ae_data.get_dataloaders(settings["batch_size"], shuffle = True, num_workers = settings["num_workers"])
    tree_autoencoder.set_data_loaders(train_loader=trn_loader, val_loader=val_loader)
    tree_autoencoder.train(num_epochs = settings["num_epochs"], seed = settings["seed"])

    # save model and normalizers
    tree_autoencoder.save_model(settings["out_prefix"] + ".ae_trained.pt")
    ae_data.save_normalizers(settings["out_prefix"])

    # make encoded tree file for 5,000 random trees from training data
    rand_idx = np.random.randint(0, ae_data.prop_train * settings["num_subset"], size = min(5000, settings["num_subset"]))
    rand_train_phy = torch.Tensor(ae_data.norm_train_phy_data[rand_idx,...])
    rand_train_aux = torch.Tensor(ae_data.norm_train_aux_data[rand_idx,...])
    latent_dat = tree_autoencoder.tree_encode(rand_train_phy, rand_train_aux)
    latent_dat_df = pd.DataFrame(latent_dat.detach().to('cpu').numpy(), columns = None, index = None)
    latent_dat_df.to_csv(settings["out_prefix"] + ".traindat_latent.csv", header = False, index = False)

    tree_autoencoder.plot_losses(settings["out_prefix"])

    #############################
    # Test Data Prediction 
    # make predictions with trained model on test data
    #############################

    # save true values of test data in cblv format
    phy_true_df = pd.DataFrame(test_phy_data.numpy())
    aux_true_df = pd.DataFrame(test_aux_data.numpy())
    phy_true_df.to_csv(settings["out_prefix"] + ".phy_true.cblv", header = False, index = False)
    aux_true_df.to_csv(settings["out_prefix"] + ".aux_true.csv", header = False, index = False)

    # normalize test data
    phy_normalizer, aux_normalizer = ae_data.get_normalizers()
    phydat = phy_normalizer.transform(test_phy_data)
    auxdat = aux_normalizer.transform(test_aux_data)
    phydat = phydat.reshape((phydat.shape[0], ae_data.num_channels, ae_data.phy_width), order = "F")
    phydat = torch.Tensor(phydat)
    auxdat = torch.Tensor(auxdat)

    # make predictions for test data (latent and reconstructed)
    test_latent_dat = tree_autoencoder.tree_encode(phydat, auxdat)
    latent_testdat_df = pd.DataFrame(test_latent_dat.detach().to('cpu').numpy(), columns = None, index = None)
    latent_testdat_df.to_csv(settings["out_prefix"] + ".testdat_latent.csv", header = False, index = False)

    phy_pred, auxpred = tree_autoencoder.predict(phydat, auxdat)

    # transform and flatten predicted data
    phy_pred = phy_normalizer.inverse_transform(phy_pred.reshape((phy_pred.shape[0], -1), order = "F"))
    auxpred  = aux_normalizer.inverse_transform(auxpred)

    # save predictions to file
    phy_pred_df = pd.DataFrame(phy_pred)
    aux_pred_df = pd.DataFrame(auxpred)
    phy_pred_df.to_csv(settings["out_prefix"] + ".phy_pred.cblv", header = False, index = False)
    aux_pred_df.to_csv(settings["out_prefix"] + ".aux_pred.csv", header  = False, index = False)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--trn_data",         required = True,  help = "Training data in hdf5 format.")
    parser.add_argument("-o", "--out_prefix",       required = False,  help = "Output prefix.")
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
    parser.add_argument("-nchans", "--num_channels",     required = False, help = "number of data channels. Default 9")
    parser.add_argument("-num_chars", "--num_chars",     required = False, help = "number of characters. Default 5")
    parser.add_argument("-ld", "--latent_output_dim", required = False, help = "latent output dimension. Default None (determined by structured encoder output shape)")
    parser.add_argument("-l", "--latent_model_type", required = False, help = "latent model type (GAUSS, DENSE, or CNN). Default GAUSS")
    parser.add_argument("-cw", "--char_weight",     required = False, help = "how much weight to give to char loss. Default 0.0")
    parser.add_argument("-k", "--kernel",           required = False, help = "kernel size. Default 3,5,5")
    parser.add_argument("-r", "--stride",           required = False, help = "stride size. Default 2,4,4")
    parser.add_argument("-oc", "--out_channels",    required = False, help = "output channels. Default 32,32,128")
    parser.add_argument("-ct", "--char_type",       required = False, help = "character type (categorical or continuous). Default categorical")

    return parser.parse_args()


def get_default_settings():
    return {
        "out_prefix": "out",
        "num_subset": 10000,
        "num_workers": 0,
        "seed": np.random.randint(0, 10000),
        "num_epochs": 100,
        "batch_size": 128,
        "num_chars": 0,
        "num_channels": 2,
        "max_tips": 1000,
        "mmd_lambda": 1.0,
        "vz_lambda": 1.0,
        "phy_loss_weight": 0.9,
        "char_weight": 0.1,
        "latent_model_type": "GAUSS",
        "latent_output_dim": None,
        "stride": [2, 4, 8],
        "kernel": [3, 5, 9],
        "out_channels": [16, 64, 128],
        "char_type": "categorical"
    }

def update_settings_from_command_line(settings, args):

    arg_map = {
        "out_prefix"    : args.out_prefix,
        "num_subset"    : args.num_subset,
        "num_workers"   : args.num_workers,
        "seed"          : args.seed,
        "num_epochs"    : args.num_epochs,
        "batch_size"    : args.batch_size,
        "num_chars"     : args.num_chars,
        "num_channels"  : args.num_channels,
        "max_tips"      : args.max_tips,
        "mmd_lambda"    : args.mmd_lambda,
        "vz_lambda"     : args.vz_lambda,
        "phy_loss_weight": args.phy_loss_weight,
        "char_weight"   : args.char_weight,
        "latent_model_type": args.latent_model_type,
        "latent_output_dim": args.latent_output_dim,
        "kernel"        : args.kernel,
        "stride"        : args.stride,
        "out_channels"  : args.out_channels,
        "char_type"     : args.char_type
    }

    # override defaults with command line args
    for k, v in arg_map.items():
        if v is not None:
            if k in {"kernel", "stride", "out_channels"} and isinstance(v, str):
                settings[k] = [int(x) for x in v.split(",")]
            elif k in {"latent_output_dim", "num_channels", "num_chars", "num_subset", 
                       "num_epochs", "batch_size", "max_tips", "num_workers", "seed"}:
                settings[k] = int(v)
            elif k in {"mmd_lambda", "vz_lambda", "phy_loss_weight", "char_weight"}:
                settings[k] = float(v)
            else:
                settings[k] = v



def update_settings_from_config(settings, config):
    for key in settings:
        if key in config:
            settings[key] = config[key]


if __name__ == "__main__":
    main()
