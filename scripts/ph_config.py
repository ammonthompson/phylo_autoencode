#!/usr/bin/env python3
settings = {    
    "num_subset"        : 120000,
    "nworkers"          : 0,
    "num_epochs"        : 1000,
    "batch_size"        : 128,
    "nchannels"         : 9,
    "max_tips"          : 1000,
    "mmd_lambda"        : 5.,
    "vz_lambda"         : 2.,
    "phy_loss_weight"   : 0.95,
    "char_weight"       : 0.5,
    "latent_model_type" : "GAUSS",
    "stride"            : [2,4,4],
    "kernel"            : [3,5,5],
    "out_channels"      : [32,64,128],
}