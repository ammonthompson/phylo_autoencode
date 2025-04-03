#!/usr/bin/env python3
settings = {    
    "num_subset"        : 10000,
    "nworkers"          : 0,
    "num_epochs"        : 100,
    "batch_size"        : 128,
    "nchannels"         : 9,
    "max_tips"          : 1000,
    "mmd_lambda"        : 1.,
    "vz_lambda"         : 1.,
    "phy_loss_weight"   : 0.9,
    "latent_model_type" : "GAUSS",
    "stride"            : [2,4,4],
    "kernel"            : [3,5,5],
    "out_channels"      : [32,32,128],
}