#!/usr/bin/env python3

settings = {
    "out_prefix"        : "EXAMPLE",
    "device"            : "auto",
    "testing"           : True,                                                                                                                                                                 "seed"              : None,
    "num_subset"        : 30000, # how much of the data to use
    "nworkers"          : 0,
    "num_epochs"        : 5000,
    "batch_size"        : 1024,
    "num_channels"      : 7,
    "num_chars"         : 3,
    "max_tips"          : 500,
    "mmd_loss_weight"   : 200.,
    "vz_loss_weight"    : 10.,
    "phy_loss_weight"   : 1.0,
    "char_loss_weight"  : 0.2,
    "aux_loss_weight"   : 0.1,
    "latent_model_type" : "GAUSS", # ["GAUSS", "CNN", "DENSE"]
    "stride"            : [2,2,2,4],
    "kernel"            : [3,3,5,5],
    "out_channels"      : [16,16,32,32],
    "latent_output_dim" : 300,
    "aux_inner_dim"     : 10,
    "char_type"         : "categorical",  #["categorical", "continuous"]
    "weight_decay"      : 1e-4,
    "learning_rate"     : 1e-3
}

