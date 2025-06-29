#!/usr/bin/env python3
settings = {    
    "out_prefix"        : "m4_10",    
    "num_subset"        : 120000,
    "nworkers"          : 0,
    "num_epochs"        : 1500,
    "batch_size"        : 1024,
    "num_channels"      : 2,
    "num_chars"         : 0,
    "max_tips"          : 1000,
    "mmd_lambda"        : 5.,
    "vz_lambda"         : 1.,
    "phy_loss_weight"   : 0.75,
    "char_weight"       : 0.5,
    "latent_model_type" : "GAUSS", # ["GAUSS", "CNN", "DENSE"]
    "stride"            : [2,6,8],
    "kernel"            : [3,7,9],
    "out_channels"      : [16,32,128],
    "latent_output_dim" : 100,
    "aux_inner_dim"     : 10,
    "rand_seed"        : 0,
    "char_type"        : "categorical",  #["categorical", "continuous"]
}
