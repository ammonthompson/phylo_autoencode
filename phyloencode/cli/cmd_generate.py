#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import torch
import phyloencode
import phyloencode.utils as utils
import random
import h5py
import sys


# TODO: generate labels. The output needs to have tree, chars, aux, and labels files.
# takes in a model and a number of samples
# should be adaptable for a phyddle pipeline
def main():
    cmd = argparse.ArgumentParser(prog = "", usage = "")
    cmd.add_argument("-m", "--model", type=str, required=True, 
                    help="Path to trained autoencoder (.pt file) used for generating new samples.")
    cmd.add_argument("-n", "--num-samples", required=True, type=int, 
                    help="Number of trees to generate.")
    cmd.add_argument("-o", "--out-prefix", type = str, required=True, 
                    help="prefix Out file paths.")
    cmd.add_argument("-s", "--seed", required=False, type=int, default=None, 
                    help="Set the seed. Default is random")


    args = cmd.parse_args()
    model = torch.load(utils.file_exists(args.model), weights_only=False)
    N = args.num_samples
    out_prefix = args.out_prefix

    # set seed
    if args.seed is None:
        seed = int(np.random.randint(0, 2**32 - 1))
    else:
        seed = args.seed
    random.seed(seed)
    rng = np.random.default_rng(seed)
    # g   = torch.Generator(model.device)
    # g.manual_seed(seed)

    # extract data information
    nc          = model.num_structured_input_channel
    nchar       = model.num_chars
    nt          = model.structured_input_width
    ntips_cidx  = model.aux_numtips_idx
    latent_dim  = model.latent_layer_dim
    chartype    = model.char_type

    # draw N samples from N(0,I)
    z = rng.standard_normal(size = (N, latent_dim))

    # decode and inverse transform
    gen_phy, gen_aux = model.decode_and_denorm(z)

    gen_numtips = gen_aux[:, ntips_cidx]


    # set padding to zero
    gen_phy = utils.set_pred_pad_to_zero(gen_phy,  gen_numtips)

    # if char in model, use argmax to simulate characters
    if nchar > 0:
        if chartype == "categorical":
            max_idx = np.argmax(gen_phy[:, (nc - nchar):nc, :], axis = 1) + nc - nchar
            gen_phy[:, (nc - nchar):nc, :] = 0
            # for i in range(gen_phy.shape[0]):
            #     for k in range(gen_phy.shape[2]):
            #         gen_phy[i, max_idx[i,k], k] = 1
            i = np.arange(N)[:,None]
            k = np.arange(nt)[None,:]
            gen_phy[i,max_idx,k] = 1

            if gen_phy.shape[1] < (nchar + 2):
                print("cblvs must be at least as long as nchar + 2.")
                sys.exit(1)
                
    gen_char = gen_phy[:, (nc - nchar):nc, :]

    # flatten and place in pd.dataframe
    df_flat_gen_phy = pd.DataFrame(gen_phy.reshape((gen_phy.shape[0], -1), order = "F"))
    df_gen_aux = pd.DataFrame(gen_aux)

    # # df.write_csv()
    df_flat_gen_phy.to_csv(out_prefix + ".cblv.csv", header=False, index=False)
    df_gen_aux.to_csv(out_prefix + ".aux.csv", header = False, index = False)

    # convert to newick or nexus
    #

    # check if num_tips is same length as cblvs
    if len(gen_numtips) != gen_phy.shape[0]:
        print("num_tips and cblvs must be same length.")
        sys.exit(1)  

    gen_phy_nwk = utils.convert_to_newick(gen_phy[:,0:2,:], gen_numtips, gen_char)

    # write tree file(s)
    # import csv
    # df_gen_phy_nwk = pd.DataFrame([x.strip() for x in gen_phy_nwk])
    # df_gen_phy_nwk.to_csv(out_prefix + ".tre", header = False, index = False,
    #                       quoting=csv.QUOTE_NONE, sep = "@")
    with open(out_prefix + ".tre", "w") as f:
        [f.write(x) for x in gen_phy_nwk] # this relies on the "\n" output by dp.Tree.as_string()



    # TODO output as hdf5 format
    # to match a phyddle generated data file, should have an 
    # aux_data, shape(N, x)
    # aux_data_name, shape(1, x)
    # idx, shape (N,)
    # label_names, shape (1, z)
    # labels, shape (N, z)
    # phy_data, shape (N, nc * nt)
    with h5py.File(out_prefix + ".hdf5", "w") as f:
        pass


if __name__ == "__main__":
    main()
