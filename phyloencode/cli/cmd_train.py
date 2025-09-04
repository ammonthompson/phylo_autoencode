#!/usr/bin/env python3

import torch
from torch.optim import AdamW
# import torch.utils.data as td
import numpy  as np
import pandas as pd
# import sklearn as sk
import phyloencode as ph
# import sys
import h5py
import argparse
from sklearn.model_selection import train_test_split

import phyloencode as ph
# from phyloencode import utils
from phyloencode.PhyloAutoencoder import PhyloAutoencoder
from phyloencode.PhyloAEModel     import AECNN
from phyloencode.DataProcessors   import AEData
from phyloencode.utils            import get_num_tips

def main():

    # Training settings: Architecture, num epochs, batch size, etc.
    # override default settings provided as command line arguments
    # Override all settings provided in config file if provided
    settings    = get_default_settings()
    args        = parse_arguments()
    update_settings_from_command_line(settings = settings, args = args)
    if args.config:
        config = ph.utils.read_config(args.config)
        update_settings_from_config(settings = settings, config = config)

    # Optimizer settings
    lr = settings['learning_rate']
    wd = settings['weight_decay']

    # required command line argument
    data_fn     = args.trn_data
    ns          = settings["num_subset"]
    mt          = settings["max_tips"]
    nc          = settings["num_channels"]
    num_test    = 1000


    # get formated tree data
    with h5py.File(data_fn, "r") as f:

        # num tips is important for masking. 
        # TODO: Check a potential discrepency in num_taxa with phyddle --format output
            
        phy_data_np = np.array(f['phy_data'][0:ns,...], dtype = np.float32)
        # extract the specified subset of channels ("num_channels")
        phy_data = get_channels(phy_data_np, nc, mt)

        aux_data = torch.tensor(f['aux_data'][0:ns,...], dtype = torch.float32)
        if len(aux_data.shape) != 2: # i.e. is an array but should be a matrix with 1 column
            aux_data = aux_data.reshape((aux_data.shape[0], 1))

        # get num tips (tree width/num_taxa)
        try:
            aux_data_names = f['aux_data_names'][...][0] 
            cidx = np.where(aux_data_names == b'num_taxa')[0][0]
            num_tips = torch.tensor(f['aux_data'][0:ns, cidx], dtype = torch.float32).view(-1,1)
            # (maybe not necessary) remove the num taxa column (will be replaced with a num tips column at column 0)
            # aux_data = np.delete(aux_data, cidx, axis = 1)
        except(KeyError, IndexError) as e:
            # get num tips from the tree itself
            print(f"""No \"num_taxa\" in aux data. Computing num tips from phy_data instead. Exception: {e}""")
            num_tips = get_num_tips(phy_data_np, mt) 

        # place num_tips in first column of aux_data
        aux_data = torch.hstack((num_tips, aux_data))

        ### TESTING ##
        # num_tips = get_num_tips(phy_data_np, mt)

        # split off test data
        (phy_data, test_phy_data, 
         aux_data, test_aux_data, 
         num_tips, test_num_tips) = train_test_split(phy_data, aux_data, num_tips, 
                                                     test_size=num_test, shuffle=True)


    ###################################
    # Set up network training objects #
    ###################################
    # create Data container
    ae_data     = AEData( 
                        phy_data         = phy_data,
                        aux_data         = aux_data,
                        prop_train       = settings['proportion_train'],  
                        num_channels     = settings["num_channels"], 
                        num_chars        = settings["num_chars"],
                        num_tips         = num_tips
                        )
    
    trn_loader, val_loader = ae_data.get_dataloaders(settings["batch_size"], shuffle = True, 
                                                    num_workers = settings["num_workers"])
        
    # create model
    ae_model    = AECNN(
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
    
    # optimizer
    # opt = AdamW(ae_model.parameters(), lr=lr, weight_decay=wd)
    opt         = AdamW(split_params_by_wd(ae_model, wd), lr=lr)
    lr_schedlr  = torch.optim.lr_scheduler.OneCycleLR(
                        opt,
                        max_lr=lr,
                        epochs=settings["num_epochs"], 
                        steps_per_epoch=len(trn_loader),
                        pct_start=0.1,               # 10% warmup
                        anneal_strategy='cos',
                        cycle_momentum=False
                        )


    # create Trainer
    tree_ae     = PhyloAutoencoder(
                        model           = ae_model, 
                        optimizer       = opt, 
                        lr_scheduler    = lr_schedlr,
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

    tree_ae.set_data_loaders(train_loader=trn_loader, val_loader=val_loader)
    tree_ae.train(num_epochs = settings["num_epochs"], seed = settings["seed"])
    if tree_ae.track_grad:
        plot_gradient_norms(tree_ae.mean_layer_grad_norm, 
                            settings["out_prefix"] + ".layer_grad_norms.pdf")
            

    # save model and normalizers
    tree_ae.save_model(settings["out_prefix"] + ".ae_trained.pt")
    ae_data.save_normalizers(settings["out_prefix"])

    # make encoded tree file for 5,000 random trees from training data
    rand_idx        = np.random.randint(0, ae_data.prop_train * phy_data.shape[0], 
                                        size = min(5000, phy_data.shape[0]))
    rand_train_phy  = torch.Tensor(ae_data.norm_train_phy_data[rand_idx,...])
    rand_train_aux  = torch.Tensor(ae_data.norm_train_aux_data[rand_idx,...])
    latent_dat      = tree_ae.tree_encode(rand_train_phy, rand_train_aux)
    latent_dat_df   = pd.DataFrame(latent_dat.detach().to('cpu').numpy(), 
                                 columns = None, index = None)
    latent_dat_df.to_csv(settings["out_prefix"] + ".traindat_latent.csv", 
                         header = False, index = False)


    #####################################################
    # Test Data Prediction 
    # make predictions with trained model on test data
    #####################################################

    # save true values of test data in cblv format
    phy_true_df = pd.DataFrame(test_phy_data.numpy())
    aux_true_df = pd.DataFrame(test_aux_data.numpy())
    phy_true_df.to_csv(settings["out_prefix"] + ".phy_true.cblv", 
                       header = False, index = False)
    aux_true_df.to_csv(settings["out_prefix"] + ".aux_true.csv", 
                       header = False, index = False)

    # normalize test data
    phy_normalizer, aux_normalizer = ae_data.get_normalizers()
    phydat = phy_normalizer.transform(test_phy_data)
    auxdat = aux_normalizer.transform(test_aux_data)
    phydat = phydat.reshape((phydat.shape[0], ae_data.num_channels, ae_data.phy_width), 
                            order = "F")
    phydat = torch.Tensor(phydat)
    auxdat = torch.Tensor(auxdat)

    # make predictions for test data (latent and reconstructed)
    test_latent_dat = tree_ae.tree_encode(phydat, auxdat)
    latent_testdat_df = pd.DataFrame(test_latent_dat.detach().to('cpu').numpy(), 
                                     columns = None, index = None)
    latent_testdat_df.to_csv(settings["out_prefix"] + ".testdat_latent.csv", 
                             header = False, index = False)

    phy_pred, auxpred = tree_ae.predict(phydat, auxdat)

    # transform and flatten predicted data
    phy_pred = phy_normalizer.inverse_transform(phy_pred.reshape((phy_pred.shape[0], -1), 
                                                                 order = "F"))
    auxpred  = aux_normalizer.inverse_transform(auxpred)

    # save predictions to file
    phy_pred_df = pd.DataFrame(phy_pred)
    aux_pred_df = pd.DataFrame(auxpred)
    phy_pred_df.to_csv(settings["out_prefix"] + ".phy_pred.cblv", 
                       header = False, index = False)
    aux_pred_df.to_csv(settings["out_prefix"] + ".aux_pred.csv", 
                       header  = False, index = False)

    # plot loss curves
    tree_ae.plot_losses(settings["out_prefix"])



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--trn_data",         required = True,  help = "Training data in hdf5 format.")
    parser.add_argument("-o", "--out_prefix",       required = False,  help = "Output prefix.")
    parser.add_argument("-cfg", "--config",         required = False, help = "Configuration file. Settings dictionary. Default None.")
    parser.add_argument("-s", "--seed",             required = False, type = int, help = "Random seed. Default random.")
    parser.add_argument("-nw", "--num_workers",     required = False, type = int, help = "Number of workers. Default 0")
    parser.add_argument("-ne", "--num_epochs",      required = False, type = int, help = "Number of training epochs. Default 100")
    parser.add_argument("-b", "--batch_size",       required = False, type = int, help = "Batch size. Default 128")
    parser.add_argument("-mmd", "--mmd_lambda",     required = False, type = float, help = "MMD lambda (>= 0). Default 1.0")
    parser.add_argument("-vz", "--vz_lambda",       required = False, type = float, help = "VZ lambda (>= 0). Default 1.0")
    parser.add_argument("-mt", "--max_tips",        required = False, type = float, help = "maximum number of tips.   Default 1000")
    parser.add_argument("-w", "--phy_loss_weight",  required = False, type = float, help = "Phylogenetic loss weight (in [0,1]) . Default 0.9")
    parser.add_argument("-ns", "--num_subset",      required = False, type = int, help = "subset of data used for training/testing. Default 10000")
    parser.add_argument("-nchans", "--num_channels",  required = False, type = int, help = "number of data channels. Default 9")
    parser.add_argument("-num_chars", "--num_chars",  required = False, type = int, help = "number of characters. Default 5")
    parser.add_argument("-ld", "--latent_output_dim", required = False, type = int, help = "latent output dimension. Default None (determined by structured encoder output shape)")
    parser.add_argument("-l", "--latent_model_type",  required = False, help = "latent model type (GAUSS, DENSE, or CNN). Default GAUSS")
    parser.add_argument("-cw", "--char_weight",     required = False, type = float, help = "how much weight to give to char loss. Default 0.0")
    parser.add_argument("-k", "--kernel",           required = False, type = int, help = "kernel size. Default 3,5,5")
    parser.add_argument("-r", "--stride",           required = False, type = int, help = "stride size. Default 2,4,4")
    parser.add_argument("-oc", "--out_channels",    required = False, type = int, help = "output channels. Default 32,32,128")
    parser.add_argument("-ct", "--char_type",       required = False, help = "character type (categorical or continuous). Default categorical")
    parser.add_argument("-pt", "--proportion-train", required = False, type = float, help = "Proportion of num-subset used for training vs validation. Default 0.85")
    parser.add_argument("-lr", "--learning-rate",   required = False, type = float, help = "Optimizer learning rate. Default 1e-3")
    parser.add_argument("-wd", "--weight-decay",    required = False, type = float, help = "Optimizer weight decay. Default 1e-3")
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
        "char_type": "categorical",
        "proportion_train" : 0.85,
        "learning_rate" : 1e-3,
        "weight_decay"  : 1e-3
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
        "char_type"     : args.char_type,
        "proportion_train": args.proportion_train,
        "learning_rate" : args.learning_rate,
        "weight_decay"  : args.weight_decay

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

def plot_gradient_norms(layer_grad_norms, out_file, plots_per_page = 4):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    laynorm = [z for z in layer_grad_norms.items()]
    n_plots = len(laynorm)
    n_pages = n_plots // 4 + ((n_plots % 4) > 0)
    with PdfPages(out_file) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(plots_per_page // 2, 2)
            axes = axes.flatten()
            for plot_i in range(plots_per_page):
                idx = page * plots_per_page + plot_i
                if idx >= n_plots:
                    axes.axis('off')
                    continue
                axes[plot_i].plot(laynorm[idx][1])
                axes[plot_i].set_title(laynorm[idx][0], size = 6.)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def update_settings_from_config(settings : dict, config : dict):
    for key in settings:
        if key in config:
            settings[key] = config[key]

def split_params_by_wd(model, wd):
    # weight decay should be zero for bias and normalization variables
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower() or "Norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def get_channels(phydata: np.ndarray, num_chans: int, max_tips: int) -> torch.Tensor:
    # sometimes the number channels in the settings is less than the data.
    # for example, if you want to ignore character data and just analyze trees
    phydata = phydata.reshape((phydata.shape[0], phydata.shape[1] // max_tips, max_tips), order = "F")
    phydata = phydata[:,0:num_chans,:] 
    phydata = phydata.reshape((phydata.shape[0],-1), order = "F")
    phydata = torch.tensor(phydata, dtype = torch.float32)
    return phydata


if __name__ == "__main__":
    main()
