#!/usr/bin/env python3

import random
import torch
from torch.optim import AdamW
import numpy  as np
import pandas as pd
import phyloencode as ph
import h5py
import argparse
from sklearn.model_selection import train_test_split

from phyloencode.PhyloAutoencoder   import PhyloAutoencoder
from phyloencode.PhyloAEModel       import AECNN
from phyloencode.DataProcessors     import AEData
from phyloencode.PhyLoss            import PhyLoss
import phyloencode.utils as utils

# TODO: take in a list of label key words to be included in aux dat eg. ['log_R0', 'log_sample',...]
#       these should be stacked onto the aux variable, or should I create a labels subnetwork in the model?
#       -> then separate before creating predicted aux and label files.
# TODO: includ min tips
def main():

    # Training settings: Architecture, num epochs, batch size, etc.
    # override default settings provided as command line arguments
    # Override all settings provided in config file if provided
    settings    = get_default_settings()
    args        = parse_arguments()
    update_settings_from_command_line(settings = settings, args = args)
    if args.config:
        config = utils.read_config(args.config)
        update_settings_from_config(settings = settings, config = config)


    # required command line argument
    data_fn     = args.trn_data
    ns          = settings["num_subset"]
    mt          = settings["max_tips"]
    nc          = settings["num_channels"]
    num_test    = 5000

    # Set seeds    
    if settings['seed'] is None:
        settings['seed'] = np.random.randint(0, 2**32 - 1)   
    seed = settings['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("seed: ", seed)
    if settings['testing']:
        # torch does some random stuff for more efficient training. 
        # causes slight differences despite same seed. Use below for exact.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # get formated tree data
    with h5py.File(data_fn, "r") as f:

        # TODO: Check a potential discrepency in num_taxa with phyddle --format output
        # TODO: maybe move some of this to backend: the Data object from DataProcessors 
        # should maybe handle all this
        # TODO: Eventually dataloaders need to load from disk, dont keep data in memory: load -> normalizer -> save
        if ns is None:
            phy_data_np = np.array(f['phy_data'][...], dtype = np.float32)
            aux_data = torch.tensor(f['aux_data'][...], dtype = torch.float32)
        else:            
            phy_data_np = np.array(f['phy_data'][0:ns,...], dtype = np.float32)
            aux_data = torch.tensor(f['aux_data'][0:ns,...], dtype = torch.float32)

        # extract the specified subset of channels ("num_channels")
        phy_data = get_channels(phy_data_np, nc, mt)

        # aux data must contain at least one column, "num_taxa"
        if len(aux_data.shape) != 2: # i.e. is an array but should be a matrix with 1 column
            aux_data = aux_data.reshape((aux_data.shape[0], 1))

        aux_data_names = f['aux_data_names'][...][0]

        # split off test data
        (phy_data, test_phy_data, 
         aux_data, test_aux_data) = train_test_split(phy_data, aux_data, test_size=num_test, 
                                                     shuffle=True, random_state=seed)

    num_train = int(settings["proportion_train"] * phy_data.shape[0])
    num_val = phy_data.shape[0] - num_train
    settings["train_phy_shape"] = (num_train, phy_data.shape[1])
    settings["val_phy_shape"] = (num_val, phy_data.shape[1])
    settings["test_phy_shape"] = tuple(test_phy_data.shape)
    settings["train_aux_shape"] = (num_train, aux_data.shape[1])
    settings["val_aux_shape"] = (num_val, aux_data.shape[1])
    settings["test_aux_shape"] = tuple(test_aux_data.shape)
    save_settings(settings, settings['out_prefix'] + "_settings.csv")
        

    ###################################
    # Set up network training objects #
    ###################################
    # create Data container
    ae_data     = AEData( 
                        phy_data         = phy_data,
                        aux_data         = aux_data,
                        aux_colnames     = aux_data_names,
                        prop_train       = settings['proportion_train'],  
                        num_channels     = settings["num_channels"], 
                        num_chars        = settings["num_chars"],
                        device           = settings["device"]
                        )
    
    trn_loader, val_loader = ae_data.get_dataloaders(settings["batch_size"], shuffle = True, 
                                                    num_workers = settings["num_workers"])
    
    phy_normalizer, aux_normalizer = ae_data.get_normalizers()

        
    # create model
    ae_model    = AECNN(
                        num_structured_input_channel  = ae_data.num_channels, 
                        structured_input_width        = ae_data.phy_width,
                        unstructured_input_width      = ae_data.aux_width,
                        aux_numtips_idx               = ae_data.ntax_cidx,
                        stride                        = settings["stride"],
                        kernel                        = settings["kernel"],
                        out_channels                  = settings["out_channels"],
                        latent_output_dim             = settings["latent_output_dim"],
                        latent_layer_type             = settings["latent_model_type"],
                        num_chars                     = settings["num_chars"],
                        char_type                     = settings["char_type"],
                        out_prefix                    = settings["out_prefix"],
                        device                        = settings["device"],
                        phy_normalizer                = phy_normalizer,
                        aux_normalizer                = aux_normalizer
                        )
    
    # optimizer
    # settings
    lr = settings['learning_rate']
    wd = settings['weight_decay']

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
    
    # Loss objects (stateful). These compute and store component
    # and the weighted sum of component losses for the final objective. 
    loss_weights = {k : v for k, v in settings.items() if "_loss_weight" in k}
    train_loss  = PhyLoss(loss_weights, ae_data.ntax_cidx, ae_model.char_type,
                        ae_model.latent_layer_type, device = settings["device"])
    val_loss    = PhyLoss(loss_weights, ae_data.ntax_cidx,  ae_model.char_type,
                        ae_model.latent_layer_type, device = settings["device"], 
                        validation = True)


    # create model trainer
    # the model, the data, and the loss come together here
    tree_ae     = PhyloAutoencoder(
                        model           = ae_model, 
                        optimizer       = opt, 
                        lr_scheduler    = lr_schedlr,
                        batch_size      = settings["batch_size"],
                        train_loss      = train_loss,
                        val_loss        = val_loss,
                        device          = settings["device"]
                        )
    

    #################################
    # Use tree_ae to train ae_model #
    # with data from ae_data.       #
    #################################
    tree_ae.set_data_loaders(train_loader=trn_loader, val_loader=val_loader) 
    tree_ae.train(num_epochs = settings["num_epochs"], seed = settings["seed"])
    if tree_ae.track_grad:
        plot_gradient_norms(tree_ae.mean_layer_grad_norm, 
                            settings["out_prefix"] + ".layer_grad_norms.pdf")
            

    # save model with normalizers
    tree_ae.save_model(settings["out_prefix"] + ".ae_trained.pt")
    # plot loss curves
    tree_ae.plot_losses(settings["out_prefix"])


    # make encoded tree file for 5,000 random trees from training data
    rand_idx        = np.random.randint(0, ae_data.prop_train * phy_data.shape[0], 
                                        size = min(5000, phy_data.shape[0]))
    rand_train_phy  = torch.Tensor(ae_data.norm_train_phy_data[rand_idx,...])
    rand_train_aux  = torch.Tensor(ae_data.norm_train_aux_data[rand_idx,...])
    latent_dat      = ae_model.encode(rand_train_phy, rand_train_aux, 
                                      inference=True, detach=True)
    latent_dat_df   = pd.DataFrame(latent_dat.detach().to('cpu').numpy(), 
                                    columns = None, index = None)
    latent_dat_df.to_csv(settings["out_prefix"] + ".traindat_latent.csv", 
                         header = False, index = False)


    #####################################################
    # Test Data Prediction                              #
    # make predictions with trained model on test data  #
    #####################################################

    # save true values of test data in cblv format
    phy_true_df = pd.DataFrame(test_phy_data)#.numpy())
    aux_true_df = pd.DataFrame(test_aux_data)#.numpy())
    phy_true_df.to_csv(settings["out_prefix"] + ".phy_true.cblv.csv", 
                       header = False, index = False)
    aux_true_df.to_csv(settings["out_prefix"] + ".aux_true.csv", 
                       header = False, index = False)


    test_latent_dat = ae_model.norm_and_encode(test_phy_data, test_aux_data)
    latent_testdat_df = pd.DataFrame(test_latent_dat, columns = None, index = None)
    latent_testdat_df.to_csv(settings["out_prefix"] + ".testdat_latent.csv", 
                             header = False, index = False)

    # # set predicted padding to zeros  (using predicted num tips)
    phy_pred, aux_pred = ae_model.norm_predict_denorm(test_phy_data, test_aux_data)
    phy_pred = utils.set_pred_pad_to_zero(phy_pred,  aux_pred[:,ae_data.ntax_cidx])    
    phy_pred = phy_pred.reshape((phy_pred.shape[0], -1), order = "F")


    # save predictions to file
    phy_pred_df = pd.DataFrame(phy_pred)
    aux_pred_df = pd.DataFrame(aux_pred)
    phy_pred_df.to_csv(settings["out_prefix"] + ".phy_pred.cblv.csv", 
                       header = False, index = False)
    aux_pred_df.to_csv(settings["out_prefix"] + ".aux_pred.csv", 
                       header  = False, index = False)




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--trn_data",         required = True,  help = "Training data in hdf5 format.")
    parser.add_argument("-o", "--out_prefix",       required = False,  help = "Output prefix.")
    parser.add_argument("-cfg", "--config",         required = False, help = "Configuration file. Settings dictionary. Default None.")
    parser.add_argument("-s", "--seed",             required = False, type = int, help = "Random seed. Default random.")
    parser.add_argument("-nw", "--num_workers",     required = False, type = int, help = "Number of workers. Default 0")
    parser.add_argument("-ne", "--num_epochs",      required = False, type = int, help = "Number of training epochs. Default 100")
    parser.add_argument("-b", "--batch_size",       required = False, type = int, help = "Batch size. Default 128")
    parser.add_argument("-mmd","--mmd_loss_weight", required = False, type = float, help = "MMD lambda (>= 0). Default 1.0")
    parser.add_argument("-vz", "--vz_loss_weight",  required = False, type = float, help = "VZ lambda (>= 0). Default 1.0")
    parser.add_argument("-pw", "--phy_loss_weight", required = False, type = float, help = "Phylogenetic loss weight. Default 0.9")
    parser.add_argument("-aw", "--aux_loss_weight", required = False, type = float, help = "Auxiliary loss weight. Default 0.1")
    parser.add_argument("-cw", "--char_loss_weight",required = False, type = float, help = "how much weight to give to char loss. Default 0.0")
    parser.add_argument("-mt", "--max_tips",        required = False, type = float, help = "maximum number of tips.   Default 1000")
    parser.add_argument("-ns", "--num_subset",      required = False, type = int , help = "subset of data used for training/testing. Default None")
    parser.add_argument("-nchans", "--num_channels",  required = False, type = int, help = "number of data channels. Default 9")
    parser.add_argument("-num_chars", "--num_chars",  required = False, type = int, help = "number of characters. Default 5")
    parser.add_argument("-ld", "--latent_output_dim", required = False, type = int, help = "latent output dimension. Default None (determined by structured encoder output shape)")
    parser.add_argument("-l", "--latent_model_type",  required = False, help = "latent model type (GAUSS, DENSE, or CNN). Default GAUSS")
    parser.add_argument("-k", "--kernel",           required = False, type = int, help = "kernel size. Default 3,5,5")
    parser.add_argument("-r", "--stride",           required = False, type = int, help = "stride size. Default 2,4,4")
    parser.add_argument("-oc", "--out_channels",    required = False, type = int, help = "output channels. Default 32,32,128")
    parser.add_argument("-ct", "--char_type",       required = False, help = "character type (categorical or continuous). Default categorical")
    parser.add_argument("-pt", "--proportion-train",required = False, type = float, help = "Proportion of num-subset used for training vs validation. Default 0.85")
    parser.add_argument("-lr", "--learning-rate",   required = False, type = float, help = "Optimizer learning rate. Default 1e-3")
    parser.add_argument("-wd", "--weight-decay",    required = False, type = float, help = "Optimizer weight decay. Default 1e-3")
    parser.add_argument("-t", "--testing",          required = False, type = bool, help = "Testing mode sets torch trianing optimization behavior to deterministic. Default True")
    parser.add_argument("-dv", "--device",          required = False, type = bool, help = "Device. Default auto")
    return parser.parse_args()

def get_default_settings():
    return {
        "testing": True,
        "out_prefix": "out",
        "num_subset": "all",
        "num_workers": 0,
        "seed": np.random.randint(0, 2**32 - 1),
        "num_epochs": 100,
        "batch_size": 128,
        "num_chars": 0,
        "num_channels": 2,
        "max_tips": 1000,
        "mmd_loss_weight": 1.0,
        "vz_loss_weight": 1.0,
        "phy_loss_weight": 0.9,
        "aux_loss_weight": 0.1,
        "char_loss_weight": 1.0,
        "latent_model_type": "GAUSS",
        "latent_output_dim": None,
        "stride": [2, 4, 8],
        "kernel": [3, 5, 9],
        "out_channels": [16, 64, 128],
        "char_type": "categorical",
        "proportion_train" : 0.85,
        "learning_rate" : 1e-3,
        "weight_decay"  : 1e-3,
        "device" : "auto"
    }

def update_settings_from_command_line(settings, args):

    arg_map = {
        "testing"       : args.testing,
        "out_prefix"    : args.out_prefix,
        "num_subset"    : args.num_subset,
        "num_workers"   : args.num_workers,
        "seed"          : args.seed,
        "num_epochs"    : args.num_epochs,
        "batch_size"    : args.batch_size,
        "num_chars"     : args.num_chars,
        "num_channels"  : args.num_channels,
        "max_tips"      : args.max_tips,
        "mmd_loss_weight" : args.mmd_loss_weight,
        "vz_loss_weight"  : args.vz_loss_weight,
        "phy_loss_weight": args.phy_loss_weight,
        "char_loss_weight" : args.char_loss_weight,
        "aux_loss_weight": args.aux_loss_weight,
        "latent_model_type": args.latent_model_type,
        "latent_output_dim": args.latent_output_dim,
        "kernel"        : args.kernel,
        "stride"        : args.stride,
        "out_channels"  : args.out_channels,
        "char_type"     : args.char_type,
        "proportion_train": args.proportion_train,
        "learning_rate" : args.learning_rate,
        "weight_decay"  : args.weight_decay,
        "device"        : args.device

    }

    # override defaults with command line args
    for k, v in arg_map.items():
        if v is not None:
            if k in {"kernel", "stride", "out_channels"} and isinstance(v, str):
                settings[k] = [int(x) for x in v.split(",")]
            elif k in {"latent_output_dim", "num_channels", "num_chars", "num_subset", 
                       "num_epochs", "batch_size", "max_tips", "num_workers", "seed"}:
                settings[k] = int(v)
            elif k in {"mmd_loss_weight", "vz_loss_weight", "aux_loss_weight", "phy_loss_weight", "char_loss_weight"}:
                settings[k] = float(v)
            else:
                settings[k] = v

def update_settings_from_config(settings : dict, config : dict):
    for key in settings:
        if key in config:
            settings[key] = config[key]

def save_settings(settings, out_file):
    df_index= [x for x in settings.keys()]
    df_val  = [str(x) for x in settings.values()]
    df = pd.DataFrame(df_val, index=df_index, columns=None)
    df.to_csv(out_file, sep="\t", header = False)
    print("Settings saved to", out_file)

# TODO: should belong to PhyloAutoencoder
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
