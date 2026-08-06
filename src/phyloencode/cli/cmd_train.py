#!/usr/bin/env python3

import random
import torch
from torch.optim import AdamW
import numpy  as np
import pandas as pd
import argparse

from phyloencode.PhyloAutoencoder   import PhyloAutoencoder
from phyloencode.PhyloAEModel       import AECNN
from phyloencode.DataProcessors     import AEData
from phyloencode.PhyLoss            import PhyLoss
import phyloencode.utils as utils


# TODO: includ min tips
def main():

    args = _parse_arguments()
    settings, checkpoint_overrides = _process_settings(args)

    ###################################
    # Set up network training objects #
    ###################################
    # AEData processes and holds the datasets, normalizers, dimensions, and column metadata.
    # It splits HDF5 rows, fits normalizers on training data, and builds lazy datasets.
    ae_data     = AEData(
                        hdf5_file        = args.trn_data,
                        prop_train       = settings['proportion_train'],
                        num_channels     = settings["num_channels"],
                        char_data_type   = settings["char_type"],
                        num_chars        = settings["num_chars"],
                        seed             = settings["seed"],
                        max_tips         = settings["max_tips"],
                        num_subset       = settings["num_subset"],
                        which_aux        = settings["which_aux"]
                        )
    aux_data_names = ae_data.aux_colnames
    settings["train_phy_shape"] = ae_data.train_phy_shape
    settings["val_phy_shape"]   = ae_data.val_phy_shape
    settings["train_aux_shape"] = ae_data.train_aux_shape
    settings["val_aux_shape"]   = ae_data.val_aux_shape
    
    trn_loader, val_loader = \
        ae_data.get_dataloaders(settings["batch_size"], shuffle = True,
                                 num_workers = settings["num_workers"])
    
    phy_normalizer, aux_normalizer = ae_data.get_normalizers()

    map_location = None if settings["device"] == "auto" else settings["device"]

    if settings["resume_from_checkpoint"] is not None:
        tree_ae = PhyloAutoencoder.load_checkpoint(
                        settings["resume_from_checkpoint"],
                        map_location = map_location,
                        **checkpoint_overrides
                        )
    else:
        # create model
        if settings["pretrained_model"] is not None:
            # TODO: important. If the pretrained model used different training data (almost certainly did)
            # then use new normalizers.
            ae_model = AECNN.load_pretrained_from_file(
                        settings["pretrained_model"],
                        map_location = map_location
                        )
        else:
            ae_model = AECNN(
                            num_structured_input_channel  = ae_data.num_channels, 
                            structured_input_width        = ae_data.phy_width,
                            unstructured_input_width      = ae_data.aux_width,
                            aux_inner_dim                 = settings["aux_inner_dim"],
                            aux_numtips_idx               = ae_data.ntax_cidx,
                            aux_data_names                = aux_data_names,
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
        opt = AdamW(utils.split_params_by_wd(ae_model, wd), lr=lr)
        lr_schedlr = torch.optim.lr_scheduler.OneCycleLR(
                            opt,
                            max_lr=lr,
                            epochs=settings["num_epochs"], 
                            steps_per_epoch=len(trn_loader),
                            pct_start=0.1,               # 10% warmup
                            anneal_strategy='cos',
                            cycle_momentum=False
                            )
        
        # PhyLoss compute and store loss and component losses for the final objective.
        loss_weights = {k : v for k, v in settings.items() if "_loss_weight" in k}
        train_loss = PhyLoss(loss_weights, ae_data.ntax_cidx, ae_model.char_type,
                            ae_model.latent_layer_type, device = settings["device"])
        val_loss   = PhyLoss(loss_weights, ae_data.ntax_cidx, ae_model.char_type,
                            ae_model.latent_layer_type, device = settings["device"], 
                            validation = True)


        # PhyloAutoencoder is the model trainer
        # the model, the data, and the loss come together here
        tree_ae = PhyloAutoencoder(
                            model           = ae_model, 
                            optimizer       = opt, 
                            lr_scheduler    = lr_schedlr,
                            train_loss      = train_loss,
                            val_loss        = val_loss,
                            device          = settings["device"],
                            track_grad      = settings["track_grad"],
                            checkpoints     = settings["checkpoints"],
                            checkpt_file_prefix = settings["out_prefix"]
                            )

    if settings["resume_from_checkpoint"] is not None:
        settings["track_grad"] = tree_ae.track_grad
        settings["checkpoints"] = tree_ae.checkpoints
        settings["out_prefix"] = tree_ae.checkpt_file_prefix
    _save_settings(settings, settings['out_prefix'] + "_settings.csv")
    tree_ae.model.write_network_to_file(settings["out_prefix"] + ".network.txt")
    # ae_model = tree_ae.model
    

    #################################
    # Use tree_ae to train ae_model #
    # with data from ae_data.       #
    #################################
    tree_ae.set_data_loaders(train_loader=trn_loader, val_loader=val_loader) 
    train_seed = None if settings["resume_from_checkpoint"] is not None else settings["seed"]
    tree_ae.train(num_epochs = settings["num_epochs"], seed = train_seed)

    if tree_ae.track_grad:
        tree_ae.plot_gradient_norms(tree_ae.mean_layer_grad_norm, 
                            settings["out_prefix"] + ".layer_grad_norms.pdf")            

    # save model with normalizers
    # tree_ae.save_model(settings["out_prefix"] + ".ae_trained.pt")
    tree_ae.model.save_model(settings["out_prefix"] + ".ae_trained.pt")

    # plot loss curves
    tree_ae.plot_losses(settings["out_prefix"])



def _process_settings(args):
    # Training settings: Architecture, num epochs, batch size, etc.
    # Override settings provided in config file if provided
    settings = _get_default_settings()
    checkpoint_override_names = {
        "track_grad": "track_grad",
        "checkpoints": "checkpoints",
        "out_prefix": "checkpt_file_prefix",
    }
    overridden_settings = set()
    if args.config:
        config = utils.read_config(args.config)
        _update_settings_from_config(settings = settings, config = config)
        overridden_settings.update(config.keys() & checkpoint_override_names.keys())
    # override settings provided as command line arguments
    _update_settings_from_command_line(settings = settings, args = args)
    settings["which_aux"] = _normalize_which_aux(settings["which_aux"])
    for setting_name in checkpoint_override_names:
        if getattr(args, setting_name) is not None:
            overridden_settings.add(setting_name)

    if settings["resume_from_checkpoint"] is not None and settings["pretrained_model"] is not None:
        raise ValueError("Cannot specify both --resume_from_checkpoint and --pretrained_model.")
    if settings["resume_from_checkpoint"] is not None:
        settings["seed"] = _load_seed_from_checkpoint(settings["resume_from_checkpoint"])

    _set_seed(settings)

    checkpoint_overrides = {
        checkpoint_override_names[name]: settings[name]
        for name in overridden_settings
    }
    return settings, checkpoint_overrides

def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--trn_data",         required = True,  help = "Training data in hdf5 format.")
    parser.add_argument("-o", "--out_prefix",       required = False,  help = "Output prefix.")
    parser.add_argument("-cfg", "--config",         required = False, help = "Configuration file. Settings dictionary. Default None.")
    parser.add_argument("-s", "--seed",             required = False, type = int, help = "Random seed. Default random.")
    parser.add_argument("-nw", "--num_workers",     required = False, type = int, help = "Number of workers. Default 0")
    parser.add_argument("-ne", "--num_epochs",      required = False, type = int, help = "Number of training epochs. Default 100")
    parser.add_argument("-b", "--batch_size",       required = False, type = int, help = "Batch size. Default 128")
    parser.add_argument("-aid", "--aux_inner_dim",  required = False, type = int, help = "Hidden width of auxiliary encoder/decoder MLPs. Default 10")
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
    parser.add_argument("-tg", "--track_grad",      required = False, help = "Track parameter gradient norms during training. Default False")
    parser.add_argument("-dv", "--device",          required = False, type = bool, help = "Device. Default auto")
    parser.add_argument("-waux", "--which_aux",     required = False, help = "Comma separated list of auxilliary data column names to inclued. Default: All")
    parser.add_argument("-ckpt", "--checkpoints", required = False, help = "Comma separated list of epochs to save training checkpoints. Default: None")
    parser.add_argument("--resume_from_checkpoint", required = False, help = "Resume training from a provided checkpoint file produced by save_checkpoint.")
    parser.add_argument("--pretrained_model", required = False, help = "Initialize model weights from a saved model and start a fresh training run.")
    return parser.parse_args()

def _load_seed_from_checkpoint(checkpoint_file: str) -> int:
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    return int(checkpoint["seed"])

def _set_seed(settings):
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

def _get_default_settings():
    return {
        "testing": True,
        "out_prefix": "out",
        "num_subset": "all",
        "num_workers": 0,
        "seed": np.random.randint(0, 2**32 - 1),
        "num_epochs": 100,
        "batch_size": 128,
        "aux_inner_dim": 10,
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
        "track_grad" : False,
        "device" : "auto",
        "which_aux" : "all",
        "checkpoints" : None,
        "resume_from_checkpoint": None,
        "pretrained_model": None,
    }

def _update_settings_from_command_line(settings, args):

    arg_map = {
        "testing"       : args.testing,
        "out_prefix"    : args.out_prefix,
        "num_subset"    : args.num_subset,
        "num_workers"   : args.num_workers,
        "seed"          : args.seed,
        "num_epochs"    : args.num_epochs,
        "batch_size"    : args.batch_size,
        "aux_inner_dim" : args.aux_inner_dim,
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
        "track_grad"    : args.track_grad,
        "device"        : args.device,
        "which_aux"     : args.which_aux,
        "checkpoints"   : args.checkpoints,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "pretrained_model": args.pretrained_model,

    }

    # override defaults and config with command line args
    # TODO: many of these havent been tested well
    for k, v in arg_map.items():
        v = _normalize_cli_value(v)
        if v is not None:
            if k in {"kernel", "stride", "out_channels", "checkpoints"} and isinstance(v, str):
                settings[k] = [int(x) for x in v.split(",")]
            elif k in {"latent_output_dim", "num_channels", "num_chars", "num_subset",
                       "num_epochs", "batch_size", "max_tips", "num_workers", "seed", "aux_inner_dim"}:
                settings[k] = int(v)
            elif k in {"mmd_loss_weight", "vz_loss_weight", "aux_loss_weight", "phy_loss_weight", "char_loss_weight"}:
                settings[k] = float(v)
            elif k in {"testing", "track_grad"}:
                settings[k] = bool(v)
            elif k in {"which_aux"}:
                settings[k] = _normalize_which_aux(v)
            else:
                settings[k] = v

def _normalize_cli_value(value):
    if value == "None":
        return None
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower == "true":
            return True
        if value_lower == "false":
            return False
    return value

def _normalize_which_aux(value):
    if isinstance(value, str):
        return "all" if value.lower() == "all" else [x.strip() for x in value.split(",")]
    if isinstance(value, (list, tuple)) and len(value) == 1 and str(value[0]).lower() == "all":
        return "all"
    return value

def _update_settings_from_config(settings : dict, config : dict):
    for key in settings:
        if key in config:
            settings[key] = config[key]

def _save_settings(settings, out_file):
    df_index= [x for x in settings.keys()]
    df_val  = [str(x) for x in settings.values()]
    df = pd.DataFrame(df_val, index=df_index, columns=None)
    df.to_csv(out_file, sep="\t", header = False)
    print("Settings saved to", out_file)

if __name__ == "__main__":
    main()
