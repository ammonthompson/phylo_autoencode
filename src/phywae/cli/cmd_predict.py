
import torch
from phywae.PhyloAEModel import AECNN
import phywae.utils as utils
import h5py
from types import SimpleNamespace
import pandas as pd
import numpy as np
from phywae.cli._output import print_output_files

# parse command line arguments
# - should take in a h5py file that contains phy and aux data (either together in phyddle output, or separate csvs)
# - take in a trained AE torch nn.Module.
# - output a predicted flattened cblv.csv and aux.csv file

# algo
# get model expected data input shapes
# check input data matches (after de-flattening)
# reshape (de-flatten)
# call model.norm_predict_denorm
# flatten
# put in pd.dataframe
# write to csv


def main(args):
    settings = _process_args(args)
    model = settings['model']
    data  = settings['data']

    pred_phy, pred_aux = model.norm_predict_denorm(data.phy_data, data.aux_data)
    pred_phy = utils.set_pred_pad_to_zero(pred_phy, pred_aux[:,model.aux_numtips_idx])
    pred_phy_flat = pred_phy.reshape((pred_phy.shape[0], -1), order="F")

    if settings['out_format'] == "csv":
        pred_phy_df = pd.DataFrame(pred_phy_flat)
        pred_aux_df = pd.DataFrame(pred_aux)

        phy_pred_out_fn = settings['out_prefix'] + ".phy_pred.cblv.csv"
        aux_pred_out_fn = settings['out_prefix'] + ".aux_pred.csv"

        pred_phy_df.to_csv(phy_pred_out_fn, header=False, index=False)
        pred_aux_df.to_csv(aux_pred_out_fn, header=False, index=False)
        output_files = [phy_pred_out_fn, aux_pred_out_fn]

    elif settings['out_format'] == "hdf5":
        hdf5_out_fn = settings['out_prefix'] + ".hdf5"
        with h5py.File(hdf5_out_fn, mode="w") as f:
            aux_names = np.asarray(data.aux_data_names, dtype="S64").reshape((1,-1))

            f.create_group("input")
            f["input"].create_dataset("phy_data", data=data.phy_data)
            f["input"].create_dataset("aux_data_names", data=aux_names, dtype="S64")
            f["input"].create_dataset("aux_data", data=data.aux_data)

            f.create_group("prediction")
            f["prediction"].create_dataset("pred_phy", data=pred_phy_flat)
            f["prediction"].create_dataset("aux_data_names", data=aux_names, dtype="S64")
            f["prediction"].create_dataset("pred_aux", data=pred_aux)
        output_files = [hdf5_out_fn]

    else:
        raise ValueError(f"Unrecognized out_format: {settings['out_format']}. Must be csv or hdf5")

    # create nwk files for trees and their reconstructions (same order in 2 files)
    true_phy = np.asarray(data.phy_data).reshape(pred_phy.shape, order="F")
    true_num_tips = np.asarray(data.aux_data)[:, model.aux_numtips_idx]
    pred_num_tips = pred_aux[:, model.aux_numtips_idx]
    num_chars = model.num_chars
    char_slice = slice(-num_chars, None) if num_chars else slice(0, 0)

    nwk_true = utils.convert_to_newick(
        true_phy[:, :2, :], true_num_tips, true_phy[:, char_slice, :]
    )
    nwk_pred = utils.convert_to_newick(
        pred_phy[:, :2, :], pred_num_tips, pred_phy[:, char_slice, :]
    )

    true_tree_out_fn = settings['out_prefix'] + ".true.tre"
    pred_tree_out_fn = settings['out_prefix'] + ".pred.tre"
    with open(true_tree_out_fn, "w") as f:
        f.writelines(nwk_true)
    with open(pred_tree_out_fn, "w") as f:
        f.writelines(nwk_pred)

    print_output_files(output_files + [true_tree_out_fn, pred_tree_out_fn])




def add_arguments(parser):
    parser.add_argument("-d", "--data", required=True,
                        help="hdf5 file. Contains cblv formated phylogenetic data from Phyddle.")
    parser.add_argument("-m", "--model", required=True,
                        help="Trained model artifact or trainer checkpoint.")
    parser.add_argument("-o", "--out-prefix", required=True,
                        help="Output files prefix.")
    parser.add_argument("-ofmt", "--out-format", required=False,
                        default="hdf5", choices=["hdf5","csv"],  help="File format for outputs.")


def _process_args(args) -> dict:
    model_fn = args.model
    data_fn  = args.data
    out_prefix = args.out_prefix
    out_fmt = args.out_format

    # get data and model
    model = AECNN.load_pretrained_from_file(model_fn)

    data = SimpleNamespace()
    with h5py.File(data_fn, mode="r") as f:
        setattr(data, "phy_data", f['phy_data'][...])
        # aux_data_names = np.array([x.decode() for x in f["aux_data_names"][0,...]])
        aux_data_names = f["aux_data_names"][0,...]
        sub_aux_data_names, sub_aux_data = \
             utils.get_aux_data(aux_data_names,
                                f['aux_data'][...],
                                model.aux_data_names)
        setattr(data, "aux_data_names", sub_aux_data_names)
        setattr(data, "aux_data", sub_aux_data)
        setattr(data, "data_fn", data_fn)

    settings = {"model":model,
                "data":data,
                "out_prefix":out_prefix,
                "out_format":out_fmt}
    _validate_settings(settings)

    return settings

def _validate_settings(settings) -> None:
    settings["model"].validate_num_tips(settings["data"].aux_data)
