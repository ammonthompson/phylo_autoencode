# TODO runs model.norm_predict_denorm

import argparse

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