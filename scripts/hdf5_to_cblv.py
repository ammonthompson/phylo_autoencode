#!/usr/bin/env python3
import pandas as pd
import numpy as np  
import h5py
import os
import argparse

argparser = argparse.ArgumentParser(description = "convert cblv h5 file to csv")
argparser.add_argument("-i", "--infile", type=str, help="input h5 file")
argparser.add_argument("-o", "--outfile", type=str, help="output csv file")

args = argparser.parse_args()
infile = args.infile    
outfile = args.outfile

with h5py.File(infile, "r") as f:
    cblv_data = f['phy_data'][...]

data_df = pd.DataFrame(cblv_data, columns = None)
data_df.to_csv(outfile, index = False, header = False)
print(f"Converted {infile} to {outfile}")