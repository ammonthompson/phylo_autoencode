#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import os
import argparse as arg
import scipy.stats as stats

cmd_params = arg.ArgumentParser()
cmd_params.add_argument("-f", "--infile", required=True, help = "File of latent encoding of training data sample.")
cmd_params.add_argument("-o", "--outfile", required=True, help = "Output pdf containg qqplots")
args = cmd_params.parse_args()
infile = args.infile
out_file = args.outfile

encoded_dat = pd.read_csv(args.infile, header = None, index_col = None)

with PdfPages(out_file) as pdf:
    for i in encoded_dat.columns:
        plt.figure()
        q = np.linspace(0.01, 0.99, 50)
        dat_quantiles = np.quantile(encoded_dat[i], q)
        stdnorm_quantiles = stats.norm.ppf(q)
        plt.scatter(stdnorm_quantiles, dat_quantiles)
        plt.title(f"QQPlot for column {i}")        
        min_val, max_val = min(dat_quantiles), max(dat_quantiles)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Red dashed line
        pdf.savefig()
        plt.close()

print(f"QQPlots saved to {out_file}")