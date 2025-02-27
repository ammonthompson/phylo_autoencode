#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import os
import argparse as parg
import scipy.stats as stats

''' Create qqplots for each column in a file of latent encodings. 
    This is to check if the latent encodings are N(0,1) distributed.'''

cmd_params = parg.ArgumentParser()
cmd_params.add_argument("-f", "--infile", required=True, help = "File of latent encoding of training data sample.")
cmd_params.add_argument("-o", "--outfile", required=True, help = "Output pdf containg qqplots")
args = cmd_params.parse_args()

encoded_dat = pd.read_csv(args.infile, header = None, index_col = None)
ks_stats = []
ks_pvals = []

# set up a reference wasserstein distance 
# which is the  distance of two random 
# standardnormal data sets
# of size equal to that of the encoded data
relative_wasserstein = []
wass_ref_set = [stats.wasserstein_distance(np.random.normal(size = encoded_dat.shape[0]),
                                           np.random.normal(size = encoded_dat.shape[0])) for i in range(5000)]
wasserstein_ref = np.mean(wass_ref_set)

print("wasserstein reference completed")
for i in encoded_dat.columns:
    encoded_dat[i] = (encoded_dat[i] - np.mean(encoded_dat[i])) / np.std(encoded_dat[i])
    ks_stats.append(stats.ks_1samp(encoded_dat[i], stats.norm.cdf))
    ks_pvals.append(ks_stats[i].pvalue)
    wass_i = stats.wasserstein_distance(encoded_dat[i], 
                                        np.random.normal(size = encoded_dat.shape[0]))
    relative_wasserstein.append(wass_i / wasserstein_ref)
print("column stats completed")
print("plotting...")
with PdfPages(args.outfile) as pdf:
    # histogram of ks statistics
    plt.figure()
    plt.hist([x.statistic for x in ks_stats], bins = 20)
    plt.title("Distribution of KS Statistics")
    plt.xlabel("KS Statistic")
    plt.ylabel("Frequency")
    # vertical line for significance of ks test for standand normal of size encoded_dat.shape[1]    
    plt.axvline(x=1.96/np.sqrt(len(encoded_dat[i])), color='r', linestyle='--')
    pdf.savefig()
    plt.close()

    # plot pval histogram
    plt.figure()
    plt.hist(ks_pvals, bins = 20)
    plt.axline((0.05, 0), (0.05, 100), color = "red", linestyle = "--")
    plt.title("Distribution of KS P-values")
    plt.xlabel("P-value")
    plt.ylabel("Frequency")
    pdf.savefig()
    plt.close()

    # relative wasserstein distance which is wasserstein distance of encoded data
    # divided by reference: a wasserstein distance of two random normal data sets of equal size
    plt.figure()
    plt.hist(relative_wasserstein, bins = 20)
    plt.title("Distribution of Relative Wasserstein Distances")
    plt.xlabel("Relative Wasserstein Distance")
    plt.ylabel("Frequency")
    pdf.savefig()
    plt.close()

    # qqplots
    for i in encoded_dat.columns:
        if i % 4 == 0:
            if i > 0:
                pdf.savefig()
                plt.close() 
            
            plt.figure()
            plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust margins
        # create qqplot
        plt.subplot(2, 2, (i % 4) + 1)
        # set up quantiles
        q = np.linspace(0.01, 0.99, 25)
        dat_quantiles = np.quantile(encoded_dat[i], q)
        stdnorm_quantiles = stats.norm.ppf(q)
        # plot quantiles
        plt.scatter(stdnorm_quantiles, dat_quantiles, c="black", s=20)  # Set point size to 20 points^2
        plt.title(f"QQPlot for column {i}", fontsize=8)        
        text_col = "red" if ks_pvals[i] < 0.05 else "black"
        plt.text(min(stdnorm_quantiles), 0.9 * max(dat_quantiles), f"KS Statistic: {ks_stats[i].statistic:.3f}", fontsize=7, c=text_col)    
        plt.text(min(stdnorm_quantiles), 0.8 * max(dat_quantiles), f"P-value: {ks_pvals[i]:.4f}", fontsize=5, c=text_col)    
        plt.text(min(stdnorm_quantiles), 0.65 * max(dat_quantiles), f"Rel. Wass.: {relative_wasserstein[i]:.1f}", fontsize=7, c=text_col)    
        min_val, max_val = min(dat_quantiles), max(dat_quantiles)
        # y = x line
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Red dashed line
        # equal mean point
        plt.plot([0], [0], '+', markersize = 10, linewidth=2, c="blue")  # Green plus sign at origin
        plt.xlabel("N(0,1)", fontsize=6)
        plt.ylabel(f"Column {i}", fontsize=6)
        # pdf.savefig()
        # plt.close()

print(f"QQPlots saved to {args.outfile}")