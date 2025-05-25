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

# set up a reference wasserstein distance  which is the  distance of two random 
# standard normal data sets of size equal to that of the encoded data
relative_wasserstein = []
relative_wass_pvals = []
wass_ref_set = [stats.wasserstein_distance(np.random.normal(size = encoded_dat.shape[0]),
                                           np.random.normal(size = encoded_dat.shape[0])) for i in range(5000)]
mean_wasserstein_ref = np.mean(wass_ref_set)
scaled_wass_ref_set = np.array([x / mean_wasserstein_ref for x in wass_ref_set])
scaled_wass_q95 = np.quantile(scaled_wass_ref_set, 0.95)
print("wasserstein reference completed")

# compute ks test and wasserstein distance for each column
for i in encoded_dat.columns:
    # encoded_dat[i] = (encoded_dat[i] - np.mean(encoded_dat[i])) / np.std(encoded_dat[i])
    ks_stats.append(stats.ks_1samp(encoded_dat[i], stats.norm.cdf))
    ks_pvals.append(ks_stats[i].pvalue)
    rel_wass_i = stats.wasserstein_distance(encoded_dat[i], 
                                        np.random.normal(size = encoded_dat.shape[0])) / mean_wasserstein_ref
    relative_wasserstein.append(rel_wass_i)
    # compute empirical p-value for relative wasserstein distance
    rel_wass_i_pval = np.sum(np.array(rel_wass_i) <= scaled_wass_ref_set) / len(scaled_wass_ref_set)
    relative_wass_pvals.append(rel_wass_i_pval)

print("column stats completed")
print("plotting...")

with PdfPages(args.outfile) as pdf:
    # KOLMOGOROV-SMIRNOV TEST
    # histogram of ks statistics
    plt.figure()
    plt.hist([x.statistic for x in ks_stats], bins = 20)
    plt.title("Distribution of KS Statistics")
    plt.xlabel("KS Statistic")
    plt.ylabel("Frequency")
    # vertical line for significance of ks test for standand normal of size encoded_dat.shape[1]    
    plt.axvline(x=1.96/np.sqrt(len(encoded_dat[i])), color='r', linestyle='--')
    plt.text(1.01 * (1.96/np.sqrt(len(encoded_dat[i]))), 0.9 * max(np.histogram([x.statistic for x in ks_stats], bins = 20)[0]),    
             f"Significance: {1.96/np.sqrt(len(encoded_dat[i])):.2f}", fontsize=8, c="red")
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


    
    # WASSERSTEIN DISTANCE
    # relative wasserstein distance which is wasserstein distance of encoded data
    # divided by reference mean: a wasserstein distance of two random normal data sets of equal size
    # relative wasserstein distance of 1 means that the data is as similar to a standard normal as a random 
    # normal data set of equal size is on average
    plt.figure()
    plt.hist(relative_wasserstein, bins = 20)
    # plt.axline((1, 0), (1, 100), color = "red", linestyle = "--")
    plt.axvline(x=np.quantile(scaled_wass_ref_set, 0.95), color='r', linestyle='--')
    plt.title("Distribution of Relative Wasserstein Distances")
    plt.xlabel("Relative Wasserstein Distance")
    plt.ylabel("Frequency")
     # add text to show the value of the 5% upper quantile
    plt.text(1.01 * scaled_wass_q95, 0.9 * max(np.histogram(scaled_wass_ref_set, bins = 20)[0]),
             f"5% upper quantile: {scaled_wass_q95:.2f}", fontsize=8, c="red")
    pdf.savefig()
    plt.close()

    # plot histogram of relative wasserstein distance p-values
    plt.figure()
    plt.hist(relative_wass_pvals, bins = 20)
    plt.axline((0.05, 0), (0.05, 100), color = "red", linestyle = "--")
    plt.title("Distribution of Relative Wasserstein Distances P-values")
    plt.xlabel("P-value")
    plt.ylabel("Frequency")
    pdf.savefig()
    plt.close()
    

    # Mahalanobis distance
    # If x is n iid N(0,1) rv's then xTx ~ chi-square(df = n) 
    df = encoded_dat.shape[1]
    mahalanobis = np.sum(encoded_dat**2, axis=1)
    x = np.linspace(np.min(mahalanobis), np.max(mahalanobis),  num = 200)
    y = stats.chi2.pdf(x, df)
    plt.figure()
    plt.hist(mahalanobis, bins = 50, density = True, color = "blue", )
    plt.plot(x, y, linestyle = "-", color = "red")
    plt.title("Distribution of Mahalanobis distance")
    plt.xlabel("Mahal. dist.")
    plt.ylabel("Density")
    pdf.savefig()
    plt.close()


    ###########
    # qqplots #
    ###########
    # create qqplots for each column
    # set up a 2x2 grid of subplots
    # set up a pdf file to save the plots

    # first plot
    plt.figure()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust margins 

    # loop over each column in the encoded data
    # and create a qqplot for each 
    for i in encoded_dat.columns:
    
        # create qqplot
        plt.subplot(2, 2, (i % 4) + 1)
        # set up quantiles
        q = np.linspace(0.01, 0.99, 25)
        dat_quantiles = np.quantile(encoded_dat[i], q)
        stdnorm_quantiles = stats.norm.ppf(q)
        # plot quantiles
        plt.scatter(stdnorm_quantiles, dat_quantiles, c="black", s=20)  # Set point size to 20 points^2
        plt.title(f"QQPlot for dim {i}", fontsize=8)        
        text_col = "red" if ks_pvals[i] < 0.05 else "black"
        plt.text(min(stdnorm_quantiles), 0.9 * max(dat_quantiles), f"KS Statistic: {ks_stats[i].statistic:.2f}", fontsize=7, c=text_col)    
        plt.text(min(stdnorm_quantiles), 0.8 * max(dat_quantiles), f"P-value: {ks_pvals[i]:.3f}", fontsize=5, c=text_col)    
        plt.text(min(stdnorm_quantiles), 0.65 * max(dat_quantiles), f"Rel. Wass.: {relative_wasserstein[i]:.2f}", fontsize=7, c=text_col)    
        plt.text(min(stdnorm_quantiles), 0.55 * max(dat_quantiles), f"P-value: {relative_wass_pvals[i]:.3f}", fontsize=5, c=text_col)        
        
        min_val, max_val = min(dat_quantiles), max(dat_quantiles)
        # y = x line
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Red dashed line
        # equal mean point
        plt.plot([0], [0], '+', markersize = 10, linewidth=2, c="blue")  # Green plus sign at origin
        plt.xlabel("N(0,1)", fontsize=6)
        plt.ylabel(f"Dim {i}", fontsize=6)

        if (i+1) % 4 == 0:
            # save the figure
            pdf.savefig()
            plt.close() 
            if i == len(encoded_dat.columns) - 1:
                # plotting is finished
                break
            else:
                # start a new page
                plt.figure()
                plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust margins


print(f"QQPlots saved to {args.outfile}")