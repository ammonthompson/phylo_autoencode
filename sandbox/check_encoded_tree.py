#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import argparse
import os
import seaborn as sns


def plot_contour(df_1, df_2, percent_variance = None):
    sns.kdeplot(
        x=df_1.iloc[:, 0], y=df_1.iloc[:, 1],
        fill=True, cmap="Blues", alpha=1.0, levels=20,
    )
    plt.scatter(df_2.iloc[:, 0], df_2.iloc[:, 1], color="red", label="Test Trees", alpha=0.7)
    pc_percent_variance = f"({percent_variance[0]:.1f}%, {percent_variance[1]:.1f}%)" if percent_variance is not None else ""
    plt.title(
        f"Contour Plot of {df_1.columns[0]} vs {df_1.columns[1]} {pc_percent_variance}",
        fontsize=14
    )
    plt.xlabel(f"{df_1.columns[0]}", fontsize=12)
    plt.ylabel(f"{df_1.columns[1]}", fontsize=12)
    plt.legend()


def plot_distributions(df_1, df_2, percent_variance = None, output_pdf = "pca_distributions.pdf"):
    """
    Creates a PDF with plots of PCA distributions for df_1 and points for df_2 values.
    The first plot is a contour plot of the first two PCs with points from df_2.
    Subsequent plots show PCA distributions (4 plots per page).

    Args:
        df_1 (pd.DataFrame): DataFrame with PCA results for the first file (used to generate distributions).
        df_2 (pd.DataFrame): DataFrame with PCA results for the second file (values to plot as points).
        percent_variance (list): Percent variance explained by each principal component.
        output_pdf (str): Path to the output PDF file.
    """
    num_components = len(df_1.columns)

    # Open a PDF for saving plots
    with PdfPages(output_pdf) as pdf:
        # First plot: Contour plot for PC1 and PC2 with points from df_2
        plt.figure(figsize=(10, 8))
        plot_contour(df_1, df_2, percent_variance)
        pdf.savefig()
        plt.close()

        # Subsequent plots: 4 per page
        for i, pc in enumerate(df_1.columns):
            if i % 4 == 0:  # Start a new page every 4 plots
                if i > 0:  # Save previous figure
                    pdf.savefig()
                    plt.close()
                plt.figure(figsize=(10, 10))

            pc_percent_variance = f"({percent_variance[i]:.1f}% of variance)" if percent_variance is not None else ""
            # Create a subplot for each PC
            plt.subplot(2, 2, (i % 4) + 1)
            sns.kdeplot(df_1[pc], fill=True, color="blue", alpha=0.5, label=f"{pc} Dist.")
            # Plot vertical lines for df_2
            for value in df_2[pc]:
                plt.axvline(x=value, color="red", linestyle="--", alpha=0.7)
            plt.title(f"{pc_percent_variance}", fontsize=10)
            plt.xlabel(pc, fontsize=10)
            plt.ylabel("Density", fontsize=10)
            # plt.legend(fontsize=8)

        # Save the last figure
        pdf.savefig()
        plt.close()

    print(f"Plots saved to {output_pdf}")


def save_pca_parameters(mean, scale, eigenvalues, eigenvectors, output_file):
    """
    Saves the mean, standard deviations, eigenvalues, and eigenvectors to a file.

    Args:
        mean (array-like): Mean of the original training data.
        scale (array-like): Standard deviations of the original training data.
        eigenvalues (array-like): Eigenvalues of the PCA.
        eigenvectors (array-like): Eigenvectors (principal components) of the PCA.
        output_file (str): Path to the output file (CSV format).
    """
    # Create a DataFrame for eigenvectors
    eigenvector_df = pd.DataFrame(eigenvectors.T, columns=[f"Eigenvector_{i+1}" for i in range(eigenvectors.shape[0])])

    # Create a DataFrame for mean, scale, and eigenvalues
    stats_df = pd.DataFrame({
        "Mean": mean,
        "Standard_Deviation": scale,
        "Eigenvalue": np.append(eigenvalues, [np.nan] * (len(mean) - len(eigenvalues)))  # Handle size mismatch
    })

    # Combine stats and eigenvectors
    combined_df = pd.concat([stats_df, eigenvector_df], axis=1)

    # Write to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"PCA parameters saved to {output_file}")

def transform_new_data(new_data, mean, scale, eigenvectors):
    """
    Transforms new data into principal component space using saved PCA parameters.

    Args:
        new_data (array-like): The new dataset to transform (rows: samples, columns: features).
        mean (array-like): Mean of the original training data.
        scale (array-like): Standard deviation of the original training data.
        eigenvectors (array-like): Eigenvectors from PCA.

    Returns:
        np.ndarray: Transformed data in principal component space.
    """
    # Standardize the new data
    standardized_data = (new_data - mean) / scale

    # Apply PCA transformation
    return np.dot(standardized_data, eigenvectors.T)

def print_variance_explained(explained_variance_ratio):
    print("Percentage of Total Variance Explained (tree_file_1):")
    for i, variance in enumerate(explained_variance_ratio, start=1):
        if variance > 0.1:
            print(f"PC{i}: {variance:.2f}%")


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-trn", "--train_trees", required=True, help = "Encoded file of training trees, assume no header or row names")
    ap.add_argument("-test", "--test_trees", required=False, help = "Encoded file, assume no header or row names")
    ap.add_argument("-o", "--out_prefix", required=True, help = "Encoded file, assume no header or row names")

    args = ap.parse_args()

    train_trees = args.train_trees
    test_trees  = getattr(args, "test_trees", None)
    out_prefix  = args.out_prefix

    assert(os.path.isfile(train_trees))

    df_1 = pd.read_csv(train_trees, header = None, index_col = None)
    df_2 = pd.read_csv(test_trees, header = None, index_col = None)

   # Standardize and perform PCA on the first file
    df1_scaler = StandardScaler()
    scaled_data_1 = df1_scaler.fit_transform(df_1)
    pca = PCA(n_components=df_1.shape[1])
    principal_components_1 = pca.fit_transform(scaled_data_1)

    # Calculate the proportion of total variance explained
    explained_variance_ratio = pca.explained_variance_ratio_ * 100

    # Print Variance Explained
    print_variance_explained(explained_variance_ratio)

    # Save tree stats and PCA-transformed data for the first file
    pc_df_1 = pd.DataFrame(
        data=principal_components_1,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    df_1.to_csv(out_prefix + ".sumstats.csv", header = True, index=False)
    pc_df_1.to_csv(out_prefix + ".pca.csv", index=False)

    # Standardize the test tree file using the same scaler and transform with the fitted PCA
    scaled_data_2 = df1_scaler.transform(df_2)
    principal_components_2 = pca.transform(scaled_data_2)

    # Create a DataFrame for PCA-transformed data from the second file
    pc_df_2 = pd.DataFrame(
        data=principal_components_2,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    # Save PCA-transformed data for the second file
    pc_df_2.to_csv(out_prefix + ".test_trees.pca.csv", index=False)

    # create plot
    plot_distributions(pc_df_1, pc_df_2, explained_variance_ratio, out_prefix + "_pca_plots.pdf")
    plot_distributions(df_1, df_2, None, out_prefix + "_sumstats_plots.pdf")
                                                                                                                         