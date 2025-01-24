import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import numpy as np
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ete3 import Tree
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

def get_tree_diameter(tree):
    """
    Compute the diameter (longest path) of a tree using an optimized two-pass method.

    Args:
        tree (ete3.Tree): An Ete3 Tree object.

    Returns:
        float: The tree diameter (longest distance between any two leaves).
    """
    # Helper function to find the farthest leaf and its distance
    def farthest_leaf(node, tree):
        """
        Find the farthest leaf from a given node and its distance.

        Args:
            node (ete3.TreeNode): The starting node.
            tree (ete3.Tree): The tree object for distance calculations.

        Returns:
            tuple: (farthest leaf node, distance to that leaf)
        """
        max_distance = 0
        farthest = None
        for leaf in tree.get_leaves():
            dist = tree.get_distance(node, leaf)
            if dist > max_distance:
                max_distance = dist
                farthest = leaf
        return farthest, max_distance

    # Step 1: Pick an arbitrary leaf
    root = tree.get_leaves()[0]

    # Step 2: Find the farthest leaf from the root
    leaf_a, _ = farthest_leaf(root, tree)

    # Step 3: Find the farthest leaf from leaf_a
    _, diameter = farthest_leaf(leaf_a, tree)

    return diameter

    # Get all leaves of the tree
    leaves = tree.get_leaves()
    num_leaves = len(leaves)
    total_cophenetic = 0

    # Iterate over all pairs of leaves
    for i in range(num_leaves):
        for j in range(i + 1, num_leaves):
            # Get the MRCA of the pair
            mrca = tree.get_common_ancestor(leaves[i], leaves[j])
            # Add the distance from the root to the MRCA
            total_cophenetic += tree.get_distance(tree.get_tree_root(), mrca)

    return total_cophenetic

def get_balance_index(tree):
    bi = sum(
        min(len(child1), len(child2)) / max(len(child1), len(child2))
        for node in tree.traverse() if len(node.children) == 2
        for child1, child2 in [(node.children[0], node.children[1])]
    )
    return(bi)

def get_colless_index(tree):
    # Colless index: Sum of absolute differences in subtree sizes
    ci = sum(
        abs(len(child1) - len(child2))
        for node in tree.traverse() if not node.is_leaf() and len(node.children) == 2
        for child1, child2 in [(node.children[0], node.children[1])]
    )
    return(ci)

def compute_ladder_statistic(tree):
    """
    Computes the ladder statistic of a phylogenetic tree.

    Args:
        tree (Tree): A phylogenetic tree in ETE3 format.

    Returns:
        float: The ladder statistic of the tree.
    """
    # Ensure the tree is rooted
    root = tree.get_tree_root()
    
    # Recursive function to compute the depth of each internal node
    def get_node_depths(node, depth=0):
        if node.is_leaf():
            return 0  # Skip leaves
        node_depth = depth  # Depth of the current node
        child_depths = [get_node_depths(child, depth + 1) for child in node.get_children()]
        return node_depth + sum(child_depths)
    
    # Calculate the ladder statistic
    ladder_statistic = get_node_depths(root)
    return ladder_statistic

def get_sackin_index(tree):
    # Sackin index: Sum of root-to-leaf distances
    si = sum(tree.get_distance(leaf) for leaf in tree.get_leaves())
    return(si)

def compute_trees_stats(nwk_string):

    tree = Tree(nwk_string, format=1)  # Load tree in Newick format

    # Number of nodes and leaves
    num_leaves = len(tree.get_leaves())
    # may need for trees with polytomies (unless I resolve those to small branches)
    # num_internal_nodes = len([node for node in tree.traverse() if not node.is_leaf()])

    # Collect branch lengths
    branch_lengths = [node.dist for node in tree.traverse() if not node.is_root()]
    total_branch_length = sum(branch_lengths)
    mean_branch_length = np.mean(branch_lengths) if branch_lengths else 0
    median_branch_length = np.median(branch_lengths) if branch_lengths else 0
    max_branch_length = max(branch_lengths) if branch_lengths else 0
    min_branch_length = min(branch_lengths) if branch_lengths else 0
    std_branch_length = np.std(branch_lengths) if branch_lengths else 0

    # terminal branch lengths
    # Collect terminal branch lengths
    terminal_branch_lengths = [leaf.dist for leaf in tree.get_leaves()]
    mean_terminal = np.mean(terminal_branch_lengths)
    median_terminal = np.median(terminal_branch_lengths)
    min_terminal = np.min(terminal_branch_lengths)
    max_terminal = np.max(terminal_branch_lengths)
    std_terminal = np.std(terminal_branch_lengths)
    total_terminal_branch = sum(terminal_branch_lengths)
    
    # Root-to-tip distances
    root_to_tip_distances = [tree.get_distance(leaf) for leaf in tree.get_leaves()]
    max_root_to_tip = max(root_to_tip_distances) if root_to_tip_distances else 0
    mean_root_to_tip = sum(root_to_tip_distances) / len(root_to_tip_distances) if root_to_tip_distances else 0
    median_root_to_tip = sorted(root_to_tip_distances)[len(root_to_tip_distances) // 2] if root_to_tip_distances else 0
    total_root_to_tip = sum(root_to_tip_distances) if root_to_tip_distances else 0
    std_root_to_tip = np.std(root_to_tip_distances) if root_to_tip_distances else 0
    range_root_to_tip = max_root_to_tip - min(root_to_tip_distances) if root_to_tip_distances else 0

    # topology metrics
    # tree_diameter is really slow and highly correlated with several other stats
    # tree_diameter = get_tree_diameter(tree)
    balance_index = get_balance_index(tree)        
    colless_index = get_colless_index(tree)
    sackin_index = get_sackin_index(tree)
    ladder_stat = compute_ladder_statistic(tree)

    # Add stats to list
    stats = {
        "log_num_leaves": np.log(num_leaves),
        "log_total_branch_length": np.log(total_branch_length),
        "log_mean_branch_length": np.log(mean_branch_length),
        "log_median_branch_length": np.log(median_branch_length),
        "log_max_branch_length": np.log(max_branch_length),
        "log_min_branch_length": np.log(min_branch_length),
        "log_std_branch_length": np.log(std_branch_length),
        "log_max_root_to_tip": np.log(max_root_to_tip),
        "log_mean_root_to_tip": np.log(mean_root_to_tip),
        "log_median_root_to_tip": np.log(median_root_to_tip),
        "log_range_root_to_tip": np.log(range_root_to_tip),
        "log_std_root_to_tip": np.log(std_root_to_tip),
        "log_total_root_to_tip": np.log(total_root_to_tip),
        # "log_tree_diameter": np.log(tree_diameter),
        "log_colless_index_plus_1": np.log(colless_index + 1),
        "log_balance_index": np.log(balance_index),
        "log_sackin_index": np.log(sackin_index),
        "log_ladder_stat" : np.log(ladder_stat),
        "log_mean_terminal" : np.log(mean_terminal),
        "log_median_terminal" : np.log(median_terminal),
        "log_min_terminal" : np.log(min_terminal),
        "log_max_terminal" : np.log(max_terminal),
        "log_std_terminal" :np.log(std_terminal),
        "log_total_terminal_branch": np.log(total_terminal_branch)
    }

    return(stats)

def parallel_compute_tree_statistics(tree_file):
    """
    Compute statistics for phylogenetic trees in a single Newick file.

    Args:
        tree_file (str): Path to the Newick file containing one or more trees.

    Returns:
        list of dict: Summary statistics for each tree.
    """
    trees = []
    with open(tree_file, "r") as f:
        for t in f:
            tree_string = t.strip()
            if len(Tree(tree_string, format=1).get_leaves()) > 2:
                trees.append(t.strip())

    with Pool(processes = cpu_count() - 2) as pool:
        tree_stats = pool.map(compute_trees_stats, trees)

    return tree_stats

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


# run if called from cmd
if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <tree_file_1> <tree_file_2>")
        sys.exit(1)

    # Paths to the input files
    tree_file_1 = sys.argv[1]
    tree_file_2 = sys.argv[2]

    out_prefix = "out"
    if len(sys.argv) > 3:
        out_prefix = sys.argv[3]

    # Compute statistics for the first file
    # stats_1 = compute_tree_statistics(tree_file_1)
    stats_1 = parallel_compute_tree_statistics(tree_file_1)
    df_1 = pd.DataFrame(stats_1)

    # Standardize and perform PCA on the first file
    df1_scaler = StandardScaler()
    scaled_data_1 = df1_scaler.fit_transform(df_1)
    pca = PCA(n_components=len(stats_1[0].keys()))
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

    # Compute statistics for the second file
    stats_2 = parallel_compute_tree_statistics(tree_file_2)
    df_2 = pd.DataFrame(stats_2)

    # Standardize the second file using the same scaler and transform with the fitted PCA
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
