#!/usr/bin/env python3
import h5py
import numpy as np
import os
import argparse


def create_MVN_dataset(template_fn, mode1_means, mode2_means, variance, num_samples=1000, output_file='mvn_dataset.h5'):
    """
    Create a synthetic dataset of multivariate normal samples based on a template file.

    Args:
        template_fn (str): Path to the template HDF5 file containing auxiliary data names.
        second_mode (np.ndarray): A numpy array representing the second mode of the multivariate normal distribution.
        num_samples (int): Number of samples to generate. Default is 1000.
        mvn_shape (tuple): Shape of the multivariate normal samples. Default is (1, 1000).
        output_file (str): Path to save the generated dataset. Default is 'mvn_dataset.h5'.

    Returns:
        None
    """
    with h5py.File(template_fn, 'r') as f:
        aux_colnames = f['aux_data_names'][:]


    data_dim = mode1_means.shape[0]  # Ensure data_dim matches the shape of the means

    Sigma = variance * np.eye(data_dim)  # scaled Identity covariance matrix
    
    # Generate multivariate normal samples
    # zero_mode_mvn_samples = np.random.multivariate_normal(mean=np.zeros(data_dim), cov=Sigma, size=num_samples//2)
    # in_mode_mvn_samples = np.random.multivariate_normal(mean=second_mode, cov=Sigma, size=num_samples//2)

    mode1_samples = np.random.multivariate_normal(mean=mode1_means, cov=Sigma, size=num_samples//2)
    mode2_samples = np.random.multivariate_normal(mean=mode2_means, cov=Sigma, size=num_samples//2)
    phy_mvn_samples = np.concatenate((mode1_samples, mode2_samples), axis=0)


    shuffled_indices = np.random.permutation(num_samples)
    phy_mvn_samples = phy_mvn_samples[shuffled_indices]


    aux_mvn_samples = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, aux_colnames.shape[1]))
    
    # Create output HDF5 file
    with h5py.File(output_file, 'w') as f_out:
        f_out.create_dataset('phy_data', data=phy_mvn_samples)
        f_out.create_dataset("aux_data", data=aux_mvn_samples)
        f_out.create_dataset('aux_data_names', data=aux_colnames)
    
    print(f"Dataset created and saved to {output_file}")

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Create a synthetic MVN dataset based on a template file.")
    argparser.add_argument("-tf", "--template_fn", type=str, required=True, help="Path to the template HDF5 file.")
    argparser.add_argument("-ns", "--num_samples", type=int, default=1000, help="Number of samples to generate.")
    argparser.add_argument("-m1g", "--mode1_gen", type=str, default="np.zeros((0, 1000))", help="Function to generate the first mode of the MVN distribution. Default is np.zeros((1, 1000)).")
    argparser.add_argument("-m2g", "--mode2_gen", type=str, default="np.zeros((1, 1000))", help="Function to generate the second mode of the MVN distribution. Default is np.zeros((1, 1000)).")
    argparser.add_argument("-v", "--variance", type=float, default=0.001, help="Variance for the MVN distribution. Default is 0.001.")  
    argparser.add_argument("-of", "--output_file", type=str, default='mvn_dataset.h5', help="Path to save the generated dataset.")

    args = argparser.parse_args()
    template_fn = args.template_fn
    num_samples = args.num_samples
    output_file = args.output_file
    mode1_gen_str = args.mode1_gen
    mode2_gen_str = args.mode2_gen
    variance = args.variance

    # convert string representation of function into a callable function
    mode1 = eval(mode1_gen_str, {"np":np})
    mode2 = eval(mode2_gen_str, {"np":np})


    create_MVN_dataset(template_fn, mode1, mode2, 
                       variance    = variance, 
                       num_samples = num_samples, 
                       output_file = output_file)
