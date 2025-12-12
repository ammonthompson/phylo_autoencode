#!/usr/bin/env python3

import os
import sys
# import command line argument parser
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser(description='Takes in two AE encoded trees and outputs a set of intermediate encoded trees.')

argparser.add_argument('-f1', '--file1',    required=True, type=str, help='First AE-encoded tree file (one tree) to compare')
argparser.add_argument('-f2', '--file2',    required=True, type=str, help='Second AE-encoded tree file (one tree) to compare')
argparser.add_argument("-n", "--num_steps", required=True, type=int, default=10, help="Number of steps to take to transform tree1 to tree2")
argparser.add_argument("-o", "--output",    required=True, type=str, default="output.txt", help="Output file name")

args = argparser.parse_args()

nsteps = args.num_steps + 1
tree1 = args.file1
tree2 = args.file2
output_file = args.output

# loop through number of steps and each dimension to create N intermediate trees
# for each dimension, take a step size of (tree2[dim_i] - tree1[dim_i]) / num_steps
# for each step, add the step size to the tree1 and create a new tree
# write the new tree to a file
# the new tree should be in the same format as the input trees
# the output file should be in the same format as the input trees
# the output file should be a list of trees, one per line

for tree in [tree1, tree2]:
    if not os.path.exists(tree):
        print(f"Error: {tree} does not exist")
        sys.exit(1)

    if not os.path.isfile(tree):
        print(f"Error: {tree} is not a file")
        sys.exit(1)

tree1 = pd.read_csv(tree1, header=None).to_numpy()
tree2 = pd.read_csv(tree2, header=None).to_numpy()

intermediate_trees = np.ndarray(shape =(nsteps, tree1.shape[1]), dtype = np.float64)

for step in range(nsteps):
    # open the output file for writing
    # loop through each dimension of the input trees
    intermediate_tree_at_step = []
    for dim in range(tree1.shape[1]):
        # calculate the step size for this dimension
        tz1 = tree1[0,dim]
        tz2 = tree2[0,dim]
        step_size = (tz2 - tz1) / nsteps
        # calculate the new tree value for this dimension
        new_tz = tree1[0,dim] + step * step_size
        intermediate_tree_at_step.append(np.round(new_tz, 8))

        # add to the intermediate trees
    intermediate_trees[step] = intermediate_tree_at_step

pd.DataFrame(intermediate_trees).to_csv(output_file, header=None, index=None)



