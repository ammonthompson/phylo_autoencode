#!/usr/bin/env python3
# libraries
import numpy as np
import dendropy as dp
import pandas as pd
import h5py
import argparse
import re

from phyddle.format import Formatter as fmt
from string import ascii_lowercase, ascii_uppercase

import sys


# prepare inorder node heights
def get_node_heights(y, num_tips, chars = None):
    # y is a dataframe with shape (2, max_num_tips)
    # inorder node heights
    nodes = [ None ] * (2*num_tips - 1)

    for i in range(num_tips):
    
        # node heights
        int_height = y.iloc[1,i]
        tip_height = y.iloc[0,i] + int_height
        if(chars is not None):
            tip_char = {"char_" + str(k) : str(np.round(v, decimals=2).item()) 
                    for k,v in enumerate(chars[:,i])}

        # make tip node
        tip_nd = dp.Node(label=f't{i}')
        tip_nd.height = tip_height
        tip_nd.value = y.iloc[0,i]
        tip_nd.index = 2*i
        if chars is not None:
            for key, value in tip_char.items():
                tip_nd.annotations.add_new(key, value)


       # Debugging: Print annotations for the tip node
        # print(f"Tip Node {tip_nd.label} Annotations: {tip_nd.annotations}")


        nodes[tip_nd.index] = tip_nd

        # do not make int node for first tip
        if i == 0:
            continue
        
        # make int node
        int_nd = dp.Node(label=f'n{i}')
        int_nd.height = int_height
        int_nd.value = y.iloc[1,i]
        int_nd.index = 2*i - 1
        nodes[int_nd.index] = int_nd

    

    return nodes

# find oldest internal node from list
def find_oldest_int_node(nodes):
    oldest = 1e6
    index = -1
    for i in range(1,len(nodes),2):
        if nodes[i].height <= oldest:
            index = i
            oldest = nodes[i].height
    return nodes[index]

# build (left,right) node relationships
def recurse(nodes, nd):
    
    # tip node
    # if len(nodes) == 1:
    #     # do nothing
    #     pass
    if nd.label.startswith('t'):
        return nd


    # internal node
    else:
        # find left and right clades
        idx = [ v.index for v in nodes ].index(nd.index)
        nodes_left = nodes[:idx]
        nodes_right = nodes[(idx+1):]

        # find daughters
        nd_left  = find_oldest_int_node(nodes_left)
        nd_right = find_oldest_int_node(nodes_right)

        # recurse
        nd_left  = recurse(nodes_left, nd_left)
        nd_right = recurse(nodes_right, nd_right)

        # attach daughters
        nd.add_child(nd_left)
        nd.add_child(nd_right)

        # update edge lengths
        # for i,ch in enumerate(nd.child_nodes()):
        for ch in nd.child_nodes():
            ch.edge_length = ch.height - nd.height
    
    return nd

# shift node heights by specified shift_value
def shift_node_heights(dp_tree, shift_value):
    for node in dp_tree:
        if node.height is not None:
            node.height += shift_value

# print newick strings to stdout
def convert_to_newick(cblv, num_tips, char_data):
    '''
    Convert cblv format to newick format and print to stdout.
    cblv: cblv format
    num_tips: number of tips in each tree
    '''

    # loop through each tree encoded in cblv format and convert to newick string then print to stdout
    for i in range(cblv.shape[0]):
        num_tips_i = int(num_tips[i])
    
        # cblv is first two channels
        cblv_i = cblv[i, 0:2, 0:num_tips_i]

        char_data_i = char_data[i, :, 0:num_tips_i]
     
        nodes = get_node_heights(pd.DataFrame(cblv_i), num_tips_i, char_data_i)
        heights = [ nd.height for nd in nodes ]        

        # find root node with smallest height and is an internal node (label starts with 'n')
        min_int_node = min([ nd.height for nd in nodes if nd.label.startswith('n') ])
        idx_root = heights.index(min_int_node)

        nd_root = nodes[idx_root] 
    
        nd_root = recurse(nodes, nd_root)
        phy_decode = dp.Tree(seed_node=nd_root)

             
        print(phy_decode.as_string("newick", suppress_leaf_node_labels=False, suppress_annotations=False))


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tree", required=True, help = "File of ntips + cblv(s) formatted phylo data.")
    # parser.add_argument("-a", "--aux", required=True, help = "File of auxilliary data.")
    parser.add_argument("-m", "--max-tips", required=True, help = "Maximum number of tips. Last dimension of cblv tensor.")
    parser.add_argument("-n", "--num-tips", required=False, help = "Num tips in trees. Single column file. Same order as cblv file.")
    parser.add_argument("-c", "--num-chars", required=True, help = "Number of characters.") # deprecated
    parser.add_argument("-x", "--num-trees", required=False, help = "Number of trees to translate.")
    args = parser.parse_args()
    max_tips = int(args.max_tips)
    num_chars = int(args.num_chars)
    

    # read in data from file. Format format for input file see: phyddle -s F
    if re.search(r"\.csv$", args.tree) or re.search(r"\.cblv$", args.tree) or re.search(r"\.cblvs$", args.tree):
        phy_data = pd.read_csv(args.tree, header = None, index_col = None).to_numpy(dtype=np.float64)

    elif re.search(r"\.hdf5$", args.tree) or re.search(r"\.h5$", args.tree):
        with h5py.File(args.tree, "r") as f:
            phy_data = pd.DataFrame(f['phy_data']).to_numpy(dtype=np.float64)

    else:
        print("Input cblv file must be a .csv or .hdf5 file.")
        sys.exit(1)
    # print(phy_data.shape)

    
    # read in num tips file (single column)
    # if num tips is negative, set to 1
    if args.num_tips is not None:
        num_tips = pd.read_csv(args.num_tips, header = None, index_col = None).to_numpy()
        num_tips[num_tips < 2] = 2
        num_tips[num_tips > max_tips] = max_tips
        num_tips = num_tips[:,0]
    # if no file provided, set all num tips to max_tips
    else:
        num_tips = [int(args.max_tips) for i in range(phy_data.shape[0])]

    # convert to cblv format
    cblvs = phy_data.reshape((phy_data.shape[0], phy_data.shape[1]//max_tips, max_tips), order = "F")

    # get character data
    char_data = cblvs[:, (cblvs.shape[1]-num_chars):cblvs.shape[1], :]

    # check if num_tips is same length as cblvs
    if len(num_tips) != cblvs.shape[0]:
        print("num_tips and cblvs must be same length.")
        sys.exit(1)

    num_trees = cblvs.shape[0]
    if args.num_trees is not None:
        num_trees = int(args.num_trees)
    if num_trees > cblvs.shape[0]:
        num_trees = cblvs.shape[0]

    # print newick trees to stdout
    convert_to_newick(cblvs[0:num_trees,...], num_tips[0:num_trees,...], char_data[0:num_trees,...])

if __name__ == "__main__":
    main()