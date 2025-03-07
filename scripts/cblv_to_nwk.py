#!/usr/bin/env python3
# libraries
import numpy as np
import dendropy as dp
import pandas as pd
import h5py
import argparse

from phyddle.format import Formatter as fmt
from string import ascii_lowercase, ascii_uppercase

import sys


# prepare inorder node heights
def get_node_heights(y, num_tips):
    # inorder node heights
    nodes = [ None ] * (2*num_tips - 1)

    for i in range(num_tips):
    
        # node heights
        int_height = y.iloc[1,i]
        tip_height = y.iloc[0,i] + int_height

        # make tip node
        tip_nd = dp.Node(label=f't{i}')
        tip_nd.height = tip_height
        tip_nd.value = y.iloc[0,i]
        tip_nd.index = 2*i
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
    if len(nodes) == 1:
        # do nothing
        pass

    # internal node
    else:
        # find left and right clades
        idx = [ v.index for v in nodes ].index(nd.index)
        nodes_left = nodes[:idx]
        nodes_right = nodes[(idx+1):]

        # find daughters
        nd_left = find_oldest_int_node(nodes_left)
        nd_right = find_oldest_int_node(nodes_right)

        # recurse
        nd_left = recurse(nodes_left, nd_left)
        nd_right = recurse(nodes_right, nd_right)

        # attach daughters
        nd.add_child(nd_left)
        nd.add_child(nd_right)

        # update edge lengths
        for i,ch in enumerate(nd.child_nodes()):
            ch.edge_length = ch.height - nd.height
    
    return nd

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--infile", required=True, help = "File of cblv formatted data sample.")
    parser.add_argument("--ntips", required=True, help = "Max num tips in trees.")
    parser.add_argument("--nchar", required=False, help = "Num characters.")
    args = parser.parse_args()
    num_tips = int(args.ntips)

    # read in data from file. Format format for input file see: phyddle -s F
    with h5py.File(args.infile, "r") as f:
        phy_shape = f['phy_data'].shape
        phy_data = pd.DataFrame(f['phy_data']).to_numpy()
        aux_data = pd.DataFrame(f['aux_data']).to_numpy()
        num_char = int(args.nchar) if args.nchar else 0
        # convert to cblv format
        cblvs = phy_data.reshape((phy_shape[0], phy_shape[1]//num_tips, num_tips), order = "F")
        cblv = cblvs[:, 0:(phy_shape[1]//num_tips - num_char), :]

    for i in range(cblv.shape[0]):
        nodes = get_node_heights(pd.DataFrame(cblv[i,...]), int(aux_data[i,14]))
        # print(nodes)
        heights = [ nd.height for nd in nodes ]
        idx_root = heights.index(min(heights))

        nd_root = nodes[idx_root] 
        nd_root = recurse(nodes, nd_root)
        phy_decode = dp.Tree(seed_node=nd_root)
        # output
        # print("Initial tree:")
        # print("")

        # print("CBLV+S")
        # print(cblvs[i,...])
        # print("")

        # print("Decoded tree:")
        print(phy_decode.extract_tree().as_string("newick"))
        # print("")

        # print("note: could match tip labels by modifying how encode_cblvs works")

        # phy_decode.write_to_path("test.nwk", "newick")


if __name__ == "__main__":
    main()