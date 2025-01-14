# libraries
import numpy as np
import dendropy as dp
import pandas as pd

from phyddle.format import Formatter as fmt
from string import ascii_lowercase, ascii_uppercase

import sys

# make tree
s1 = "(((a:4,b:1):1,(c:2,d:1):2):1,e:3);"
s2 = "((((a:2,b:1):1,c:3):1,d:5):1,(e:2,f:1):1);"
phy_init = dp.Tree.get(data=s2, schema="newick")
taxon_names = [ nd.taxon.label for nd in phy_init.leaf_nodes() ]

# make data matrix
num_tips = len(phy_init.leaf_nodes())
num_char = 1
tree_width = 10
raw_dat = np.random.choice(a=[0,1], size=(num_char, num_tips))
dat = pd.DataFrame(raw_dat, columns=taxon_names)

# make CBLV+S
x = fmt.encode_cblvs(None, phy_init, dat, tree_width, "height_only")
x[:,0:2] = x[:,0:2] * 6
# print(x)
# print(type(x))


# subset to tips (remove buffering zeroes)
num_tips = x[:,0].tolist().index(0) 
num_char = x.shape[1] - 2
x = x[0:num_tips,:]
y = pd.DataFrame(x, index=list(range(num_tips)))
# print(y)
# print(type(y))
# quit()

#my data
num_tips = 1000
num_char = 3

true_trees = pd.read_csv(sys.argv[1], header = None, index_col=0).to_numpy()
print(true_trees.shape)
ym = true_trees[0,...].reshape((1000,7))[:,[0,1,4,5,6]]
y = pd.DataFrame(ym, index = list(range(num_tips)))
# print(y.shape)
# print(y[0:8])
# quit()

# prepare inorder node heights
nodes = [ None ] * (2*num_tips - 1)
for i in range(y.shape[0]):
   
    # node heights
    int_height = y.iloc[i,1]
    tip_height = y.iloc[i,0] + int_height

    # make tip node
    tip_nd = dp.Node(label=f't{i}')
    tip_nd.height = tip_height
    tip_nd.value = y.iloc[i,0]
    tip_nd.index = 2*i
    nodes[tip_nd.index] = tip_nd

    # do not make int node for first tip
    if i == 0:
        continue
    
    # make int node
    int_nd = dp.Node(label=f'n{i}')
    int_nd.height = int_height
    int_nd.value = y.iloc[i,1]
    int_nd.index = 2*i - 1
    nodes[int_nd.index] = int_nd

# find oldest internal node from list
def find_oldest_int_node(nodes):
    oldest = 1e6
    index = -1
    for i in range(1,len(nodes),2):
        if nodes[i].height <= oldest:
            index = i
            oldest = nodes[i].height
    return nodes[index]

# build (left,right)node relationships
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

# build tree
# idx_root = 5
# approx_root = np.min(ym[0,0])
heights = [ nd.height for nd in nodes ]
idx_root = heights.index(min(heights))
# idx_root = [ nd.height for nd in nodes ].index(0.0)
nd_root = nodes[idx_root] 
nd_root = recurse(nodes, nd_root)
phy_decode = dp.Tree(seed_node=nd_root)

# output
print("Initial tree:")
print(phy_init)
print("")

print("CBLV+S")
print(y)
print("")

print("Decoded tree:")
print(phy_decode)
print("")

print("note: could match tip labels by modifying how encode_cblvs works")