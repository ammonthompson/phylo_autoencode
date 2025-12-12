#!/usr/bin/env python3

from ete3 import Tree, TreeStyle, NodeStyle
import imageio.v2 as imageio  # Use imageio.v2 to avoid deprecation warnings
import os
import argparse

# Command line argument parser
parser = argparse.ArgumentParser(description="Create a GIF animation from Newick trees.")
parser.add_argument("-f", "--file", required=True, help="Path to the Newick file containing trees")
parser.add_argument("-o", "--output", required=True, help="Output GIF file path")
args = parser.parse_args()

# Input and output files
newick_file = args.file  # Path to your Newick file
output_gif = args.output  # Output GIF file
frames_dir = "frames"  # Temporary directory to store frames

# Create a directory to store frames
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Function to render a tree and save it as an image
def render_tree(newick_str, output_path):
    # Parse the Newick string
    tree = Tree(newick_str.strip(), format=1)  # `format=1` assumes branch lengths are present

    # Customize the tree style
    ts = TreeStyle()

    ts.mode = "r"  # Default rectangular layout (not aligned)
    ts.show_leaf_name = False  # Hide tip labels
    ts.show_branch_length = False  # Hide branch lengths
    ts.force_topology = False  # Ensure branch lengths are respected
    ts.show_branch_support = False  # Hide branch support values
    ts.branch_vertical_margin = 15  # Increase vertical spacing between branches
    ts.show_scale = False  # Remove the scale bar

    # Disable extra branch lines and guiding lines
    ts.extra_branch_line_color = None  # Disable extra branch lines
    ts.extra_branch_line_type = None  # Disable extra branch line types
    ts.guiding_lines_color = None  # Disable guiding lines
    ts.guiding_lines_type = None  # Disable guiding line types

    ns = NodeStyle()
    ns["size"] = 0
    # ns["fgcolor"] = None  # Remove any foreground color
    # ns["bgcolor"] = None  # Remove any background color
    for node in tree.traverse():
        node.set_style(ns)

    # Custom layout function to prevent adding faces to nodes
    def custom_layout(node):
        # Do not add any faces to nodes (internal or leaf)
        pass

    ts.layout_fn = custom_layout  # Apply the custom layout function

    # Dynamically adjust the scale based on the maximum branch length
    max_branch_length = max([node.dist for node in tree.traverse() if node.dist is not None])
    ts.scale = 100 / max_branch_length  # Adjust scale inversely proportional to branch length

    # Render the tree to an image file
    tree.render(output_path, tree_style=ts, w=1200, h=800, dpi=400)

# Read the Newick trees and render each one
frames = []
with open(newick_file, "r") as f:
    image_count = 0
    for i, line in enumerate(f):
        line = line.strip()  # Remove leading/trailing whitespace
        if not line:  # Skip empty lines
            continue
        image_count += 1
        frame_path = os.path.join(frames_dir, f"frame_{image_count:03d}.png")
        render_tree(line, frame_path)
        frames.append(frame_path)

# Add reverse frames for the bounce effect
gif_frames = frames.copy()
gif_frames += frames[::-1]  # Reverse the frames

# Combine frames into a GIF
with imageio.get_writer(output_gif, mode="I", fps=25, loop=0) as writer:
    for frame in gif_frames:
        image = imageio.imread(frame)
        writer.append_data(image)

# Clean up temporary frames
for frame in frames:
    os.remove(frame)
os.rmdir(frames_dir)

print(f"GIF saved as {output_gif}")