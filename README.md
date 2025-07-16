# PhyloEncode 0.1
![status](https://img.shields.io/badge/status-active--development-orange)

**PhyloAutoEncode** is an **MMD-VAE** that uses **PyTorch**. PhyloAutoEncode is designed for **phylogenetic data, tip-associated data, and other unstructured datasets**. It is designed to represent a training set of trees and character data as an N-dimensional multivariate standard normal distribution with minimum loss of information. It is built to work with tree files formatted by **phyddle** ([phyddle.org](https://phyddle.org)).

---

## Features
- Works with **phylogenies, tip data, and auxiliary datasets**.
- Requires **HDF5-formatted** input data.
- Provides tools for **training, encoding, and extracting feature representations** from phylogenetic data.

---

## Installation
Clone repository and from the **package root directory**, install via pip:

```bash
pip install .
```

---

## Training
To train the autoencoder, ensure your **phylogenetic and auxiliary data** are in **HDF5 format**. Use the phytrain command. There is also an example config file for network settings.

- **Example command:** `phytrain --trn_data train_data.hdf5 --config scripts/ph_config.py`  
- Type `phytrain -h` for more info

---

## Encoding with a Trained Autoencoder
To encode and decode a set of phylogenetic trees, use the **`phyencode`** and **`phydecode`** commands:

For more details on input formats and options, run:

```bash
phyencode -h
phydecode -h
```

---

## Documentation & Support
For detailed documentation of Phyddle tree formatting files, visit:  
[**phyddle.org**](https://phyddle.org/pipeline.html#format) or check the provided example scripts.

---

