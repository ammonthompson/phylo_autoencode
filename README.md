# PhyloAutoEncode 0.1
### This package is under development.

**PhyloAutoEncode** is an **autoencoder** designed for **phylogenetic data, tip-associated data, and other unstructured datasets**. It is built to work with tree files formatted by **phyddle** ([phyddle.org](https://phyddle.org)).

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
To train the autoencoder, ensure your **phylogenetic and auxiliary data** are in **HDF5 format**.

- **Example script:** `scripts/train.py`  

---

## Encoding with a Trained Autoencoder
To encode a set of phylogenetic trees, use the **`phyencode`** command:

For more details on input formats and options, run:

```bash
phyencode -h
```

---

## Documentation & Support
For detailed documentation of Phyddle tree formatting files, visit:  
[**phyddle.org**](https://phyddle.org/pipeline.html#format) or check the provided example scripts.

---

