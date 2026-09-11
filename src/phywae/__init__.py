"""PhyWAE: phylogenetic data encoding with autoencoders.

This package contains utilities and PyTorch modules for encoding structured phylogenetic
representations (e.g. CBLV-like tensors) into a latent space using an autoencoder, and for
decoding latent vectors back into structured outputs.

Key modules:
    - ``phywae.DataProcessors``: Data containers and PyTorch datasets/loaders.
      See ``phywae.DataProcessors.AEData`` for splitting and normalization helpers.
    - ``phywae.PhyloAEModel``: The autoencoder network definition.
      See ``phywae.PhyloAEModel.AECNN``.
    - ``phywae.PhyloAutoencoder``: Training loop / trainer wrapper around a model, optimizer,
      and loss objects. See ``phywae.PhyloAutoencoder.AETrainer``.
    - ``phywae.PhyLoss``: Composite objectives and component metrics used during training.
      See ``phywae.PhyLoss.PhyLoss``.
    - ``phywae.utils``: Assorted helper functions (plotting, reshaping, etc.).

Typical workflow:
    1. Build datasets/loaders with ``phywae.DataProcessors.AEData``.
    2. Instantiate an autoencoder model with ``phywae.PhyloAEModel.AECNN``.
    3. Configure loss objects (e.g. ``phywae.PhyLoss.PhyLoss``) and an optimizer.
    4. Train using ``phywae.PhyloAutoencoder.AETrainer``.
    5. Encode/decode with ``AECNN.encode`` / ``AECNN.decode`` (optionally using the normalization helpers).
"""

from . import DataProcessors
from . import PhyloAEModel
from . import ResNet
from . import PhyloAutoencoder
from .PhyloAutoencoder import AETrainer
from . import utils
