"""PhyloEncode: phylogenetic data encoding with autoencoders.

This package contains utilities and PyTorch modules for encoding structured phylogenetic
representations (e.g. CBLV-like tensors) into a latent space using an autoencoder, and for
decoding latent vectors back into structured outputs.

Key modules:
    - ``phyloencode.DataProcessors``: Data containers and PyTorch datasets/loaders.
      See ``phyloencode.DataProcessors.AEData`` for splitting and normalization helpers.
    - ``phyloencode.PhyloAEModel``: The autoencoder network definition.
      See ``phyloencode.PhyloAEModel.AECNN``.
    - ``phyloencode.PhyloAutoencoder``: Training loop / trainer wrapper around a model, optimizer,
      and loss objects. See ``phyloencode.PhyloAutoencoder.PhyloAutoencoder``.
    - ``phyloencode.PhyLoss``: Stateful composite losses used during training.
      See ``phyloencode.PhyLoss.PhyLoss``.
    - ``phyloencode.utils``: Assorted helper functions (plotting, reshaping, etc.).

Typical workflow:
    1. Build datasets/loaders with ``phyloencode.DataProcessors.AEData``.
    2. Instantiate an autoencoder model with ``phyloencode.PhyloAEModel.AECNN``.
    3. Configure loss objects (e.g. ``phyloencode.PhyLoss.PhyLoss``) and an optimizer.
    4. Train using ``phyloencode.PhyloAutoencoder.PhyloAutoencoder``.
    5. Encode/decode with ``AECNN.encode`` / ``AECNN.decode`` (optionally using the normalization helpers).
"""

from . import DataProcessors
from . import PhyloAEModel
from . import ResNet
from . import PhyloAutoencoder
from . import utils
