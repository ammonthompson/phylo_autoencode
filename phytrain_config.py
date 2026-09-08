"""Configuration template for ``phytrain --config phytrain_config.py``."""

settings = {
    # Output and runtime
    "out_prefix": "out",
    "device": "auto",
    "seed": None,
    "testing": True,
    "num_workers": 0,
    "track_grad": False,
    "checkpoints": None,  # For example: [10, 50, 100]
    "resume_from_checkpoint": None,
    "pretrained_model": None,

    # Data; these dimensions must match the input HDF5 file.
    "num_subset": "all",
    "proportion_train": 0.85,
    "batch_size": 128,
    "max_tips": 1000,
    "num_channels": 2,
    "num_chars": 0,
    "char_type": "categorical",
    "which_aux": "all",
    "optimize_data": False,  # Set True to create/reuse <input-prefix>.phywae.hdf5.

    # Model
    "kernel": [3, 5, 9],
    "stride": [2, 4, 8],
    "out_channels": [16, 64, 128],
    "aux_inner_dim": 10,
    "latent_output_dim": None,

    # Optimization and loss
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
    "phy_loss_weight": 0.9,
    "char_loss_weight": 1.0,
    "aux_loss_weight": 0.1,
    "mmd_loss_weight": 1.0,
    "mmd_num_kernels": 3,
}
