import random

import h5py
import numpy as np
import pytest
import torch
from torch.optim import AdamW

from phyloencode import utils
from phyloencode.DataProcessors import AEData
from phyloencode.PhyloAEModel import AECNN
from phyloencode.PhyloAutoencoder import PhyloAutoencoder
from phyloencode.PhyLoss import PhyLoss


SEED = 731
NUM_ROWS = 24
MAX_TIPS = 8
NUM_CHANNELS = 2
BATCH_SIZE = 6
TOTAL_EPOCH_ARG = 4


def _seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _write_small_dataset(path):
    rng = np.random.default_rng(2026)
    num_taxa = rng.integers(4, MAX_TIPS + 1, size=NUM_ROWS)
    phy = np.zeros((NUM_ROWS, NUM_CHANNELS, MAX_TIPS), dtype=np.float32)
    for row, tips in enumerate(num_taxa):
        phy[row, :, :tips] = rng.uniform(
            0.05, 0.95, size=(NUM_CHANNELS, tips)
        )

    aux = np.column_stack(
        (num_taxa.astype(np.float32), np.arange(NUM_ROWS, dtype=np.float32))
    )
    with h5py.File(path, "w") as h5:
        h5.create_dataset("phy_data", data=phy.reshape(NUM_ROWS, -1, order="F"))
        h5.create_dataset("aux_data", data=aux)
        h5.create_dataset(
            "aux_data_names",
            data=np.asarray([[b"num_taxa", b"row_id"]], dtype="S32"),
        )


def _make_data_and_loaders(data_file):
    data = AEData(
        hdf5_file=str(data_file),
        prop_train=0.75,
        num_channels=NUM_CHANNELS,
        char_data_type="continuous",
        num_chars=0,
        seed=SEED,
        max_tips=MAX_TIPS,
    )
    loaders = data.get_dataloaders(
        BATCH_SIZE, shuffle=True, num_workers=0
    )
    return data, loaders


def _make_trainer(data, loaders, checkpoint_prefix=None):
    phy_normalizer, aux_normalizer = data.get_normalizers()
    model = AECNN(
        num_structured_input_channel=NUM_CHANNELS,
        structured_input_width=MAX_TIPS,
        unstructured_input_width=data.aux_width,
        aux_inner_dim=4,
        aux_numtips_idx=data.ntax_cidx,
        aux_data_names=data.aux_colnames,
        stride=[1, 1],
        kernel=[3, 3],
        out_channels=[4, 4],
        latent_output_dim=4,
        latent_layer_type="DENSE",
        num_chars=0,
        char_type="continuous",
        out_prefix="integration-test",
        device="cpu",
        phy_normalizer=phy_normalizer,
        aux_normalizer=aux_normalizer,
    )
    optimizer = AdamW(utils.split_params_by_wd(model, 1e-3), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-3,
        epochs=TOTAL_EPOCH_ARG,
        steps_per_epoch=len(loaders[0]),
        pct_start=0.1,
        anneal_strategy="cos",
        cycle_momentum=False,
    )
    weights = {
        "phy_loss_weight": 1.0,
        "char_loss_weight": 0.0,
        "aux_loss_weight": 0.1,
        "mmd_loss_weight": 0.0,
        "vz_loss_weight": 0.0,
    }
    train_loss = PhyLoss(
        weights,
        data.ntax_cidx,
        model.char_type,
        model.latent_layer_type,
        device="cpu",
    )
    val_loss = PhyLoss(
        weights,
        data.ntax_cidx,
        model.char_type,
        model.latent_layer_type,
        device="cpu",
        validation=True,
    )
    trainer = PhyloAutoencoder(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loss=train_loss,
        val_loss=val_loss,
        seed=SEED,
        device="cpu",
        checkpoints=[1] if checkpoint_prefix is not None else None,
        checkpt_file_prefix=(
            str(checkpoint_prefix) if checkpoint_prefix is not None else "unused"
        ),
    )
    trainer.set_data_loaders(*loaders)
    return trainer


def _assert_nested_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_nested_equal(actual_value, expected_value)
    elif torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    else:
        assert actual == expected


def _loader_order(loader):
    return torch.cat([batch[1][:, 1] for batch in loader]).tolist()


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    data_file = tmp_path / "training.hdf5"
    checkpoint_prefix = tmp_path / "resume"
    checkpoint_file = tmp_path / "resume_epoch_1.ckpt.pt"
    _write_small_dataset(data_file)

    _seed_everything()
    full_data, full_loaders = _make_data_and_loaders(data_file)
    uninterrupted = _make_trainer(full_data, full_loaders)
    uninterrupted.train(TOTAL_EPOCH_ARG, seed=SEED)

    _seed_everything()
    first_data, first_loaders = _make_data_and_loaders(data_file)
    first_segment = _make_trainer(
        first_data, first_loaders, checkpoint_prefix
    )
    first_segment.train(2, seed=SEED)
    assert checkpoint_file.exists()

    resumed_data, resumed_loaders = _make_data_and_loaders(data_file)
    resumed = PhyloAutoencoder.load_checkpoint(
        checkpoint_file, map_location="cpu"
    )
    resumed.set_data_loaders(*resumed_loaders)
    resumed.train(TOTAL_EPOCH_ARG)

    assert uninterrupted.epoch == resumed.epoch == 3
    _assert_nested_equal(
        resumed.model.state_dict(), uninterrupted.model.state_dict()
    )
    _assert_nested_equal(
        resumed.optimizer.state_dict(), uninterrupted.optimizer.state_dict()
    )
    _assert_nested_equal(
        resumed.lr_sched.state_dict(), uninterrupted.lr_sched.state_dict()
    )
    for history_name in (
        "epoch_total_loss",
        "epoch_phy_loss",
        "epoch_char_loss",
        "epoch_aux_loss",
        "epoch_mmd_loss",
        "epoch_vz_loss",
    ):
        assert getattr(resumed.train_loss, history_name) == getattr(
            uninterrupted.train_loss, history_name
        )
        assert getattr(resumed.val_loss, history_name) == getattr(
            uninterrupted.val_loss, history_name
        )

    assert resumed.checkpoints == [1]
    assert resumed.checkpt_file_prefix == str(checkpoint_prefix)
    assert resumed._pending_data_loader_rng_state is None
    assert _loader_order(resumed.train_loader) == _loader_order(
        uninterrupted.train_loader
    )


def test_save_checkpoint_refuses_to_overwrite_existing_file(tmp_path):
    checkpoint_file = tmp_path / "existing.ckpt.pt"
    checkpoint_file.write_bytes(b"existing checkpoint")
    trainer = object.__new__(PhyloAutoencoder)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        trainer.save_checkpoint(checkpoint_file)

    assert checkpoint_file.read_bytes() == b"existing checkpoint"
