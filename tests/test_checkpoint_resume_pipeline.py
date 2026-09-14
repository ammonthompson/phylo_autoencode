import random

import h5py
import numpy as np
import pytest
import torch
from torch.optim import AdamW

from phywae import utils
from phywae.DataProcessors import AEData
from phywae.PhyloAEModel import AECNN
from phywae.PhyloAutoencoder import PhyloAutoencoder, _LossMetricTracker
from phywae.PhyLoss import PhyLoss


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
        num_chars=0,
        char_type="continuous",
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
    }
    loss = PhyLoss(
        weights,
        data.ntax_cidx,
        model.char_type,
    )
    trainer = PhyloAutoencoder(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        loss=loss,
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
    # Simulate a checkpoint emitted by an in-flight run using the former option.
    first_segment.model.latent_layer_type = "GAUSS"
    first_segment.loss.latent_layer_type = "GAUSS"
    first_segment.train(2, seed=SEED)
    assert checkpoint_file.exists()

    checkpoint_model = AECNN.load_pretrained_from_file(
        checkpoint_file, map_location="cpu"
    )
    _assert_nested_equal(
        checkpoint_model.state_dict(), first_segment.model.state_dict()
    )
    assert not checkpoint_model.training
    assert "latent_layer_type" not in checkpoint_model.get_config_dict()
    assert "out_prefix" not in checkpoint_model.get_config_dict()
    assert not hasattr(checkpoint_model, "out_prefix")

    legacy_model_file = tmp_path / "legacy_gaussian.ae_trained.pt"
    checkpoint_model.save_model(legacy_model_file)
    legacy_artifact = torch.load(
        legacy_model_file, map_location="cpu", weights_only=False
    )
    assert "out_prefix" not in legacy_artifact["model_config"]
    legacy_artifact["model_config"]["latent_layer_type"] = "GAUSS"
    torch.save(legacy_artifact, legacy_model_file)
    legacy_model = AECNN.load_pretrained_from_file(
        legacy_model_file, map_location="cpu"
    )
    _assert_nested_equal(
        legacy_model.state_dict(), checkpoint_model.state_dict()
    )

    legacy_artifact["model_config"]["latent_layer_type"] = "DENSE"
    torch.save(legacy_artifact, legacy_model_file)
    with pytest.raises(ValueError, match="Only Gaussian latent models"):
        AECNN.load_pretrained_from_file(legacy_model_file, map_location="cpu")

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
    assert resumed.train_metrics.epoch_history == (
        uninterrupted.train_metrics.epoch_history
    )
    assert resumed.val_metrics.epoch_history == (
        uninterrupted.val_metrics.epoch_history
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


def test_loss_metric_tracker_migrates_legacy_loss_history():
    weights = {
        "phy_loss_weight": 1.0,
        "char_loss_weight": 0.0,
        "aux_loss_weight": 0.1,
        "mmd_loss_weight": 0.0,
    }
    loss = PhyLoss(weights, ntax_cidx=0)
    for name in ("total", "phy", "char", "aux", "mmd"):
        setattr(loss, f"epoch_{name}_loss", [1.0])
        setattr(loss, f"batch_{name}_loss", [torch.tensor(2.0, requires_grad=True)])
    loss.validation = True

    metrics = _LossMetricTracker()
    metrics.load_legacy_loss_history(loss)

    assert metrics.epoch_history == {
        name: [1.0] for name in ("total", "phy", "char", "aux", "mmd")
    }
    assert all(
        not values[0].requires_grad for values in metrics._batch_history.values()
    )
    assert not hasattr(loss, "epoch_total_loss")
    assert not hasattr(loss, "batch_total_loss")
    assert not hasattr(loss, "validation")


@pytest.mark.parametrize("num_kernels", [0, 2])
def test_mmd_kernel_count_must_be_positive_and_odd(num_kernels):
    weights = {
        "phy_loss_weight": 1.0,
        "char_loss_weight": 0.0,
        "aux_loss_weight": 0.1,
        "mmd_loss_weight": 1.0,
    }
    assert PhyLoss(weights, ntax_cidx=0, mmd_num_kernels=5).mmd.num_kernels == 5
    with pytest.raises(ValueError, match="positive odd integer"):
        PhyLoss(weights, ntax_cidx=0, mmd_num_kernels=num_kernels)
