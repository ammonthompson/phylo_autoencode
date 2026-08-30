from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from phyloencode import utils
from phyloencode.PhyloAEModel import AECNN
from phyloencode.PhyloAutoencoder import AETrainer
from phyloencode.PhyLoss import PhyLoss


NUM_CHANNELS = 7
NUM_CHARS = 3
NUM_TREE_CHANNELS = NUM_CHANNELS - NUM_CHARS
WIDTH = 4


def test_positive_scaler_includes_observed_zero_values():
    values = np.array([[0.0], [2.0], [4.0]])

    scaler = utils.PositiveStandardScaler().fit(
        values, mask=np.ones_like(values, dtype=bool)
    )

    np.testing.assert_allclose(scaler.mean_, [2.0])
    np.testing.assert_allclose(scaler.std_, [2.0])


@pytest.mark.parametrize("num_chars", [1, 3])
def test_continuous_characters_use_standard_scaler(num_chars):
    num_tree_channels = 2
    num_channels = num_tree_channels + num_chars
    width = 2
    structured = np.ones((3, num_channels, width), dtype=np.float32)
    offsets = 10.0 * np.arange(num_chars)
    structured[:, num_tree_channels:, :] = (
        np.array([0.0, 2.0, 100.0])[:, None, None]
        + offsets[None, :, None]
    )
    flat = structured.reshape(3, -1, order="F")
    structured_mask = np.ones_like(structured, dtype=bool)
    structured_mask[-1] = False
    mask = structured_mask.reshape(3, -1, order="F")
    scaler = utils.StandardScalerPhyContinuous(
        num_chars, num_channels, width
    ).fit(flat, mask=mask)

    transformed = scaler.transform(flat).reshape(
        3, num_channels, width, order="F"
    )

    np.testing.assert_allclose(scaler.char_scaler.mean_, 1.0 + offsets)
    np.testing.assert_allclose(scaler.char_scaler.scale_, 1.0)
    np.testing.assert_allclose(
        transformed[:2, num_tree_channels:, :].mean(axis=0), 0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        transformed[:2, num_tree_channels:, :].std(axis=0), 1.0, atol=1e-7
    )
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(flat)), flat)


def _make_model(phy_normalizer):
    aux_normalizer = SimpleNamespace(mean_=np.array([3.0]), scale_=np.array([1.0]))
    return AECNN(
        num_structured_input_channel=NUM_CHANNELS,
        structured_input_width=WIDTH,
        unstructured_input_width=1,
        aux_numtips_idx=0,
        num_chars=NUM_CHARS,
        char_type="continuous",
        stride=[1, 1],
        kernel=[3, 3],
        out_channels=[2, 2],
        device="cpu",
        phy_normalizer=phy_normalizer,
        aux_normalizer=aux_normalizer,
    )


def test_model_constructs_bounds_from_tree_statistics():
    tree_mean = np.arange(NUM_TREE_CHANNELS * WIDTH, dtype=float).reshape(
        NUM_TREE_CHANNELS, WIDTH
    )
    tree_sd = np.full_like(tree_mean, 2.0)
    phy_normalizer = SimpleNamespace(
        tree_statistics=lambda: (tree_mean, tree_sd)
    )
    model = _make_model(phy_normalizer)

    torch.testing.assert_close(
        model.phy_two_sided_ReLU.min_val,
        torch.tensor(-tree_mean / tree_sd, dtype=torch.float32),
    )
    torch.testing.assert_close(
        model.phy_two_sided_ReLU.max_val,
        torch.tensor((1.0 - tree_mean) / tree_sd, dtype=torch.float32),
    )


def test_trainer_uses_model_schema_without_cached_proxies():
    phy_normalizer = SimpleNamespace(
        tree_statistics=lambda: (
            np.zeros((NUM_TREE_CHANNELS, WIDTH)),
            np.ones((NUM_TREE_CHANNELS, WIDTH)),
        )
    )
    model = _make_model(phy_normalizer)
    trainer = AETrainer(
        model=model,
        optimizer=AdamW(model.parameters(), lr=1e-3),
        device="cpu",
    )

    for attribute in (
        "nchars", "char_type", "phy_channels", "num_tree_chans", "best_model"
    ):
        assert not hasattr(trainer, attribute)
    for passthrough in ("predict", "tree_encode", "latent_decode", "get_latent_shape"):
        assert not hasattr(trainer, passthrough)

    phy = torch.zeros(2, NUM_CHANNELS, WIDTH)
    mask = torch.ones_like(phy, dtype=torch.bool)
    tree, char, tree_mask, char_mask = trainer._split_tree_char(phy, mask)

    assert tree.shape[1] == tree_mask.shape[1] == model.char_start_idx
    assert char.shape[1] == char_mask.shape[1] == model.num_chars


def test_continuous_character_training_minibatch():
    phy_normalizer = SimpleNamespace(
        tree_statistics=lambda: (
            np.zeros((NUM_TREE_CHANNELS, WIDTH)),
            np.ones((NUM_TREE_CHANNELS, WIDTH)),
        )
    )
    model = _make_model(phy_normalizer)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    weights = {
        "phy_loss_weight": 1.0,
        "char_loss_weight": 1.0,
        "aux_loss_weight": 0.0,
        "mmd_loss_weight": 0.0,
    }
    train_loss = PhyLoss(weights, ntax_cidx=0, char_type="continuous")
    trainer = AETrainer(
        model=model,
        optimizer=optimizer,
        train_loss=train_loss,
        device="cpu",
    )
    batch_size = 5
    phy = torch.rand(batch_size, NUM_CHANNELS, WIDTH)
    aux = torch.zeros(batch_size, 1)
    mask = torch.ones_like(phy, dtype=torch.bool)
    mask[:, :, -1] = False
    trainer.set_data_loaders(
        DataLoader(TensorDataset(phy, aux, mask), batch_size=batch_size)
    )

    trainer.train(num_epochs=2, seed=1)

    assert len(trainer.train_metrics.epoch_history["char"]) == 1
    assert np.isfinite(trainer.train_metrics.epoch_history["char"][0])
    assert isinstance(trainer.train_metrics.epoch_history["char"][0], float)
    assert not hasattr(train_loss, "epoch_char_loss_history")
