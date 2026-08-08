import sys

import h5py
import numpy as np
import pytest

from phyloencode.DataProcessors import (
    AEData,
    is_hdf5_optimized,
    optimize_hdf5,
    optimized_hdf5_path,
)
from phyloencode.cli.cmd_train import _get_default_settings, _parse_arguments


def _write_training_file(path):
    phy_data = np.arange(12 * 8, dtype=np.float32).reshape(12, 8)
    aux_data = np.column_stack((np.full(12, 4), np.arange(12))).astype(np.float32)
    with h5py.File(path, "w") as h5:
        h5.attrs["source"] = "phyddle"
        h5.create_dataset("phy_data", data=phy_data)
        h5.create_dataset("aux_data", data=aux_data, compression="gzip")
        h5.create_dataset(
            "aux_data_names", data=np.asarray([[b"num_taxa", b"row_id"]])
        )
        h5.create_dataset("labels", data=np.arange(12))
    return phy_data, aux_data


def test_nonoptimized_input_warns_without_changing_the_file(tmp_path, capsys):
    source = tmp_path / "train.hdf5"
    _write_training_file(source)

    data = AEData(
        source,
        num_channels=2,
        char_data_type="continuous",
        max_tips=4,
    )

    assert "Warning: slow HDF5 layout" in capsys.readouterr().out
    assert data.hdf5_file == str(source)
    assert not optimized_hdf5_path(source).exists()
    assert not is_hdf5_optimized(source)


def test_optimization_preserves_data_and_reuses_the_copy(tmp_path, capsys):
    source = tmp_path / "train.hdf5"
    phy_data, aux_data = _write_training_file(source)
    output = tmp_path / "train.phywae.hdf5"

    data = AEData(
        source,
        num_channels=2,
        char_data_type="continuous",
        max_tips=4,
        optimize=True,
    )

    assert "Using uncompressed training copy" in capsys.readouterr().out
    assert data.hdf5_file == str(output)
    assert is_hdf5_optimized(output)
    with h5py.File(source, "r") as original, h5py.File(output, "r+") as optimized:
        assert original["phy_data"].chunks is None
        assert optimized["phy_data"].chunks is None
        assert optimized["aux_data"].chunks is None
        assert optimized["aux_data"].compression is None
        np.testing.assert_array_equal(optimized["phy_data"][...], phy_data)
        np.testing.assert_array_equal(optimized["aux_data"][...], aux_data)
        np.testing.assert_array_equal(optimized["labels"][...], original["labels"][...])
        assert optimized.attrs["source"] == "phyddle"
        assert optimized.attrs["phyloencode_format"] == "phywae"
        assert optimized.attrs["phyloencode_format_version"] == 2
        optimized.attrs["reuse_check"] = True

    reused = AEData(
        source,
        num_channels=2,
        char_data_type="continuous",
        max_tips=4,
        optimize=True,
    )
    assert reused.hdf5_file == str(output)
    with h5py.File(output, "r") as optimized:
        assert optimized.attrs["reuse_check"]


def test_optimization_fails_fast_for_malformed_input(tmp_path):
    source = tmp_path / "broken.hdf5"
    with h5py.File(source, "w") as h5:
        h5.create_dataset("phy_data", data=np.ones((4, 8)))

    with pytest.raises(ValueError, match="aux_data"):
        optimize_hdf5(source)
    assert not optimized_hdf5_path(source).exists()


def test_optimization_refuses_to_overwrite_an_existing_file(tmp_path):
    source = tmp_path / "train.hdf5"
    _write_training_file(source)
    output = optimized_hdf5_path(source)
    _write_training_file(output)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        optimize_hdf5(source)


def test_phytrain_exposes_optimization_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["phytrain", "-d", "train.hdf5", "--optimize_data"])

    assert _parse_arguments().optimize_data is True
    assert _get_default_settings()["optimize_data"] is False
