import h5py
import numpy as np
import torch

from phywae.DataProcessors import AEData


def test_ae_data_partitions_all_selected_rows_between_train_and_validation(tmp_path):
    data_file = tmp_path / "train.hdf5"
    num_rows = 20
    max_tips = 4
    num_channels = 2

    phy_data = np.arange(
        num_rows * num_channels * max_tips, dtype=np.float32
    ).reshape(num_rows, -1)
    aux_data = np.column_stack(
        (
            np.full(num_rows, max_tips, dtype=np.float32),
            np.arange(num_rows, dtype=np.float32),
        )
    )

    with h5py.File(data_file, "w") as h5:
        h5.create_dataset("phy_data", data=phy_data)
        h5.create_dataset("aux_data", data=aux_data)
        h5.create_dataset(
            "aux_data_names",
            data=np.asarray([[b"num_taxa", b"feature"]], dtype="S64"),
        )

    data = AEData(
        hdf5_file=str(data_file),
        prop_train=0.75,
        num_channels=num_channels,
        char_data_type="continuous",
        seed=123,
        max_tips=max_tips,
    )

    selected_indices = np.concatenate(
        (data.train_dataset.indices, data.val_dataset.indices)
    )

    assert len(data.train_dataset) == 15
    assert len(data.val_dataset) == 5
    np.testing.assert_array_equal(np.sort(selected_indices), np.arange(num_rows))
    assert data.train_phy_shape == (15, num_channels * max_tips)
    assert data.val_phy_shape == (5, num_channels * max_tips)
    assert data.train_aux_shape == (15, 2)
    assert data.val_aux_shape == (5, 2)
    assert not hasattr(data, "test_phy_data")
    assert not hasattr(data, "test_aux_data")


def test_dataloader_uses_ordered_batched_dataset_reads(tmp_path):
    data_file = tmp_path / "train.hdf5"
    num_rows = 12
    with h5py.File(data_file, "w") as h5:
        h5.create_dataset(
            "phy_data",
            data=np.arange(num_rows * 8, dtype=np.float32).reshape(num_rows, 8),
        )
        h5.create_dataset(
            "aux_data",
            data=np.column_stack((np.full(num_rows, 4), np.arange(num_rows))),
        )
        h5.create_dataset(
            "aux_data_names",
            data=np.asarray([[b"num_taxa", b"row_id"]]),
        )

    data = AEData(
        hdf5_file=str(data_file),
        num_channels=2,
        char_data_type="continuous",
        seed=123,
        max_tips=4,
    )
    requested = [3, 0, 7, 1]
    requested_samples = data.train_dataset.__getitems__(requested)
    expected_aux = data.aux_normalizer.transform(np.column_stack((
        np.full(len(requested), 4),
        data.train_dataset.indices[requested],
    )))
    torch.testing.assert_close(
        torch.stack([sample[1] for sample in requested_samples]),
        torch.as_tensor(expected_aux, dtype=torch.float32),
    )

    expected_samples = data.train_dataset.__getitems__([0, 1, 2, 3])
    calls = []
    batched_read = data.train_dataset.__getitems__

    def record_batched_read(indices):
        calls.append(list(indices))
        return batched_read(indices)

    data.train_dataset.__getitems__ = record_batched_read
    train_loader, _ = data.get_dataloaders(
        batch_size=4, shuffle=False, num_workers=0
    )
    actual_batch = next(iter(train_loader))

    assert calls == [[0, 1, 2, 3]]
    for actual, expected in zip(actual_batch, zip(*expected_samples)):
        torch.testing.assert_close(actual, torch.stack(expected))
