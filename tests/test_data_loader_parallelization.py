import h5py
import numpy as np

from phywae.DataProcessors import AEData


def test_dataloaders_keep_workers_alive_when_multiprocessing(tmp_path):
    data_file = tmp_path / "train.hdf5"
    with h5py.File(data_file, "w") as h5:
        h5.create_dataset("phy_data", data=np.ones((8, 8), dtype=np.float32))
        h5.create_dataset("aux_data", data=np.full((8, 1), 4, dtype=np.float32))
        h5.create_dataset("aux_data_names", data=np.asarray([[b"num_taxa"]]))

    data = AEData(
        hdf5_file=str(data_file),
        num_channels=2,
        char_data_type="continuous",
        seed=123,
        max_tips=4,
    )

    single_process_loaders = data.get_dataloaders(num_workers=0)
    persistent_loaders = data.get_dataloaders(num_workers=2)

    assert all(not loader.persistent_workers for loader in single_process_loaders)
    assert all(loader.persistent_workers for loader in persistent_loaders)
