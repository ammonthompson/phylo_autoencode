from types import SimpleNamespace
import warnings

import numpy as np

from phyloencode.utils import make_loss_plots


def test_loss_plots_handle_zero_components_without_runtime_warning(tmp_path):
    epochs = 12

    def loss_history():
        return SimpleNamespace(
            epoch_history={
                "total": np.linspace(2.0, 1.0, epochs).tolist(),
                "phy": np.linspace(1.0, 0.5, epochs).tolist(),
                "char": [0.0] * epochs,
                "aux": [0.0] * epochs,
                "mmd": [0.0] * epochs,
            }
        )

    out_prefix = str(tmp_path / "losses")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        make_loss_plots(
            loss_history(),
            loss_history(),
            out_prefix=out_prefix,
            starting_epoch=0,
        )

    assert (tmp_path / "losses.component_loss.pdf").is_file()
    assert not (tmp_path / "losses.loss.pdf").exists()
