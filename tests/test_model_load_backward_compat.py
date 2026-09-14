from contextlib import redirect_stdout
from io import StringIO
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Test runner may not install the package; allow importing from src/.
sys.path.insert(0, str(_repo_root() / "src"))

from phywae import utils
from phywae.PhyloAEModel import AECNN


def _make_model() -> AECNN:
    phy_data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5, 1.2, 2.2, 3.2, 4.2, 1.7, 2.7, 3.7, 4.7],
            [1.1, 2.1, 3.1, 4.1, 1.6, 2.6, 3.6, 4.6, 1.3, 2.3, 3.3, 4.3, 1.8, 2.8, 3.8, 4.8],
            [1.2, 2.2, 3.2, 4.2, 1.7, 2.7, 3.7, 4.7, 1.4, 2.4, 3.4, 4.4, 1.9, 2.9, 3.9, 4.9],
            [1.3, 2.3, 3.3, 4.3, 1.8, 2.8, 3.8, 4.8, 1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    aux_data = np.array([[2.0], [3.0], [4.0], [5.0]], dtype=np.float32)

    phy_normalizer = utils.StandardScalerPhyCategorical(
        num_chars=0,
        num_chans=2,
        max_tips=8,
    ).fit(phy_data)
    aux_normalizer = StandardScaler().fit(aux_data)

    return AECNN(
        num_structured_input_channel=2,
        structured_input_width=8,
        unstructured_input_width=1,
        aux_inner_dim=2,
        aux_numtips_idx=0,
        stride=[1, 1],
        kernel=[3, 3],
        out_channels=[4, 4],
        latent_output_dim=4,
        device="cpu",
        phy_normalizer=phy_normalizer,
        aux_normalizer=aux_normalizer,
    )


class TestModelLoadBackwardCompat(unittest.TestCase):
    def test_loading_model_does_not_print_network_shapes(self):
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model.save_model(model_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                AECNN.load_pretrained_from_file(str(model_path), map_location="cpu")

        self.assertEqual(stdout.getvalue(), "")

    def test_validate_num_tips_rejects_trees_wider_than_model(self):
        model = _make_model()
        aux = np.array([[9.0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "supports at most 8"):
            model.validate_num_tips(aux)

    def test_load_pretrained_backfills_legacy_derived_buffers(self):
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model.save_model(model_path)

            artifact = torch.load(model_path, map_location="cpu", weights_only=False)
            for key in AECNN._LEGACY_DERIVED_BUFFER_KEYS:
                artifact["model_state_dict"].pop(key)
            torch.save(artifact, model_path)

            loaded_model = AECNN.load_pretrained_from_file(str(model_path), map_location="cpu")

        expected_state = model.state_dict()
        loaded_state = loaded_model.state_dict()
        self.assertEqual(set(expected_state), set(loaded_state))
        for key, expected_value in expected_state.items():
            torch.testing.assert_close(loaded_state[key], expected_value)

    def test_load_pretrained_still_raises_on_other_missing_keys(self):
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model.save_model(model_path)

            artifact = torch.load(model_path, map_location="cpu", weights_only=False)
            missing_key = next(
                key
                for key in artifact["model_state_dict"]
                if key not in AECNN._LEGACY_DERIVED_BUFFER_KEYS
            )
            artifact["model_state_dict"].pop(missing_key)
            torch.save(artifact, model_path)

            with self.assertRaisesRegex(RuntimeError, re.escape(missing_key)):
                AECNN.load_pretrained_from_file(str(model_path), map_location="cpu")


if __name__ == "__main__":
    unittest.main()
