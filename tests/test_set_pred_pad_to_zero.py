import sys
import unittest
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Test runner may not install the package; allow importing from src/.
sys.path.insert(0, str(_repo_root() / "src"))

from phywae import utils  # noqa: E402


class TestSetPredPadToZero(unittest.TestCase):
    def test_padding_is_exclusive_of_num_tips(self):
        # phy_pred: (batch, channels, max_tips)
        phy = np.zeros((2, 3, 5), dtype=np.float32)
        for tip_idx in range(phy.shape[2]):
            phy[:, :, tip_idx] = float(tip_idx + 1)

        num_tips = np.array([2.0, 4.0], dtype=np.float32)
        masked = utils.set_pred_pad_to_zero(phy, num_tips)

        # Tree 0: keep indices 0..1, zero 2..end.
        np.testing.assert_allclose(masked[0, :, :2], phy[0, :, :2])
        np.testing.assert_allclose(masked[0, :, 2:], 0.0)

        # Tree 1: keep indices 0..3, zero 4..end.
        np.testing.assert_allclose(masked[1, :, :4], phy[1, :, :4])
        np.testing.assert_allclose(masked[1, :, 4:], 0.0)

        # Ensure (bs, 1) num_tips input behaves the same.
        masked_col = utils.set_pred_pad_to_zero(phy, num_tips.reshape(-1, 1))
        np.testing.assert_allclose(masked_col, masked)


if __name__ == "__main__":
    unittest.main()
