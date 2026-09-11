import pytest
import torch

from phywae.PhyLoss import MMDLoss


def _squared_distances(x):
    return torch.cdist(x, x).square()


def test_small_batch_uses_every_unique_pair(monkeypatch):
    x = torch.tensor([[0.0], [1.0], [3.0], [7.0]])
    distances = _squared_distances(x)
    loss = MMDLoss(max_bw_pairs=6)

    def unexpected_randint(*args, **kwargs):
        raise AssertionError("small batches should not sample pairs")

    monkeypatch.setattr(torch, "randint", unexpected_randint)

    expected = torch.quantile(
        torch.tensor([1.0, 9.0, 49.0, 4.0, 36.0, 16.0]), 0.5
    )
    actual = loss._estimate_base_bw(x, distances)

    torch.testing.assert_close(actual, expected)


def test_large_batch_caps_sampled_pair_count(monkeypatch):
    x = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    distances = _squared_distances(x)
    loss = MMDLoss(max_bw_pairs=4)
    sampled_indices = iter((
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0, 0, 0, 0]),
    ))
    calls = []

    def fixed_randint(high, size, *, device):
        calls.append((high, size, device))
        return next(sampled_indices).to(device)

    monkeypatch.setattr(torch, "randint", fixed_randint)

    actual = loss._estimate_base_bw(x, distances)

    assert calls == [(6, (4,), x.device), (5, (4,), x.device)]
    torch.testing.assert_close(actual, torch.tensor(2.5))


def test_sampled_bandwidth_tracks_exact_median():
    generator = torch.Generator().manual_seed(4096)
    x = torch.randn(512, 8, generator=generator)
    distances = _squared_distances(x)
    exact = MMDLoss(max_bw_pairs=512 * 511 // 2)._estimate_base_bw(
        x, distances
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4096)
        sampled = MMDLoss(max_bw_pairs=4096)._estimate_base_bw(x, distances)

    torch.testing.assert_close(sampled, exact, rtol=0.05, atol=0.0)


def test_max_bw_pairs_must_be_positive():
    with pytest.raises(ValueError, match="max_bw_pairs must be positive"):
        MMDLoss(max_bw_pairs=0)
