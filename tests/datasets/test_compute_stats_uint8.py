#!/usr/bin/env python

# Regression tests for uint8 image statistics.
#
# `RunningQuantileStats.update` squares the batch to track the mean of squares.
# On a uint8 batch that multiplication wraps modulo 256, so the mean of squares
# can fall below the squared mean, the variance goes negative, `std` is clamped
# to 0 by `np.maximum(0, variance)`, and MEAN_STD normalization divides by eps
# (1e-8), blowing images up to the 1e7 range. These tests pin the batch to
# float64 before any squared statistics are computed.

import numpy as np

from lerobot.datasets.compute_stats import RunningQuantileStats


def _stats(batch: np.ndarray) -> dict[str, np.ndarray]:
    stats = RunningQuantileStats()
    stats.update(batch)
    return stats.get_statistics()


def test_uint8_image_batch_has_nonzero_std():
    rng = np.random.default_rng(0)
    batch = rng.integers(0, 256, size=(64, 32), dtype=np.uint8)

    stats = _stats(batch)

    # Uniform 0..255 has a population std of ~73.9; anything near zero means
    # the squares overflowed before the variance was computed. Individual
    # dimensions vary with only 64 samples, so compare the per-dimension means.
    assert np.all(stats["std"] > 50.0)
    assert abs(stats["std"].mean() - 73.9) < 3.0
    assert abs(stats["mean"].mean() - 127.5) < 3.0
    assert np.all(stats["max"] <= 255.0)


def test_uint8_normalization_stays_in_range():
    rng = np.random.default_rng(1)
    batch = rng.integers(0, 256, size=(128, 16), dtype=np.uint8)

    stats = _stats(batch)
    normalized = (batch.astype(np.float64) - stats["mean"]) / stats["std"]

    # With std == 0 the division uses eps and the result explodes.
    assert np.abs(normalized).max() < 10.0


def test_float_batch_statistics_unchanged():
    rng = np.random.default_rng(2)
    batch = rng.random((64, 32), dtype=np.float32)

    stats = _stats(batch)

    assert abs(stats["std"].mean() - 1.0 / np.sqrt(12.0)) < 0.02
    assert abs(stats["mean"].mean() - 0.5) < 0.02
