import numpy as np
import pytest
from spectral_coherence.density import (
    _is_B_valid,
    _periodogram,
    _select_indices,
    _smooth,
    density,
)


@pytest.mark.parametrize(
    "B, n_samples, is_valid",
    [
        # B is negative
        (-1, 5, False),
        # B is larger than the number of samples
        (6, 5, False),
        # B is even
        (2, 5, False),
        # ok case
        (3, 5, True),
    ],
)
def test__is_B_valid(B, n_samples, is_valid):
    assert _is_B_valid(B, n_samples) == is_valid


@pytest.mark.parametrize(
    "n, target_count, expected_indices",
    [
        (1, 1, [0]),
        (1, 2, [0]),
        (2, 1, [0]),
        (10, 3, [0, 4, 9]),
    ]
)
def test__select_indices(n, target_count, expected_indices):
    indices = _select_indices(n, target_count)
    assert np.allclose(indices, expected_indices)


@pytest.mark.parametrize(
    ("x, B, expected_smoothed_x"),
    [
        # real cases with increasing B
        (
            [[[1]], [[2]], [[3]], [[4]], [[5]]],
            1,
            [[[1]], [[2]], [[3]], [[4]], [[5]]],
        ),
        (
            [[[1]], [[2]], [[3]], [[4]], [[5]]],
            3,
            [[[8 / 3]], [[6 / 3]], [[9 / 3]], [[12 / 3]], [[10 / 3]]],
        ),
        (
            [[[1]], [[2]], [[3]], [[4]], [[5]]],
            5,
            [[[15 / 5]], [[15 / 5]], [[15 / 5]], [[15 / 5]], [[15 / 5]]],
        ),
        # complex cases with increasing B
        (
            [[[1 + 1j]], [[2 + 2j]], [[3 + 3j]], [[4 + 4j]], [[5 + 5j]]],
            1,
            [[[1 + 1j]], [[2 + 2j]], [[3 + 3j]], [[4 + 4j]], [[5 + 5j]]],
        ),
        (
            [[[1 + 1j]], [[2 + 2j]], [[3 + 3j]], [[4 + 4j]], [[5 + 5j]]],
            3,
            [
                [[2.66666667 + 2.66666667j]],
                [[2.0 + 2.0j]],
                [[3.0 + 3.0j]],
                [[4.0 + 4.0j]],
                [[3.33333333 + 3.33333333j]],
            ],
        ),
    ],
)
def test__smooth(x, B, expected_smoothed_x):
    x, expected_smoothed_x = np.array(x), np.array(expected_smoothed_x)
    smoothed_x = _smooth(x, B)
    assert np.allclose(smoothed_x, expected_smoothed_x, atol=1e-5)


@pytest.mark.parametrize(
    "x, expected_periodogram, expected_freqs",
    [
        # only 1 feature
        (
            [[1], [2], [3], [4], [5]],
            [
                [[45.0 + 0.0j]],
                [[3.61803399 + 0.0j]],
                [[1.38196601 + 0.0j]],
                [[1.38196601 + 0.0j]],
                [[3.61803399 + 0.0j]],
            ],
            [0.0, 0.2, 0.4, -0.4, -0.2],
        ),
        # 2 features
        (
            [[1, 2], [2, 3], [3, 4]],
            [
                [[12.0 + 0.0j, 18.0 + 0.0j], [18.0 + 0.0j, 27.0 + 0.0j]],
                [[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j]],
                [[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j]],
            ],
            [0.0, 0.33333333, -0.33333333],
        ),
        # complex case
        (
            [[1 + 1j], [2 + 2j], [3 + 3j], [4 + 5j], [5 + 5j]],
            [
                [[96.2 + 0.0j]],
                [[8.52837085 + 0.0j]],
                [[1.49524723 + 0.0j]],
                [[3.19654884 + 0.0j]],
                [[9.57983308 + 0.0j]],
            ],
            [0.0, 0.2, 0.4, -0.4, -0.2],
        ),
    ],
)
def test__periodogram(x, expected_periodogram, expected_freqs):
    x, expected_periodogram, expected_freqs = (
        np.array(x),
        np.array(expected_periodogram),
        np.array(expected_freqs),
    )
    periodogram, freqs = _periodogram(x)
    assert np.allclose(periodogram, expected_periodogram)
    assert np.allclose(freqs, expected_freqs)


@pytest.mark.parametrize(
    "x, B, expected_density",
    [
        (
            [[1], [2], [3], [4], [5]],
            1,
            [
                [[45.0 + 0.0j]],
                [[3.61803399 + 0.0j]],
                [[1.38196601 + 0.0j]],
                [[1.38196601 + 0.0j]],
                [[3.61803399 + 0.0j]],
            ],
        ),
    ],
)
def test_density(x, B, expected_density):
    x, expected_density = np.array(x), np.array(expected_density)
    estimated_density, _ = density(x, B)  # test only the density, not the freqs
    assert np.allclose(estimated_density, expected_density)
