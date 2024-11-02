import numpy as np
import pytest

from spectral_coherence.density import (
    _compute_autocors,
    _get_B_spaced_freqs_mask,
    _is_B_valid,
    _periodogram,
    _select_indices,
    _smooth,
    lag_window,
    smoothed_periodogram,
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
    ],
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
    "x, n_max_freqs, B, expected_periodogram, expected_freqs, expected_mask_estimated_freqs",
    [
        # only 1 feature
        (
            [[1], [2], [3], [4], [5]],
            None,
            None,
            [
                [[1.38196601 + 0.0j]],
                [[3.61803399 + 0.0j]],
                [[45.0 + 0.0j]],
                [[3.61803399 + 0.0j]],
                [[1.38196601 + 0.0j]],
            ],
            [-0.4, -0.2, 0.0, 0.2, 0.4],
            [True, True, True, True, True],
        ),
        # 2 features
        (
            [[1, 2], [2, 3], [3, 4]],
            None,
            None,
            [
                [[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j]],
                [[12.0 + 0.0j, 18.0 + 0.0j], [18.0 + 0.0j, 27.0 + 0.0j]],
                [[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j]],
            ],
            [-0.33333333, 0.0, 0.33333333],
            [True, True, True],
        ),
        # complex case
        (
            [[1 + 1j], [2 + 2j], [3 + 3j], [4 + 5j], [5 + 5j]],
            None,
            None,
            [
                [[3.19654884 + 0.0j]],
                [[9.57983308 + 0.0j]],
                [[96.2 + 0.0j]],
                [[8.52837085 + 0.0j]],
                [[1.49524723 + 0.0j]],
            ],
            [-0.4, -0.2, 0.0, 0.2, 0.4],
            [True, True, True, True, True],
        ),
        # case with only some frequencies estimated, and B=1 so no edge frequencies to include
        (
            [[1], [2], [3], [4], [5]],
            3,
            1,
            [[[1.38196601 + 0.0j]], [[45.0 + 0.0j]], [[1.38196601 + 0.0j]]],
            [-0.4, 0.0, 0.4],
            [True, True, True],
        ),
        # case with only some frequencies estimated, and B=3 so edge frequencies to include
        (
            [[1], [2], [3], [4], [5]],
            3,
            3,
            [
                [[1.38196601 + 0.0j]],
                [[3.61803399 + 0.0j]],
                [[45.0 + 0.0j]],
                [[3.61803399 + 0.0j]],
                [[1.38196601 + 0.0j]],
            ],
            [-0.4, -0.2, 0.0, 0.2, 0.4],
            [True, False, True, False, True],
        ),
    ],
)
def test__periodogram(
    x,
    n_max_freqs,
    B,
    expected_periodogram,
    expected_freqs,
    expected_mask_estimated_freqs,
):
    x, expected_periodogram, expected_freqs, expected_mask_estimated_freqs = (
        np.array(x),
        np.array(expected_periodogram),
        np.array(expected_freqs),
        np.array(expected_mask_estimated_freqs),
    )
    periodogram, freqs, mask_estimated_freqs = _periodogram(x, n_max_freqs, B)
    assert np.allclose(periodogram, expected_periodogram)
    assert np.allclose(freqs, expected_freqs)
    assert np.allclose(mask_estimated_freqs, expected_mask_estimated_freqs)


@pytest.mark.parametrize(
    "x, B, expected_density",
    [
        (
            [[1], [2], [3], [4], [5]],
            1,
            [
                [[1.38196601 + 0.0j]],
                [[3.61803399 + 0.0j]],
                [[45.0 + 0.0j]],
                [[3.61803399 + 0.0j]],
                [[1.38196601 + 0.0j]],
            ],
        ),
    ],
)
def test_smoothed_periodogram(x, B, expected_density):
    x, expected_density = np.array(x), np.array(expected_density)
    estimated_density, _ = smoothed_periodogram(
        x, B
    )  # test only the density, not the freqs
    assert np.allclose(estimated_density, expected_density)


@pytest.mark.parametrize(
    "x, L, expected_autocors",
    [
        (
            [[1], [2], [3], [4], [5]],
            2,
            [[8.66666667], [10.0], [11.0], [10.0], [8.66666667]],
        ),
        (
            [[1, 2], [2, 3], [3, 4]],
            1,
            [[4.0, 9.0], [4.66666667, 9.66666667], [4.0, 9.0]],
        ),
    ],
)
def test__compute_autocors(x, L, expected_autocors):
    x, expected_autocors = np.array(x), np.array(expected_autocors)
    autocors = _compute_autocors(x, L)
    assert np.allclose(autocors, expected_autocors)


@pytest.mark.parametrize(
    "X, L, expected_lag_window",
    [
        (
            [[1], [2], [3], [4], [5]],
            2,
            [48.33333333],
        ),
        (
            [[1, 2], [2, 3], [3, 4]],
            1,
            [[12.66666667 + 0.0j, 27.66666667 + 0.0j]],
        ),
    ],
)
def test_lag_window(X, L, expected_lag_window):
    X, expected_lag_window = np.array(X), np.array(expected_lag_window)
    lag_window_estimator = lag_window(X, L)
    assert np.allclose(lag_window_estimator(0), expected_lag_window)


@pytest.mark.parametrize(
    "n_samples, B, expected_mask",
    [
        (5, 3, [True, False, False, True, False]),
        (5, 1, [True, True, True, True, True]),
        (5, 5, [True, False, False, False, False]),
    ],
)
def test__get_B_spaced_freqs_mask(n_samples, B, expected_mask):
    mask = _get_B_spaced_freqs_mask(n_samples, B)
    assert np.allclose(mask, expected_mask)
