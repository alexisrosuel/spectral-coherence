import numpy as np
import pytest
from spectral_coherence.utils import is_sane_time_series


@pytest.mark.parametrize(
    "x, is_valid, msg",
    [
        # x is not a numpy array
        (1, False, "x must be a numpy array"),
        # x is not a 2D array
        (np.array([1]), False, "x must be a 2D array"),
        # x contains NaNs
        (np.array([[1, 2], [3, np.nan]]), False, "x must not contain NaNs or infs"),
        # x contains infs
        (np.array([[1, 2], [3, np.inf]]), False, "x must not contain NaNs or infs"),
        # x contains complex numbers
        (np.array([[1, 2], [3, 1 + 1j]]), True, ""),
        # x contains only real numbers
        (np.array([[1, 2], [3, 4]]), True, ""),
    ],
)
def test_is_sane_time_series(x, is_valid, msg):
    assert is_sane_time_series(x) == (is_valid, msg)
