import mlx.core as mx
import numpy as np
from spectral_coherence.density import _fourier_matrix


def test__fourier_matrix():
    N, B = 4, 3

    # Expected frequencies for freqs=None case (manually computed)
    expected_freqs = mx.array([-0.375, -0.125, 0.125, 0.375])

    # Expected As matrix (manually computed, real and imaginary parts separately)
    expected_As = np.array(
        [
            [
                [np.exp(-1j * 2 * np.pi * (-0.375) * n / N) for n in range(N)],
                [np.exp(-1j * 2 * np.pi * (-0.125) * n / N) for n in range(N)],
                [np.exp(-1j * 2 * np.pi * (0.125) * n / N) for n in range(N)],
                [np.exp(-1j * 2 * np.pi * (0.375) * n / N) for n in range(N)],
            ]
        ]
    ) / np.sqrt(N)
    expected_As = mx.array(expected_As)

    # Call the function and get results
    As, freqs = _fourier_matrix(N, B)

    # Check that output matches expected values within a tolerance
    mx.array_equal(freqs, expected_freqs)
    mx.array_equal(As, expected_As)
