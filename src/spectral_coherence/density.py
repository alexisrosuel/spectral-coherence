from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from spectral_coherence.utils import mx_conj, mx_einsum


def _is_B_valid(B: int, n_samples: int) -> bool:
    """
    Check if the smoothing parameter B is valid.

    Parameters
    ----------
    B : int
        Smoothing parameter
    n_samples : int
        Number of samples in the signal

    Returns
    -------
    bool
        True if the smoothing parameter is valid, False otherwise
    """
    return 0 < B < n_samples and B % 2 == 1


def _fourier_matrix(
    N: int, B: int, freqs: Optional[np.ndarray] = None
) -> Tuple[mx.array, mx.array]:
    """
    Compute the Fourier matrix for spectral analysis.

    Parameters
    ----------
    N : int
        Number of samples
    B : int
        Smoothing parameter
    freqs : Optional[np.ndarray]
        Frequency array, if None, it will be computed

    Returns
    -------
    Tuple[mx.array, mx.array]
        Fourier matrix and frequency array
    """
    if freqs is None:
        freqs = mx.arange(-N / B / 2, N / B / 2) * B / N
    else:
        freqs = mx.array(freqs)

    J = len(freqs)

    j_range = mx.arange(J)[:, mx.newaxis, mx.newaxis]
    n_range = mx.arange(N)[mx.newaxis, :, mx.newaxis]
    b_range = (mx.arange(B) - (B - 1) // 2)[mx.newaxis, mx.newaxis, :]

    exponent = -1j * 2 * np.pi * ((freqs[j_range] + b_range / N) * n_range)
    As = mx.exp(exponent) / mx.sqrt(N)

    return As, freqs


def half_smoothed_periodograms(
    x: mx.array, B: int = 1, freqs: Optional[np.ndarray] = None
) -> Tuple[mx.array, mx.array]:
    """
    Compute half-smoothed periodograms.

    Parameters
    ----------
    x : mx.array
        Input signal
    B : int, optional
        Smoothing parameter (default is 1)
    freqs : Optional[np.ndarray], optional
        Frequency array (default is None)

    Returns
    -------
    Tuple[mx.array, mx.array]
        Half-smoothed periodograms and frequency array
    """
    N, M = x.shape

    if not _is_B_valid(B, N):
        raise ValueError(f"B must be odd and between 1 and {N - 1}")

    fourier_matrix, freqs = _fourier_matrix(N, B, freqs)
    hPs = mx_einsum("ijk,jl->ikl", mx_conj(fourier_matrix), x) / mx.sqrt(B)
    return hPs, freqs


def smoothed_periodograms(
    x: mx.array, B: int = 1, freqs: Optional[np.ndarray] = None
) -> Tuple[mx.array, mx.array]:
    """
    Compute smoothed periodograms.

    Parameters
    ----------
    x : mx.array
        Input signal
    B : int, optional
        Smoothing parameter (default is 1)
    freqs : Optional[np.ndarray], optional
        Frequency array (default is None)

    Returns
    -------
    Tuple[mx.array, mx.array]
        Smoothed periodograms and frequency array
    """
    hPs, freqs = half_smoothed_periodograms(x, B, freqs)
    result = mx_einsum("ikl,ikm->ilm", hPs, mx_conj(hPs))
    return result, freqs
