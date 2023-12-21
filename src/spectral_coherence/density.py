from typing import Optional

import numpy as np
from scipy.signal import fftconvolve

from spectral_coherence.utils import is_sane_time_series


def _smooth(x: np.ndarray, B: int = 1) -> np.ndarray:
    """
    Smooth an array using the Dirichlet (rectangular) window of size B.
    The computation is performed in the frequency domain for efficiency. Note that
    the convolution is circular (i.e. the first element is convolved with the last one).

    Parameters
    ----------
    x : np.ndarray
        Series of shape (n_samples, n_features, n_features)
    B : int, optional
        Smoothing parameter, by default 1

    Returns
    -------
    smoothed_x : np.ndarray
        Smoothed x of shape (n_samples, n_features, n_features)
    """
    n_samples, n_features, _ = x.shape

    # Ensure B is odd; performs the assert as per your original code.
    assert B % 2 == 1
    half_window = (B - 1) // 2

    # before doing the convolution, we need to expand the signal on the first axis
    # to force the circular convolution. It is enough to add (B-1)//2 entries on each side
    # of the signal
    if half_window != 0:
        first_entries = x[:half_window, :, :]
        last_entries = x[-half_window:, :, :]
        x = np.concatenate((last_entries, x, first_entries), axis=0)

    window = np.ones(B) / B  # Dirichlet window
    window = np.tile(window, (n_features, n_features, 1)).T

    return fftconvolve(x, window, mode="valid", axes=0)


def _select_indices(n: int, target_count=50) -> np.ndarray:
    if n <= target_count:
        return np.arange(n)  # if we have less than target_count values, return them all

    # otherwise, select the indices among the existing ones, and make the sampling uniform
    indices = np.round(np.linspace(0, n - 1, target_count)).astype(int)
    return indices


def _periodogram(
    x: np.ndarray, n_max_freqs: Optional[int] = None, B: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the periodogram of a signal at each Fourier frequency. The periodogram is computed using a Dirichlet
    (rectangular) window.

    Parameters
    ----------
    x : np.ndarray
        Signal of shape (n_samples, n_features)
    n_max_freqs : int, optional
        Maximum number of frequencies to return, by default None. If None, all Fourier frequencies are returned.
        Otherwise, only n_max_freqs uniformly spaced Fourier frequencies are returned.
    B : int, optional
        Smoothing parameter, by default None. Used only if n_max_freqs is not None. In this case we need to know
        how many fft values to keep around the selected frequencies to report.

    Returns
    -------
    periodogram : np.ndarray
        Periodogram of shape (n_samples, n_features, n_features)
    freqs : np.ndarray
        Frequencies at which the periodogram is estimated
    mask_estimated_freqs: np.ndarray
        Mask of the frequencies that can be estimated. If n_max_freqs is None, all frequencies can estimated.
        Otherwise, only n_max_freqs frequencies have the sufficient amount of fft values around them to be estimated.
    """
    # Validation on the inputs
    msg = "n_max_freqs and B should be either both None or both not None"
    assert (n_max_freqs is None and B is None) or (
        n_max_freqs is not None and B is not None
    ), msg
    if n_max_freqs is not None:
        assert n_max_freqs > 0, "n_max_freqs must be positive"

    fft = np.fft.fft(x, axis=0, norm="ortho")
    n_samples = x.shape[0]
    freqs = np.fft.fftfreq(n_samples)

    # reorder the freqs and fft in increasing order of frequency. That is useful in case we want to plot the periodogram
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fft = fft[idx, :]

    # todo: select only a subset of the frequencies. We only want maximum 10 estimations of the spectral density
    # don't forget to also include the B fft around each frequency that we want to compute the periodogram at
    if n_max_freqs is not None:
        indices = _select_indices(n_samples, n_max_freqs)
        estimated_freqs = freqs[
            indices
        ]  # these are the freqs at which the density will be estimated

        # include the B fft around each frequency that we want to compute the periodogram at
        edge_indices = np.concatenate(
            [np.arange(i - B // 2, i + B // 2 + 1) for i in indices]
        )
        edge_indices = np.mod(edge_indices, n_samples)
        edge_indices = np.sort(np.unique(edge_indices))
        fft = fft[edge_indices]
        freqs = freqs[edge_indices]
        mask_estimated_freqs = np.isin(freqs, estimated_freqs)
    else:
        mask_estimated_freqs = np.ones_like(freqs, dtype=bool)

    # fft is of shape (n_samples, n_features, 1). We want to compute the outer product of fft with its conjugate
    fft = fft[:, :, np.newaxis]
    periodogram = np.einsum("ijm,ikm->ijk", fft, np.conj(fft), optimize=True)

    return periodogram, freqs, mask_estimated_freqs


def _is_B_valid(B: int, n_samples: int) -> bool:
    """
    Check if the smoothing parameter B is valid

    Parameters
    ----------
    B : int
        Smoothing parameter
    n_samples : int
        Number of samples in the signal

    Returns
    -------
    is_valid : bool
        True if the smoothing parameter is valid, False otherwise
    """
    return B > 0 and B < n_samples and B % 2 == 1


def density(
    x: np.ndarray, B: int = 1, n_max_freqs: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes an estimation of the spectral density of a signal at each Fourier frequency.
    The spectral density is estimated by the frequency smoothed periodogram, using a Dirichlet
    (rectangular) window.

    Parameters
    ----------
    x : np.ndarray
        Signal of shape (n_samples, n_features)
    B : int, optional
        Smoothing parameter, by default 1
    n_max_freqs : int, optional
        Maximum number of frequencies to return, by default None. If None, all Fourier frequencies are returned.
        Otherwise, only n_max_freqs uniformly spaced Fourier frequencies are returned.

    Returns
    -------
    spectral_density : np.ndarray
        Spectral density of shape (n_features, n_features)
    """
    is_valid, msg = is_sane_time_series(x)
    assert is_valid, msg

    # check that the smoothing parameter B is valid
    msg = f"{B=} must be positive, smaller than the number of samples, and odd"
    assert _is_B_valid(B, x.shape[0]), msg

    # use scipy.signal.periodogram to compute the periodogram. Force the window to use
    # all the samples, and use the default scaling (density)
    prdg, freqs, mask_estimated_freqs = _periodogram(
        x, n_max_freqs, B if n_max_freqs is not None else None
    )
    smoothed_prdg = _smooth(prdg, B)

    # return only the frequencies that are allowed to be estimated (ie. those for which we kept the B fft values
    # around them)
    return smoothed_prdg[mask_estimated_freqs], freqs[mask_estimated_freqs]
