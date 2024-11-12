from typing import Optional

import mlx.core as mx
import numpy as np

from spectral_coherence.density import half_smoothed_periodograms
from spectral_coherence.utils import mx_conj, mx_einsum, mx_real


def half_coherences(
    x: mx.array, B: int, freqs: Optional[np.ndarray] = None
) -> tuple[mx.array, mx.array]:
    hSs, freqs = half_smoothed_periodograms(x, B, freqs)

    # Renormalize each half coherency matrix by its diagonal
    Ds = mx_real(mx_einsum("fkm,fkm->fm", hSs, mx_conj(hSs)))
    Ds = 1 / mx.sqrt(Ds)
    hCs = hSs * Ds[:, None, :]

    return hCs, freqs


def coherences(
    x: mx.array, B: int, freqs: Optional[np.ndarray] = None
) -> tuple[mx.array, np.ndarray]:
    """
    Compute the coherence matrix for a given signal. The coherence matrix is
    defined as the outer product of the half coherences.

    Parameters
    ----------
    x : mx.array
        The signal to compute the coherence matrix for. Shape (n_samples, n_channels).
    B : int
        The number of samples to use for the smoothing window.
    freqs : Optional[np.ndarray], optional
        The frequencies to compute the coherence matrix for. If None, the frequencies
        are computed as Fourier frequencies spaced by B samples. By default None.

    Returns
    -------
    Cs : mx.array
        The coherence matrices. Shape (n_freqs, n_channels, n_channels).
    freqs : np.ndarray
        The frequencies used to compute the coherence matrix.

    Examples
    --------
    >>> import mlx.core as mx
    >>> import numpy as np
    >>> from spectral_coherence import coherences
    >>> x = mx.array(np.random.randn(2, 1000))
    >>> Cs, freqs = coherences(x, 100)
    >>> Cs.shape
    (50, 2, 2)
    """
    hCs, freqs = half_coherences(x, B, freqs)

    # Compute the coherence matrix as the product of the
    # two half coherence matrices for each frequency
    Cs = mx_einsum("ikl,ikm->ilm", hCs, mx_conj(hCs))

    return Cs, freqs
