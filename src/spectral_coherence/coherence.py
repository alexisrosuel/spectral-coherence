from typing import Optional

import mlx.core as mx
import numpy as np

from spectral_coherence.density import half_smoothed_periodograms
from spectral_coherence.utils import mx_conj, mx_einsum, mx_real


def half_coherences(
    x: mx.array, B: int = 1, freqs: Optional[np.ndarray] = None
) -> tuple[mx.array, mx.array]:
    hSs, freqs = half_smoothed_periodograms(x, B, freqs)

    n_freqs, B, M = hSs.shape

    # Renormalize each half coherency matrix by its diagonal
    Ds = mx_real(mx_einsum("fkm,fkm->fm", hSs, mx_conj(hSs)))
    Ds = 1 / mx.sqrt(Ds)
    hCs = hSs * Ds[:, None, :]

    return hCs, freqs


def coherences(
    x: mx.array, B: int = 1, freqs: Optional[np.ndarray] = None
) -> tuple[mx.array, np.ndarray]:
    hCs, freqs = half_coherences(x, B, freqs)
    Cs = mx_einsum("ikl,ikm->ilm", hCs, mx_conj(hCs))
    return Cs, freqs
