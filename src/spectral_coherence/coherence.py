import numpy as np

from spectral_coherence.density import density


def _normalize(S: np.ndarray) -> np.ndarray:
    """
    Normalize a stack of matrices by their diagonals using np.einsum.

    Parameters
    ----------
    S : np.ndarray
        Stack of matrices, where each matrix is along the last two dimensions.
        Shape is (n_samples, n_features, n_features).

    Returns
    -------
    normalized_S : np.ndarray
        Stack of normalized matrices.
    """
    # Obtain the inverse square root of the diagonal elements
    diag_inv_sqrt = 1 / np.sqrt(np.einsum('...ii->...i', S))
    
    # Apply the normalization using einsum to perform (D^-0.5 * S * D^-0.5)
    normalized_S = np.einsum('...i,...ij,...j->...ij', diag_inv_sqrt, S, diag_inv_sqrt)
    
    return normalized_S

def coherence(x: np.ndarray, B: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes an estimation of the coherence of a signal at each Fourier frequency.
    The spectral density is estimated by the frequency smoothed periodogram, using a Dirichlet
    (rectangular) window.

    Parameters
    ----------
    x : np.ndarray
        Signal of shape (n_samples, n_features)
    B : int, optional
        Smoothing parameter, by default 1

    Returns
    -------
    coherence : np.ndarray
        Coherence at each frequency of shape (n_samples, n_features, n_features)
    freqs : np.ndarray
        Frequencies at which the coherence is estimated
    """

    S_hats, freqs = density(x, B)
    C_hats = _normalize(S_hats)
    return C_hats, freqs
