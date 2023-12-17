import numpy as np


def is_sane_time_series(x: np.ndarray) -> tuple[bool, str]:
    """
    Checks that the input is a valid time series. It must be a 2D array with
    at least 2 samples, all values must be numbers (real or complex), no nans,
    no infs.

    Parameters
    ----------
    x : np.ndarray
        Signal of shape (n_samples, n_features)

    Returns
    -------
    is_valid : bool
        True if the input is a valid time series
    msg : str
        Error message if the input is not a valid time series
    """
    is_valid = True
    msgs = []

    if not isinstance(x, np.ndarray):
        is_valid = False
        msgs.append("x must be a numpy array")
    elif x.ndim != 2:
        is_valid = False
        msgs.append("x must be a 2D array")
    elif not np.isfinite(x).all():
        is_valid = False
        msgs.append("x must not contain NaNs or infs")
    elif not (np.iscomplexobj(x) or np.issubdtype(x.dtype, np.number)):
        is_valid = False
        msgs.append("x must contain only numbers (real or complex)")

    msg = "\n".join(msgs)
    return is_valid, msg
