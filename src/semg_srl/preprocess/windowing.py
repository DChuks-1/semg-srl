import numpy as np

def sliding_windows(x, fs: int, win_ms: int = 200, overlap: float = 0.5):
    """
    Segment EMG array into overlapping windows.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_samples, n_channels).
    fs : int
        Sampling rate (Hz).
    win_ms : int
        Window length in milliseconds.
    overlap : float
        Fractional overlap in [0, 0.95].

    Returns
    -------
    np.ndarray
        Shape (n_windows, n_channels, win_len).
    """
    assert x.ndim == 2, "x must be (n_samples, n_channels)"
    win_len = int(fs * (win_ms / 1000))
    win_len = max(win_len, 1)
    step = int(win_len * (1 - overlap))
    step = max(step, 1)

    starts = np.arange(0, x.shape[0] - win_len + 1, step)
    windows = np.stack([x[s:s + win_len].T for s in starts], axis=0) if len(starts) else \
              np.empty((0, x.shape[1], win_len))
    return windows

def sliding_windows_with_index(x, fs: int, win_ms: int = 200, overlap: float = 0.5):
    """
    Returns:
      windows: (n_windows, n_channels, win_len)
      starts:  (n_windows,) sample indices into the original signal
    """
    assert x.ndim == 2, "x must be (n_samples, n_channels)"
    win_len = int(fs * (win_ms / 1000))
    win_len = max(win_len, 1)
    step = int(win_len * (1 - overlap))
    step = max(step, 1)
    starts = np.arange(0, x.shape[0] - win_len + 1, step, dtype=int)
    if len(starts) == 0:
        return np.empty((0, x.shape[1], win_len)), starts
    windows = np.stack([x[s:s+win_len].T for s in starts], axis=0)
    return windows, starts
