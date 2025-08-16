import numpy as np
from scipy.signal import welch

def rms(sig, axis=-1): return np.sqrt(np.mean(sig**2, axis=axis))
def mav(sig, axis=-1): return np.mean(np.abs(sig), axis=axis)
def wl(sig, axis=-1):  return np.sum(np.abs(np.diff(sig, axis=axis)), axis=axis)

def zc(sig, axis=-1, eps=1e-12):
    """Zero-crossing count with small epsilon to reduce spurious flips."""
    s = np.signbit(sig + eps)
    return np.sum(np.diff(s, axis=axis), axis=axis)

def mdf_mpf(windows, fs):
    """
    Compute Median Frequency (MDF) and Mean Power Frequency (MPF)
    per window and channel.

    windows: (n_windows, n_channels, win_len)
    Returns: (mdf, mpf) each (n_windows, n_channels)
    """
    assert windows.ndim == 3, "windows must be (n_windows, n_channels, win_len)"
    n_w, n_c, _ = windows.shape
    mdf = np.zeros((n_w, n_c), dtype=float)
    mpf = np.zeros((n_w, n_c), dtype=float)

    for w in range(n_w):
        for c in range(n_c):
            f, Pxx = welch(windows[w, c], fs=fs, nperseg=min(256, windows.shape[-1]))
            if np.sum(Pxx) <= 0:
                mdf[w, c] = np.nan
                mpf[w, c] = np.nan
                continue
            cdf = np.cumsum(Pxx) / np.sum(Pxx)
            mdf[w, c] = np.interp(0.5, cdf, f)
            mpf[w, c] = np.sum(f * Pxx) / np.sum(Pxx)
    return mdf, mpf
