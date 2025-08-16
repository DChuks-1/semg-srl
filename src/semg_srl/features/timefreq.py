import numpy as np
from scipy.signal import welch

def rms(sig, axis=-1): return np.sqrt(np.mean(sig**2, axis=axis))
def mav(sig, axis=-1): return np.mean(np.abs(sig), axis=axis)
def wl(sig, axis=-1):  return np.sum(np.abs(np.diff(sig, axis=axis)), axis=axis)

def zc(sig, axis=-1, eps=1e-12):
    s = np.signbit(sig + eps)
    return np.sum(np.diff(s, axis=axis), axis=axis)

def ssc(sig, delta=1e-6):
    """Slope Sign Changes along last axis."""
    x0 = sig[..., :-2]
    x1 = sig[..., 1:-1]
    x2 = sig[..., 2:]
    cond = ((x1 - x0) * (x1 - x2)) > delta
    return np.sum(cond, axis=-1)

def mdf_mpf(windows, fs):
    assert windows.ndim == 3, "windows must be (n_windows, n_channels, win_len)"
    n_w, n_c, _ = windows.shape
    mdf = np.zeros((n_w, n_c), dtype=float)
    mpf = np.zeros((n_w, n_c), dtype=float)
    for w in range(n_w):
        for c in range(n_c):
            f, Pxx = welch(windows[w, c], fs=fs, nperseg=min(256, windows.shape[-1]))
            ps = np.sum(Pxx)
            if ps <= 0:
                mdf[w, c] = np.nan; mpf[w, c] = np.nan; continue
            cdf = np.cumsum(Pxx) / ps
            mdf[w, c] = np.interp(0.5, cdf, f)
            mpf[w, c] = np.sum(f * Pxx) / ps
    return mdf, mpf

def build_feature_matrix(windows, fs, channel_axis=1):
    """
    windows: (n_windows, n_channels, win_len)
    Returns (X, cols)
      X: (n_windows, n_channels * n_features)
      cols: list of column names
    """
    # time-domain
    feats = {
        "RMS": rms(windows, axis=-1),
        "MAV": mav(windows, axis=-1),
        "WL":  wl(windows, axis=-1),
        "ZC":  zc(windows, axis=-1),
        "SSC": ssc(windows),
    }
    # freq-domain
    mdf, mpf = mdf_mpf(windows, fs)
    feats["MDF"] = mdf
    feats["MPF"] = mpf

    # flatten channel-wise
    cols, blocks = [], []
    for name, arr in feats.items():
        for ch in range(arr.shape[1]):
            cols.append(f"{name}_ch{ch+1}")
        blocks.append(arr)
    X = np.concatenate(blocks, axis=1)  # (n_windows, n_channels * n_features)
    return X, cols
