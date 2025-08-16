import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def _apply_per_channel(x, fn):
    # x: (N, C)
    if x.ndim != 2:
        raise ValueError("EMG must be (n_samples, n_channels)")
    out = np.empty_like(x, dtype=float)
    for c in range(x.shape[1]):
        out[:, c] = fn(x[:, c].astype(float))
    return out

def bandpass(emg: np.ndarray, fs: int, low: float = 20.0, high: float = 450.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999999)
    b, a = butter(order, [low_n, high_n], btype="bandpass")
    return _apply_per_channel(emg, lambda ch: filtfilt(b, a, ch, method="gust"))

def notch50(emg: np.ndarray, fs: int, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    w0 = freq / (0.5 * fs)
    b, a = iirnotch(w0, q)
    return _apply_per_channel(emg, lambda ch: filtfilt(b, a, ch, method="gust"))

def demean(emg: np.ndarray) -> np.ndarray:
    return emg - np.mean(emg, axis=0, keepdims=True)

def preprocess_emg(emg: np.ndarray, fs: int, use_notch: bool = True) -> np.ndarray:
    """
    Pipeline: demean -> band-pass 20-450 -> (optional) 50 Hz notch.
    Returns (N, C).
    """
    x = demean(emg)
    x = bandpass(x, fs)
    if use_notch:
        x = notch50(x, fs)
    return x
