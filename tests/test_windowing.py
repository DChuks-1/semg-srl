import numpy as np
from semg_srl.preprocess.windowing import sliding_windows

def test_sliding_windows_basic():
    fs = 2000
    x = np.zeros((fs, 4))  # 1 s, 4 channels
    w = sliding_windows(x, fs, win_ms=200, overlap=0.5)
    assert w.ndim == 3
    assert w.shape[1] == 4
    assert w.shape[0] >= 1
