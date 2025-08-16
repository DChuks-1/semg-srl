import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from semg_srl.io.ninapro_db2 import load_db2_subject
from semg_srl.preprocess.filtering import preprocess_emg
from semg_srl.preprocess.windowing import sliding_windows_with_index
from semg_srl.features.timefreq import build_feature_matrix
from collections import Counter

def _mode_ignore_nan(a):
    """
    Return the mode of a 1D slice, ignoring NaN/None.
    Works for numeric arrays and object/string arrays.
    """
    arr = np.asarray(a)
    if arr.size == 0:
        return np.nan

    # Object / string path
    if arr.dtype.kind in ("O", "U", "S"):
        vals = []
        for v in arr.ravel().tolist():
            if v is None:
                continue
            sv = str(v)
            if sv.lower() == "nan":
                continue
            vals.append(v)
        if not vals:
            return np.nan
        return Counter(vals).most_common(1)[0][0]

    # Numeric path
    arr = arr.astype(float, copy=False)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    # Keep IDs as integers if they are label codes
    vals, counts = np.unique(arr.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])

def _stable_mask_from_stimulus(stim_vec, trim_samp: int):
    """
    Returns a boolean mask (N,) where True = sample is at least trim_samp away from any label change.
    """
    if stim_vec is None:
        return None
    stim = np.asarray(stim_vec).astype(float)
    N = stim.shape[0]
    changes = np.zeros(N, dtype=bool)
    changes[1:] = stim[1:] != stim[:-1]
    change_idxs = np.flatnonzero(changes)
    stable = np.ones(N, dtype=bool)
    for idx in change_idxs:
        lo = max(0, idx - trim_samp)
        hi = min(N, idx + trim_samp)
        stable[lo:hi+1] = False
    return stable

def _rest_zscore_inplace(x, stim_vec):
    """
    Unsupervised per-channel z-scoring using only rest (stim==0) samples.
    Modifies x in-place.
    """
    if stim_vec is None:
        return
    stim = np.asarray(stim_vec).astype(float)
    rest_mask = (stim == 0)
    if not np.any(rest_mask):
        return
    mu = np.nanmean(x[rest_mask, :], axis=0, keepdims=True)
    sd = np.nanstd(x[rest_mask, :], axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    x -= mu
    x /= sd


def main():
    ap = argparse.ArgumentParser(description="Build window-level features for NinaPro DB2 subjects.")
    ap.add_argument("--subjects", nargs="*", default=None, help="e.g., S01 S02 ... (defaults to configs/datasets.yaml)")
    ap.add_argument("--exercises", nargs="*", type=int, default=None, help="e.g., 1 2")
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--outdir", type=str, default="data/processed/db2/features")
    ap.add_argument("--cfg-ds", type=str, default="configs/datasets.yaml")
    ap.add_argument("--cfg-lbl", type=str, default="configs/labels_db2.yaml")
    ap.add_argument("--trim-ms", type=int, default=100, help="Trim windows that overlap label changes by this margin.")
    ap.add_argument("--min-majority", type=float, default=0.8, help="Require this fraction of dominant label within a window.")
    ap.add_argument("--rest-zscore", action="store_true", help="Per-subject channel z-score using rest-only samples (unsupervised).")

    args = ap.parse_args()

    with open(args.cfg_ds, "r") as f:
        ds = yaml.safe_load(f)["db2"]
    subjects = args.subjects or ds["subjects"]
    exercises = args.exercises or ds.get("exercises", [1,2,3])
    fs = int(ds.get("fs_hz", 2000))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for sid in subjects:
        subj = load_db2_subject(sid, cfg_ds_path=args.cfg_ds, cfg_lbl_path=args.cfg_lbl, exercise_filter=exercises)

        for ex, pack in subj.items():
            emg = pack["emg"]           # (N, C)
            stim = pack["stimulus"]     # (N,) or None
            rep  = pack["repetition"]   # (N,) or None
            labels_named = pack["labels_named"]  # (N,) or None

            # 1) preprocess
            emg_f = preprocess_emg(emg, fs=fs)
            # optional unsupervised rest-based z-score
            if args.rest_zscore:
                _rest_zscore_inplace(emg_f, stim)


            # 2) window
            windows, starts = sliding_windows_with_index(emg_f, fs=fs, win_ms=args.win_ms, overlap=args.overlap)
            if windows.shape[0] == 0:
                print(f"[WARN] No windows for {sid} E{ex} — skipping.")
                continue
            win_len = windows.shape[-1]

            # 2b) build stable-window mask (no label changes within +/- trim_ms)
            valid_mask = np.ones(len(starts), dtype=bool)
            if stim is not None and args.trim_ms > 0:
                trim_samp = int(fs * (args.trim_ms / 1000.0))
                stable = _stable_mask_from_stimulus(stim, trim_samp)
                for i, s in enumerate(starts):
                    e = s + win_len
                    if not np.all(stable[s:e]):
                        valid_mask[i] = False

            # 3) window labels + majority check
            y = y_name = y_rep = None
            if stim is not None:
                y_slice = np.array([_mode_ignore_nan(stim[s:s+win_len]) for s in starts])
                # majority fraction (dominant code proportion in window)
                maj_frac = []
                for s in starts:
                    sl = stim[s:s+win_len]
                    vals, counts = np.unique(sl[~np.isnan(sl)], return_counts=True)
                    maj = (counts.max() / counts.sum()) if counts.size else 0.0
                    maj_frac.append(maj)
                maj_frac = np.array(maj_frac)
                if args.min_majority > 0:
                    valid_mask &= (maj_frac >= float(args.min_majority))

            if labels_named is not None:
                y_name = np.array([_mode_ignore_nan(labels_named[s:s+win_len]) for s in starts], dtype=object)
            if rep is not None:
                y_rep = np.array([_mode_ignore_nan(rep[s:s+win_len]) for s in starts])

            # 3b) apply valid mask
            windows = windows[valid_mask]
            starts = starts[valid_mask]
            if y is not None: y = y[valid_mask]
            if y_name is not None: y_name = y_name[valid_mask]
            if y_rep is not None: y_rep = y_rep[valid_mask]

            if windows.shape[0] == 0:
                print(f"[WARN] {sid} E{ex}: no valid windows after trimming/majority — skipped.")
                continue

            # 4) features
            X, cols = build_feature_matrix(windows, fs=fs)


            # 5) assemble dataframe
            df = pd.DataFrame(X, columns=cols)
            df.insert(0, "subject", sid)
            df.insert(1, "exercise", ex)
            df.insert(2, "win_start", starts)
            df.insert(3, "win_len", win_len)
            if y is not None:
                df["label_id"] = y
            if y_name is not None:
                df["label_name"] = y_name
            if y_rep is not None:
                df["repetition"] = y_rep

            # 6) save
            out_path = outdir / f"{sid}_E{ex}_win{args.win_ms}_ov{int(args.overlap*100)}.csv"
            df.to_csv(out_path, index=False)
            print(f"[OK] {sid} E{ex}: {len(df)} windows -> {out_path}")

if __name__ == "__main__":
    main()
