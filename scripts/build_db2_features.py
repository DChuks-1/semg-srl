import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from semg_srl.io.ninapro_db2 import load_db2_subject
from semg_srl.preprocess.filtering import preprocess_emg
from semg_srl.preprocess.windowing import sliding_windows_with_index
from semg_srl.features.timefreq import build_feature_matrix

def _mode_ignore_nan(a):
    a = np.asarray(a)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    vals, counts = np.unique(a.astype(int), return_counts=True)
    return vals[np.argmax(counts)]

def main():
    ap = argparse.ArgumentParser(description="Build window-level features for NinaPro DB2 subjects.")
    ap.add_argument("--subjects", nargs="*", default=None, help="e.g., S01 S02 ... (defaults to configs/datasets.yaml)")
    ap.add_argument("--exercises", nargs="*", type=int, default=None, help="e.g., 1 2")
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--outdir", type=str, default="data/processed/db2/features")
    ap.add_argument("--cfg-ds", type=str, default="configs/datasets.yaml")
    ap.add_argument("--cfg-lbl", type=str, default="configs/labels_db2.yaml")
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

            # 2) window
            windows, starts = sliding_windows_with_index(emg_f, fs=fs, win_ms=args.win_ms, overlap=args.overlap)
            if windows.shape[0] == 0:
                print(f"[WARN] No windows for {sid} E{ex} — skipping.")
                continue
            win_len = windows.shape[-1]

            # 3) features
            X, cols = build_feature_matrix(windows, fs=fs)

            # 4) window labels (majority label and repetition inside each window)
            y, y_name, y_rep = None, None, None
            if stim is not None:
                y = np.array([_mode_ignore_nan(stim[s:s+win_len]) for s in starts])
            if labels_named is not None:
                y_name = np.array([_mode_ignore_nan(labels_named[s:s+win_len]) for s in starts], dtype=object)
            if rep is not None:
                y_rep = np.array([_mode_ignore_nan(rep[s:s+win_len]) for s in starts])

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
