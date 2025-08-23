# scripts/fatigue_markers_db2.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

EXCLUDE_COLS = {"subject","exercise","win_start","win_len","repetition","label_id","label_name"}

def _collect(root: Path):
    files = sorted(root.glob("S*_E*_*.csv"))
    return pd.concat((pd.read_csv(f) for f in files), axis=0, ignore_index=True)

def _filter(df, label_col, include_labels=None, exclude_rest=False, exercises=None):
    df = df.dropna(subset=[label_col]).copy()
    if exercises:
        df = df[df["exercise"].isin(exercises)]
    if include_labels:
        if label_col == "label_id":
            keep = set(int(v) for v in include_labels)
            df = df[df[label_col].astype(int).isin(keep)]
        else:
            keep = set(str(v) for v in include_labels)
            df = df[df[label_col].astype(str).isin(keep)]
    if exclude_rest:
        if label_col == "label_id":
            df = df[df["label_id"].astype(int) != 0]
        else:
            df = df[df["label_name"].astype(str).str.lower() != "rest"]
    return df

def _agg_channels(df):
    mdf_cols = [c for c in df.columns if c.startswith("MDF_ch")]
    mpf_cols = [c for c in df.columns if c.startswith("MPF_ch")]
    rms_cols = [c for c in df.columns if c.startswith("RMS_ch")]
    out = pd.DataFrame(index=df.index)
    if mdf_cols: out["mdf_mean"] = df[mdf_cols].mean(axis=1)
    if mpf_cols: out["mpf_mean"] = df[mpf_cols].mean(axis=1)
    if rms_cols: out["rms_mean"] = df[rms_cols].mean(axis=1)
    return out

def _baseline_norm(x, frac=0.10):
    n0 = max(1, int(len(x) * frac))
    base = np.median(x[:n0])
    if base == 0: base = 1.0
    return x / base

def _lin_slope(y):
    # slope vs. window index (0..N-1)
    if len(y) < 2: return np.nan
    x = np.arange(len(y), dtype=float)
    # least squares slope
    x_ = x - x.mean()
    denom = (x_**2).sum()
    if denom == 0: return np.nan
    return float((x_ * (y - y.mean())).sum() / denom)

def _per_repetition_stats(df_rep):
    # df_rep already sorted by window order
    feats = {}
    for name in ["mdf_mean","mpf_mean","rms_mean"]:
        if name not in df_rep: continue
        y = df_rep[name].to_numpy(dtype=float)
        y_norm = _baseline_norm(y, frac=0.10)
        slope = _lin_slope(y_norm)
        rho, p = spearmanr(np.arange(len(y_norm)), y_norm)
        feats[f"{name}_slope"] = slope
        feats[f"{name}_rho"] = rho
        feats[f"{name}_rho_p"] = p
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", default="data/processed/db2/features")
    ap.add_argument("--label-col", default="label_id", choices=["label_id","label_name"])
    ap.add_argument("--include-labels", nargs="*", default=None)
    ap.add_argument("--exclude-rest", action="store_true")
    ap.add_argument("--exercises", nargs="*", type=int, default=[1,2])
    ap.add_argument("--outdir", default="reports/fatigue")
    args = ap.parse_args()

    root = Path(args.features_root)
    outdir = Path(args.outdir); (outdir/"figures").mkdir(parents=True, exist_ok=True)

    df = _collect(root)
    df = _filter(df, args.label_col, args.include_labels, args.exclude_rest, args.exercises)

    # Aggregate channel features
    agg = _agg_channels(df)
    df2 = pd.concat([df[["subject","exercise","repetition","win_start",args.label_col]].reset_index(drop=True),
                     agg.reset_index(drop=True)], axis=1)

    # For plotting order
    df2 = df2.sort_values(["subject","exercise", args.label_col, "repetition", "win_start"]).reset_index(drop=True)
    label_name = args.label_col

    rows = []
    # Compute per-repetition stats
    for (sid, ex, lbl, rep), g in df2.groupby(["subject","exercise",label_name,"repetition"]):
        g = g.sort_values("win_start")
        stats = _per_repetition_stats(g)
        stats.update({"subject": sid, "exercise": ex, "label": lbl, "repetition": rep, "n": len(g)})
        rows.append(stats)
    summary = pd.DataFrame(rows)

    out_csv = outdir / "fatigue_markers_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    # Subject-wise trend plots (median over repetitions)
    for (sid, ex, lbl), g in df2.groupby(["subject","exercise",label_name]):
        g = g.sort_values(["repetition","win_start"])
        # build a per-repetition normalised time series then median across reps
        curves = {"mdf_mean": [], "mpf_mean": [], "rms_mean": []}
        for rep, r in g.groupby("repetition"):
            r = r.sort_values("win_start")
            for name in list(curves.keys()):
                if name in r:
                    y = r[name].to_numpy(dtype=float)
                    y_norm = _baseline_norm(y, frac=0.10)
                    curves[name].append(pd.Series(y_norm).reset_index(drop=True))
        # Align by min length
        for name, series_list in curves.items():
            series_list = [s.dropna() for s in series_list if len(s)>0]
            if len(series_list) == 0: continue
            L = min(len(s) for s in series_list)
            M = np.vstack([s.iloc[:L].to_numpy() for s in series_list])
            med = np.median(M, axis=0)
            lo = np.percentile(M, 25, axis=0)
            hi = np.percentile(M, 75, axis=0)

            plt.figure(figsize=(6,3))
            x = np.arange(L)
            plt.plot(x, med, label=f"{name} (median)")
            plt.fill_between(x, lo, hi, alpha=0.2, label="IQR")
            plt.axhline(1.0, ls="--", lw=1, alpha=0.7)
            plt.xlabel("Window index (within repetition)")
            pretty = { "mdf_mean":"MDF", "mpf_mean":"MPF", "rms_mean":"RMS" }[name]
            plt.ylabel(f"{pretty} (baseline-normalised)")
            plt.ylim(0, max(1.6, med.max()+0.2))
            plt.title(f"{sid} E{ex} label={lbl} — {pretty} trend")
            plt.legend(loc="best", fontsize=8)
            p = outdir/"figures"/f"trend_{sid}_E{ex}_{lbl}_{name}.png"
            plt.tight_layout()
            plt.savefig(p, dpi=220)
            plt.close()
            print(f"[FIG] {p}")

if __name__ == "__main__":
    main()
