# scripts/plot_box_whiskers.py
"""
Box & whisker utilities for:
1) LOSO macro-F1 distributions by classifier (across subjects)
2) Within-subject (K-fold) vs LOSO per subject (overlay)
3) Fatigue-proxy metrics (F1 / ROC AUC) by classifier

Inputs (used if present):
- reports/loso_benchmark_all.csv
- reports/within_{SUBJECT}_{CLF}_*.csv  (e.g., within_S01_svm_rbf_label_id.csv)
- reports/fatigue_proxy/fatigue_proxy_loso_metrics.csv

Outputs:
- reports/figures/box_loso_f1_by_clf.png/.pdf
- reports/figures/box_within_vs_loso_{SUBJECT}_{clf}.png/.pdf
- reports/figures/box_fatigue_proxy_{metric}_by_clf.png/.pdf
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = Path("reports/figures")
PRINTDIR = Path("reports/print_pack")
FIGDIR.mkdir(parents=True, exist_ok=True)
PRINTDIR.mkdir(parents=True, exist_ok=True)

def _save(fig, out_png: Path, also_pdf=True):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    # copy to print_pack
    try:
        shutil.copy2(out_png, PRINTDIR / out_png.name)
        if also_pdf:
            shutil.copy2(out_png.with_suffix(".pdf"), PRINTDIR / out_png.with_suffix(".pdf").name)
    except Exception:
        pass
    print(f"[FIG] {out_png}")

# -----------------------------
# 1) LOSO: box of macro-F1 by classifier
# -----------------------------
def box_loso_by_clf(loso_csv: Path):
    if not loso_csv.exists():
        print(f"[SKIP] {loso_csv} not found.")
        return
    df = pd.read_csv(loso_csv)
    if "clf" not in df or "f1_macro" not in df or "test_subject" not in df:
        print("[SKIP] loso_benchmark_all.csv missing required columns.")
        return
    # pivot: subjects x classifiers of f1_macro
    # we want a list of f1 per clf
    clf_order = sorted(df["clf"].unique().tolist())
    data = [df.loc[df["clf"]==c, "f1_macro"].dropna().values for c in clf_order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, labels=clf_order, notch=True, patch_artist=True, showmeans=True)
    # styling
    for patch in bp['boxes']:
        patch.set_alpha(0.4)
    ax.set_ylabel("Macro-F1 (LOSO)")
    ax.set_title("LOSO Macro-F1 by Classifier (per-subject distribution)")
    ax.grid(True, axis="y", alpha=0.25)

    # annotate medians
    meds = [np.median(d) if len(d)>0 else np.nan for d in data]
    for i, m in enumerate(meds, start=1):
        if np.isfinite(m):
            ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    _save(fig, FIGDIR/"box_loso_f1_by_clf.png")

# -----------------------------
# 2) Within vs LOSO (per subject overlay)
# -----------------------------
def _find_within_csvs(subject: str, clf: str) -> list[Path]:
    # Accept any suffix after clf (label_id/name)
    pat = f"within_{subject}_{clf}_*.csv"
    return sorted(Path("reports").glob(pat))

def box_within_vs_loso(subjects: list[str], clf: str, loso_csv: Path):
    if not loso_csv.exists():
        print(f"[SKIP] {loso_csv} not found.")
        return
    loso = pd.read_csv(loso_csv)
    for sid in subjects:
        rows = loso[(loso["test_subject"]==sid) & (loso["clf"]==clf)]
        if rows.empty:
            print(f"[WARN] No LOSO row found for {sid} and clf={clf}. Skipping.")
            continue
        loso_f1 = float(rows["f1_macro"].iloc[0])

        # load within CSV(s)
        wcsvs = _find_within_csvs(sid, clf)
        if not wcsvs:
            print(f"[WARN] No within-subject CSVs found for {sid} ({clf}). Expected reports/within_{sid}_{clf}_*.csv")
            continue
        wdfs = []
        for f in wcsvs:
            try:
                d = pd.read_csv(f)
                if "f1_macro" in d.columns:
                    wdfs.append(d[["f1_macro"]].copy())
            except Exception:
                pass
        if not wdfs:
            print(f"[WARN] Could not read f1_macro from any within CSVs for {sid}.")
            continue
        win = pd.concat(wdfs, axis=0, ignore_index=True)["f1_macro"].dropna().values
        if len(win) == 0:
            print(f"[WARN] Empty within F1 list for {sid}.")
            continue

        fig, ax = plt.subplots(figsize=(5,4))
        bp = ax.boxplot([win], labels=["Within (folds)"], notch=True, patch_artist=True, showmeans=True)
        for patch in bp['boxes']:
            patch.set_alpha(0.4)
        # overlay LOSO as a marker
        ax.scatter([1.1], [loso_f1], marker="D", s=60, label=f"LOSO ({loso_f1:.2f})")
        ax.set_ylim(0.0, max(1.0, max(win.max(), loso_f1)+0.05))
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{sid}: Within vs LOSO — {clf}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
        _save(fig, FIGDIR/f"box_within_vs_loso_{sid}_{clf}.png")

# -----------------------------
# 3) Fatigue proxy: box of metric by classifier
# -----------------------------
def box_fatigue_proxy_by_clf(fp_csv: Path, metric="f1_macro"):
    if not fp_csv.exists():
        print(f"[SKIP] {fp_csv} not found.")
        return
    df = pd.read_csv(fp_csv)
    if "clf" not in df or metric not in df:
        print(f"[SKIP] fatigue_proxy csv missing columns (need clf and {metric}).")
        return
    clf_order = sorted(df["clf"].unique().tolist())
    data = [df.loc[df["clf"]==c, metric].dropna().values for c in clf_order]

    ylab = {"f1_macro":"Macro-F1 (LOSO)", "roc_auc":"ROC AUC (LOSO)"}[metric] if metric in ("f1_macro","roc_auc") else metric
    title = f"Fatigue Proxy — {ylab} by Classifier"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, labels=clf_order, notch=True, patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_alpha(0.4)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    meds = [np.median(d) if len(d)>0 else np.nan for d in data]
    for i, m in enumerate(meds, start=1):
        if np.isfinite(m):
            ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    suffix = metric.replace("_","")
    _save(fig, FIGDIR/f"box_fatigue_proxy_{suffix}_by_clf.png")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loso-csv", default="reports/loso_benchmark_all.csv")
    p.add_argument("--fatigue-csv", default="reports/fatigue_proxy/fatigue_proxy_loso_metrics.csv")
    p.add_argument("--subjects", nargs="*", default=["S01","S12","S33"])
    p.add_argument("--clf", default="svm_rbf")
    p.add_argument("--make", nargs="*", default=["loso_by_clf","within_vs_loso","fatigue_f1","fatigue_auc"],
                   help="Which plots to generate")
    args = p.parse_args()

    loso_csv = Path(args.loso_csv)
    fat_csv  = Path(args.fatigue_csv)

    if "loso_by_clf" in args.make:
        box_loso_by_clf(loso_csv)
    if "within_vs_loso" in args.make:
        box_within_vs_loso(args.subjects, args.clf, loso_csv)
    if "fatigue_f1" in args.make:
        box_fatigue_proxy_by_clf(fat_csv, metric="f1_macro")
    if "fatigue_auc" in args.make:
        box_fatigue_proxy_by_clf(fat_csv, metric="roc_auc")

if __name__ == "__main__":
    main()
