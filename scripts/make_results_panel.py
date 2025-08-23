# scripts/make_results_panel.py
"""
Builds a single-page summary figure and a print-pack folder:
- 3 confusion matrices (chosen subjects)
- LOSO vs Within line chart (chosen clf)
- LOSO multi-model line chart
- Summary table from reports/loso_benchmark_all.csv
Also copies the individual PNGs into reports/print_pack/.
"""

import argparse
from pathlib import Path
import shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import textwrap

DEFAULT_SUBJECTS = ["S01", "S12", "S33"]

def _exists(p: Path) -> bool:
    try: return p.is_file() and p.stat().st_size > 0
    except Exception: return False

def _short(p: Path) -> str:
    return str(p).replace("\\", "/")

def build_panel(
    outdir: Path,
    subjects: list[str],
    cm_clf: str,
    line_clf: str,
    loso_csv: Path,
    figs_dir: Path,
    print_pack: Path
):
    outdir.mkdir(parents=True, exist_ok=True)
    print_pack.mkdir(parents=True, exist_ok=True)

    # --- Locate figures we want to compose ---
    cm_paths = []
    for sid in subjects:
        # use the ensemble CM filenames we generate
        cm = figs_dir / f"cm_loso_bench_{cm_clf}_{sid}.png"
        if _exists(cm):
            cm_paths.append(cm)
        else:
            print(f"[WARN] missing CM for {sid}: {_short(cm)}")

    line_loso_within = figs_dir / f"lines_loso_vs_within_{line_clf}.png"
    if not _exists(line_loso_within):
        print(f"[WARN] missing LOSO vs Within: {_short(line_loso_within)}")

    line_models = figs_dir / "lines_loso_models.png"
    if not _exists(line_models):
        print(f"[WARN] missing LOSO multi-model: {_short(line_models)}")

    # --- Read LOSO CSV and prepare summary table ---
    df = pd.read_csv(loso_csv)
    # median macro-F1 per classifier across included subjects
    med_by_clf = (
        df[df["test_subject"].isin(subjects)]
        .groupby("clf")["f1_macro"].median().sort_values(ascending=False)
    )
    # per-subject macro-F1 for the chosen line_clf
    per_sub = (
        df[(df["clf"] == line_clf) & (df["test_subject"].isin(subjects))]
        .groupby("test_subject")["f1_macro"].median()
        .reindex(subjects)
    )

    # --- Compose panel ---
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1,1,1.2])

    # Row 1: 3 CMs (if available)
    for i, sid in enumerate(subjects[:3]):
        ax = fig.add_subplot(gs[0, i])
        if i < len(cm_paths):
            img = plt.imread(cm_paths[i])
            ax.imshow(img)
            ax.set_title(f"Confusion Matrix — {sid} ({cm_clf})", fontsize=10)
        else:
            ax.text(0.5, 0.5, f"No CM for {sid}", ha="center", va="center")
        ax.axis("off")

    # Row 2: LOSO vs Within (col 0-1), Models LOSO (col 2)
    ax1 = fig.add_subplot(gs[1, 0:2])
    if _exists(line_loso_within):
        ax1.imshow(plt.imread(line_loso_within))
        ax1.set_title(f"LOSO vs Within — {line_clf}", fontsize=10)
        ax1.axis("off")
    else:
        ax1.text(0.5, 0.5, "Missing LOSO vs Within chart", ha="center", va="center")
        ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 2])
    if _exists(line_models):
        ax2.imshow(plt.imread(line_models))
        ax2.set_title("LOSO by Model", fontsize=10)
        ax2.axis("off")
    else:
        ax2.text(0.5, 0.5, "Missing LOSO multi-model chart", ha="center", va="center")
        ax2.axis("off")

    # Row 3: Text/table summary
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis("off")
    # Build a compact report string
    lines = []
    lines.append("Median Macro-F1 by Classifier (selected subjects):")
    for name, val in med_by_clf.items():
        lines.append(f"  • {name:14s}  {val:0.3f}")
    lines.append("")
    lines.append(f"{line_clf} Macro-F1 by Subject:")
    for sid in subjects:
        v = per_sub.get(sid)
        if pd.isna(v):
            lines.append(f"  • {sid}:   n/a")
        else:
            lines.append(f"  • {sid}:   {v:0.3f}")
    text = "\n".join([textwrap.fill(s, 90) for s in lines])
    ax3.text(0.01, 0.98, "Results Summary", fontsize=12, fontweight="bold", va="top")
    ax3.text(0.01, 0.90, text, fontsize=10, va="top", family="monospace")

    fig.tight_layout()
    out_png = outdir / "results_panel.png"
    out_pdf = outdir / "results_panel.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[OK] wrote {_short(out_png)}")
    print(f"[OK] wrote {_short(out_pdf)}")

    # --- Copy source images into print pack ---
    to_copy = []
    to_copy += [p for p in cm_paths if _exists(p)]
    if _exists(line_loso_within): to_copy.append(line_loso_within)
    if _exists(line_models): to_copy.append(line_models)
    for p in to_copy:
        dst = print_pack / p.name
        try:
            shutil.copy2(p, dst)
            print(f"[COPY] {_short(p)} -> {_short(dst)}")
        except Exception as e:
            print(f"[WARN] failed copy {p}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=DEFAULT_SUBJECTS,
                    help="Which subjects to show (need their CMs).")
    ap.add_argument("--cm-clf", default="ensemble_soft",
                    help="Classifier used for the confusion matrices.")
    ap.add_argument("--line-clf", default="svm_rbf",
                    help="Classifier used for LOSO vs Within line chart.")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--figs-dir", default="reports/figures")
    ap.add_argument("--print-pack", default="reports/print_pack")
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    figs_dir = Path(args.figs_dir)
    print_pack = Path(args.print_pack)
    loso_csv = reports_dir / "loso_benchmark_all.csv"
    outdir = figs_dir

    build_panel(outdir, args.subjects, args.cm_clf, args.line_clf,
                loso_csv, figs_dir, print_pack)

if __name__ == "__main__":
    main()
