# scripts/plot_score_lines.py
import argparse
from pathlib import Path
import re, glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe
import matplotlib.pyplot as plt

def read_loso(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["test_subject", "clf", "f1_macro"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"LOSO CSV missing columns: {missing}")
    return df[need].copy()

def read_within(rep_dir: Path) -> pd.DataFrame:
    """Reads files like: reports/within_S01_<clf>_label_*.csv (our summary CSVs)."""
    rows = []
    pat = re.compile(r"within_(S\d+)_([a-z_]+)_label_.*\.csv", re.I)
    for f in glob.glob(str(rep_dir / "within_*.csv")):
        m = pat.search(Path(f).name)
        if not m: 
            continue
        sid, clf = m.group(1), m.group(2)
        try:
            dfi = pd.read_csv(f)
            if "f1_macro" in dfi.columns and len(dfi):
                rows.append({"test_subject": sid, "clf": clf, "f1_macro": float(dfi["f1_macro"].iloc[0])})
        except Exception:
            pass
    return pd.DataFrame(rows)

def subj_order(subjects):
    def key(s):
        try: return int(str(s).strip().upper().replace("S",""))
        except: return 10_000
    return sorted(list(subjects), key=key)

def plot_loso_vs_within(loso_df, within_df, outdir: Path, clf: str):
    outdir.mkdir(parents=True, exist_ok=True)
    L = loso_df[loso_df["clf"] == clf].copy()
    subs = subj_order(L["test_subject"].unique())
    x = range(len(subs))
    # LOSO values
    y_loso = []
    for s in subs:
        v = L.loc[L["test_subject"] == s, "f1_macro"]
        y_loso.append(v.median() if len(v) else None)
    # WITHIN values (only if provided)
    y_within = []
    if within_df is not None and not within_df.empty:
        W = within_df[within_df["clf"] == clf]
        for s in subs:
            v = W.loc[W["test_subject"] == s, "f1_macro"]
            y_within.append(v.median() if len(v) else None)
    else:
        y_within = [None] * len(subs)

    plt.figure()
    xs = [i for i,v in enumerate(y_loso) if pd.notna(v)]
    ys = [y_loso[i] for i in xs]
    if xs:
        plt.plot(xs, ys, marker="o", label=f"LOSO ({clf})")
    xs = [i for i,v in enumerate(y_within) if pd.notna(v)]
    ys = [y_within[i] for i in xs]
    if xs:
        plt.plot(xs, ys, marker="o", label=f"Within ({clf})")
    plt.xticks(range(len(subs)), subs, rotation=45, ha="right")
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1 per subject: LOSO vs Within")
    plt.legend()
    plt.tight_layout()
    out = outdir / f"lines_loso_vs_within_{clf}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[FIG] {out}")

def plot_loso_multi(loso_df, outdir: Path, clfs: list[str]):
    outdir.mkdir(parents=True, exist_ok=True)
    subs = subj_order(loso_df["test_subject"].unique())
    x = range(len(subs))
    plt.figure()
    for clf in clfs:
        vals = []
        for s in subs:
            v = loso_df[(loso_df["test_subject"] == s) & (loso_df["clf"] == clf)]["f1_macro"]
            vals.append(v.median() if len(v) else None)
        xs = [i for i,v in enumerate(vals) if pd.notna(v)]
        ys = [vals[i] for i in xs]
        if xs:
            plt.plot(xs, ys, marker="o", label=clf)
    plt.xticks(range(len(subs)), subs, rotation=45, ha="right")
    plt.ylabel("Macro-F1")
    plt.title("LOSO Macro-F1 per subject (models)")
    plt.legend()
    plt.tight_layout()
    out = outdir / "lines_loso_models.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[FIG] {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loso", default="reports/loso_benchmark_all.csv")
    p.add_argument("--within-dir", default="reports")
    p.add_argument("--outdir", default="reports/figures")
    p.add_argument("--clf", default="ensemble_soft")
    p.add_argument("--clfs", nargs="*", default=["ensemble_soft","rf","mlp"])
    args = p.parse_args()

    loso_df = read_loso(Path(args.loso))
    within_df = read_within(Path(args.within_dir))
    plot_loso_vs_within(loso_df, within_df, Path(args.outdir), clf=args.clf)
    plot_loso_multi(loso_df, Path(args.outdir), clfs=args.clfs)

if __name__ == "__main__":
    main()
