# scripts/plot_perclass_bars.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from semg_srl.models.classifiers import make_clf
from semg_srl.models.ensembles import SoftVotingEnsemble

EXCLUDE_COLS = {"subject","exercise","win_start","win_len","repetition","label_id","label_name"}

# Display names for your 5 codes
LABEL_MAP = {
    "6":  "Fist",
    "7":  "Index",
    "10": "Pronation",
    "13": "Wrist Flexion",
    "14": "Wrist Extension",
}

def _map_label(c):
    s = str(c).strip()
    try:
        i = int(float(s))
        key = str(i)
        if key in LABEL_MAP:
            return LABEL_MAP[key]
    except Exception:
        pass
    return LABEL_MAP.get(s, s)

def _collect_all(root: Path, subjects=None, exercises=None):
    files = sorted(root.glob("S*_E*_*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), axis=0, ignore_index=True)
    if subjects:
        df = df[df["subject"].isin(subjects)]
    if exercises:
        df = df[df["exercise"].isin(exercises)]
    return df

def _features_and_labels(df: pd.DataFrame, label_col: str, include_labels=None, exclude_rest=False):
    df = df.dropna(subset=[label_col]).copy()
    if include_labels:
        if label_col == "label_id":
            keep = set(int(v) for v in include_labels)
            df = df[df[label_col].astype(int).isin(keep)]
        else:
            keep = set(str(v) for v in include_labels)
            df = df[df[label_col].astype(str).isin(keep)]
    if exclude_rest:
        df = df[(df.get("label_id", None) != 0) if label_col=="label_id"
                else (df["label_name"].astype(str).str.lower()!="rest")]
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].values.astype(float)
    y = df[label_col].astype(str).values
    return df, X, y, feat_cols

def _auto_weights_from_csv(csv_path: Path):
    try:
        dfw = pd.read_csv(csv_path)
        take = dfw[dfw["clf"].isin(["svm_rbf","rf","mlp"])]
        med = take.groupby("clf")["f1_macro"].median().reindex(["svm_rbf","rf","mlp"])
        med = med.fillna(0.0)
        s = float(med.sum())
        if s <= 0: return [1.0, 1.0, 1.0]
        return (med / s).tolist()
    except Exception:
        return [1.0, 1.0, 1.0]

def _make_model(name: str, outdir: Path):
    if name != "ensemble_soft":
        return make_clf(name), None
    # ensemble: calibrated + weighted soft vote
    svm = make_clf("svm_rbf")
    try: svm.set_params(clf__probability=True)
    except Exception: pass
    rf  = make_clf("rf")
    mlp = make_clf("mlp")
    wts = _auto_weights_from_csv(outdir / "loso_benchmark_all.csv")
    ens = SoftVotingEnsemble([svm, rf, mlp], weights=wts, calibrate="sigmoid", cv=3)
    return ens, wts

def plot_perclass_for_subject(df_all, subject: str, clf_name: str,
                              label_col: str, include_labels, exclude_rest: bool,
                              outdir: Path):
    train_df = df_all[df_all["subject"] != subject].copy()
    test_df  = df_all[df_all["subject"] == subject].copy()

    _, Xtr, ytr, _ = _features_and_labels(train_df, label_col, include_labels, exclude_rest)
    _, Xte, yte, _ = _features_and_labels(test_df,  label_col, include_labels, exclude_rest)

    if len(ytr)==0 or len(yte)==0:
        print(f"[SKIP] {subject}: no data after filtering")
        return

    le = LabelEncoder()
    ytr_enc = le.fit_transform(ytr)
    mask = np.isin(yte, le.classes_)
    Xte, yte = Xte[mask], yte[mask]
    yte_enc = le.transform(yte) if len(yte) else np.array([], dtype=int)
    classes = list(le.classes_)
    if len(yte_enc)==0:
        print(f"[SKIP] {subject}: empty test after alignment")
        return

    model, wts = _make_model(clf_name, outdir)
    model.fit(Xtr, ytr_enc)
    yhat = model.predict(Xte)

    labels_order = list(range(len(classes)))
    f1s = f1_score(yte_enc, yhat, labels=labels_order, average=None, zero_division=0.0)

    names = [_map_label(c) for c in classes]
    plt.figure(figsize=(7,4))
    plt.bar(range(len(names)), f1s)
    plt.xticks(range(len(names)), names, rotation=30, ha="right")
    plt.ylabel("F1 (per class)")
    title = f"{subject} — {clf_name}"
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    outpath = outdir / f"perclass_f1_{clf_name}_{subject}.png"
    plt.savefig(outpath, dpi=220)
    plt.close()
    print(f"[FIG] {outpath}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", default="data/processed/db2/features")
    ap.add_argument("--exercises", nargs="*", type=int, default=None)
    ap.add_argument("--label-col", default="label_id", choices=["label_id","label_name"])
    ap.add_argument("--include-labels", nargs="*", default=None)
    ap.add_argument("--exclude-rest", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--clf", default="ensemble_soft",
                    choices=["lda","svm_rbf","svm_linear","rf","mlp","logreg","ensemble_soft"])
    ap.add_argument("--outdir", default="reports/figures")
    args = ap.parse_args()

    root = Path(args.features_root)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df_all = _collect_all(root, subjects=None, exercises=args.exercises)
    subs = sorted(df_all["subject"].unique().tolist())
    if args.subjects:
        subs = [s for s in subs if s in args.subjects]

    for sid in subs:
        plot_perclass_for_subject(
            df_all, sid, args.clf, args.label_col, args.include_labels, args.exclude_rest, outdir
        )

if __name__ == "__main__":
    main()
