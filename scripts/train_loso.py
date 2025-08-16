import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from semg_srl.models.classifiers import make_clf
from semg_srl.eval.metrics import basic_metrics
from semg_srl.eval.plots import save_confusion_matrix

EXCLUDE_COLS = {"subject","exercise","win_start","win_len","repetition","label_id","label_name"}

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
        keep = set([int(v) if label_col == "label_id" else str(v) for v in include_labels])
        df = df[df[label_col].isin(keep)].copy()
    if exclude_rest:
        if label_col == "label_name":
            df = df[df["label_name"].astype(str).str.lower() != "rest"].copy()
        else:
            df = df[df["label_id"] != 0].copy()

    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].values.astype(float)
    y = df[label_col].astype(str).values
    return df, X, y, feat_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", default="data/processed/db2/features")
    ap.add_argument("--subjects", nargs="*", default=None, help="Limit to a subset (e.g., S01 S02 ...)")
    ap.add_argument("--exercises", nargs="*", type=int, default=None)
    ap.add_argument("--label-col", default="label_name", choices=["label_name","label_id"])
    ap.add_argument("--include-labels", nargs="*", default=None)
    ap.add_argument("--exclude-rest", action="store_true")
    ap.add_argument("--clf", default="lda", choices=["lda","svm_rbf","svm_linear"])
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    root = Path(args.features_root)
    outdir = Path(args.outdir); (outdir/"figures").mkdir(parents=True, exist_ok=True)

    df_all = _collect_all(root, subjects=args.subjects, exercises=args.exercises)
    subjects = sorted(df_all["subject"].unique().tolist())

    results = []
    for test_sid in subjects:
        train_df = df_all[df_all["subject"] != test_sid].copy()
        test_df  = df_all[df_all["subject"] == test_sid].copy()

        train_df, Xtr, ytr, feat_cols = _features_and_labels(train_df, args.label_col, args.include_labels, args.exclude_rest)
        test_df,  Xte, yte, _         = _features_and_labels(test_df,  args.label_col, args.include_labels, args.exclude_rest)

        # Align label spaces
        le = LabelEncoder()
        ytr_enc = le.fit_transform(ytr)
        # Drop test rows whose labels don't exist in train
        mask = np.isin(yte, le.classes_)
        Xte, yte = Xte[mask], yte[mask]
        yte_enc = le.transform(yte) if len(yte) else np.array([], dtype=int)
        classes = list(le.classes_)

        clf = make_clf(args.clf)
        clf.fit(Xtr, ytr_enc)
        ypred = clf.predict(Xte) if len(yte_enc) else np.array([], dtype=int)

        if len(yte_enc):
            m = basic_metrics(yte_enc, ypred, labels_order=list(range(len(classes))))
            results.append({"test_subject": test_sid, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"], "n_test": len(yte_enc)})

            cm_path = outdir / "figures" / f"cm_loso_{args.clf}_{args.label_col}_{test_sid}.png"
            save_confusion_matrix(m["cm"], classes, f"LOSO {test_sid} ({args.clf})", str(cm_path))
            print(f"[FIG] {cm_path}")
        else:
            results.append({"test_subject": test_sid, "accuracy": np.nan, "f1_macro": np.nan, "n_test": 0})

        print(f"[LOSO] test={test_sid}, n={len(yte)}, clf={args.clf}")

    res = pd.DataFrame(results)
    out_csv = outdir / f"loso_{args.clf}_{args.label_col}.csv"
    res.to_csv(out_csv, index=False)
    print(f"[OK] LOSO summary -> {out_csv}")
    print(res.describe(include="all"))

if __name__ == "__main__":
    main()
