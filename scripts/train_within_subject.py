import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from semg_srl.models.classifiers import make_clf
from semg_srl.eval.metrics import basic_metrics
from semg_srl.eval.plots import save_confusion_matrix

EXCLUDE_COLS = {"subject","exercise","win_start","win_len","repetition","label_id","label_name"}

def _load_subject_frames(root: Path, subject: str, exercises=None):
    pats = list((root).glob(f"{subject}_E*_*.csv"))
    dfs = []
    for p in pats:
        if exercises:
            # keep only requested exercises
            try:
                ex = int(str(p.name).split("_E")[1].split("_")[0])
            except Exception:
                ex = None
            if ex not in exercises:
                continue
        df = pd.read_csv(p)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No feature CSVs for {subject} under {root}")
    return pd.concat(dfs, axis=0, ignore_index=True)

def _select_features(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[cols].values.astype(float)
    return X, cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="e.g., S01")
    ap.add_argument("--features-root", default="data/processed/db2/features")
    ap.add_argument("--label-col", default="label_name", choices=["label_name","label_id"])
    ap.add_argument("--include-labels", nargs="*", default=None, help="names or ids to keep; others dropped")
    ap.add_argument("--exclude-rest", action="store_true")
    ap.add_argument("--exercises", nargs="*", type=int, default=None, help="e.g., 1 2")
    ap.add_argument("--clf", default="lda", choices=["lda","svm_rbf","svm_linear"])
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    root = Path(args.features_root)
    outdir = Path(args.outdir); (outdir/"figures").mkdir(parents=True, exist_ok=True)

    df = _load_subject_frames(root, args.subject, exercises=args.exercises)

    # drop rows without labels
    y = df[args.label_col]
    mask = ~y.isna()
    df = df.loc[mask].copy()
    y = df[args.label_col]

    # filter label set
    if args.include_labels:
        keep = set([int(v) if args.label_col == "label_id" else str(v) for v in args.include_labels])
        df = df[df[args.label_col].isin(keep)].copy()
        y = df[args.label_col]

    if args.exclude_rest:
        if args.label_col == "label_name":
            df = df[df["label_name"].astype(str).str.lower() != "rest"].copy()
        else:
            # if you know the rest code (usually 0), filter here
            df = df[df["label_id"] != 0].copy()
        y = df[args.label_col]

    # feature matrix
    X, feat_cols = _select_features(df)

    # label encoding to ints (keeps human-readable classes list)
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    classes = list(le.classes_)

    # groups by repetition if available; otherwise stratified kfold
    groups = df["repetition"] if "repetition" in df.columns else None

    # CV
    if groups is not None and groups.notna().any():
        splitter = GroupKFold(n_splits=min(args.splits, len(np.unique(groups))))
        split_iter = splitter.split(X, y_enc, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=min(args.splits, np.unique(y_enc).size), shuffle=True, random_state=42)
        split_iter = splitter.split(X, y_enc)

    metrics_list = []
    cm_sum = None
    for fold, (tr, te) in enumerate(split_iter, 1):
        clf = make_clf(args.clf)
        clf.fit(X[tr], y_enc[tr])
        y_pred = clf.predict(X[te])

        m = basic_metrics(y_enc[te], y_pred, labels_order=list(range(len(classes))))
        metrics_list.append({"fold": fold, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
        cm = m["cm"].astype(int)
        cm_sum = cm if cm_sum is None else (cm_sum + cm)

    # aggregate
    res = pd.DataFrame(metrics_list)
    res_path = outdir / f"within_{args.subject}_{args.clf}_{args.label_col}.csv"
    res.to_csv(res_path, index=False)

    # save CM
    cm_path = outdir / "figures" / f"cm_within_{args.subject}_{args.clf}_{args.label_col}.png"
    title = f"Within-subject {args.subject} ({args.clf})"
    save_confusion_matrix(cm_sum, classes, title, str(cm_path))

    print(f"[OK] {args.subject}: acc mean={res['accuracy'].mean():.3f} ± {res['accuracy'].std():.3f}, "
          f"f1_macro mean={res['f1_macro'].mean():.3f} -> {res_path}")
    print(f"[FIG] {cm_path}")

if __name__ == "__main__":
    main()
