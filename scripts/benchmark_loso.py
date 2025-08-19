import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from semg_srl.models.classifiers import make_clf
from semg_srl.models.ensembles import SoftVotingEnsemble
from semg_srl.eval.metrics import extended_metrics
from semg_srl.eval.plots import save_confusion_matrix
from scipy.stats import wilcoxon

EXCLUDE_COLS = {"subject","exercise","win_start","win_len","repetition","label_id","label_name"}

def _collect_all(root: Path, subjects=None, exercises=None):
    files = sorted(root.glob("S*_E*_*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), axis=0, ignore_index=True)
    if subjects: df = df[df["subject"].isin(subjects)]
    if exercises: df = df[df["exercise"].isin(exercises)]
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
        df = df[(df.get("label_id", None) != 0) if label_col=="label_id" else (df["label_name"].astype(str).str.lower()!="rest")]

    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].values.astype(float)
    y = df[label_col].astype(str).values
    return df, X, y, feat_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", default="data/processed/db2/features")
    ap.add_argument("--exercises", nargs="*", type=int, default=None)
    ap.add_argument("--label-col", default="label_id", choices=["label_id","label_name"])
    ap.add_argument("--include-labels", nargs="*", default=None)
    ap.add_argument("--exclude-rest", action="store_true")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--no-figs", action="store_true", help="Skip saving confusion matrices (faster).") #No figures change when ready

    args = ap.parse_args()

    root = Path(args.features_root)
    outdir = Path(args.outdir); (outdir/"figures").mkdir(parents=True, exist_ok=True)

    df_all = _collect_all(root, exercises=args.exercises)
    subjects = sorted(df_all["subject"].unique().tolist())

    clf_names = ["lda", "svm_rbf", "svm_linear", "rf", "mlp", "logreg", "ensemble_soft"]
    per_subj_rows = []

    for test_sid in subjects:
        train_df = df_all[df_all["subject"] != test_sid].copy()
        test_df  = df_all[df_all["subject"] == test_sid].copy()

        train_df, Xtr, ytr, _ = _features_and_labels(train_df, args.label_col, args.include_labels, args.exclude_rest)
        test_df,  Xte, yte, _ = _features_and_labels(test_df,  args.label_col, args.include_labels, args.exclude_rest)

        # Align label spaces
        le = LabelEncoder()
        ytr_enc = le.fit_transform(ytr)
        mask = np.isin(yte, le.classes_)
        Xte, yte = Xte[mask], yte[mask]
        yte_enc = le.transform(yte) if len(yte) else np.array([], dtype=int)
        classes = list(le.classes_)
        if len(yte_enc) == 0:
            continue

        # Build models
        models = {
            "lda": make_clf("lda"),
            "svm_rbf": make_clf("svm_rbf"),
            "svm_linear": make_clf("svm_linear"),
            "rf": make_clf("rf"),
            "mlp": make_clf("mlp"),
            "logreg": make_clf("logreg"),
        }
        for n, m in models.items():
            m.fit(Xtr, ytr_enc)

        # Ensemble (prob-capable bases)
        ens_bases = [models["svm_rbf"], models["rf"], models["mlp"]]
        ens = SoftVotingEnsemble([b for b in ens_bases])  # all have predict_proba
        ens.fit(Xtr, ytr_enc)
        models["ensemble_soft"] = ens

        # Evaluate each
        for name, m in models.items():
            if hasattr(m, "predict_proba"):
                P = m.predict_proba(Xte)
                yhat = np.argmax(P, axis=1)
            else:
                P = None
                yhat = m.predict(Xte)
            mets = extended_metrics(yte_enc, yhat, y_proba=P, labels_order=list(range(len(classes))))
            per_subj_rows.append({
                "test_subject": test_sid, "clf": name,
                **mets, "n_test": len(yte_enc)
            })

            # Save CM (guarded)
            if not args.no_figs:
                from semg_srl.eval.metrics import basic_metrics
                b = basic_metrics(yte_enc, yhat, labels_order=list(range(len(classes))))
                cm_path = outdir / "figures" / f"cm_loso_bench_{name}_{test_sid}.png"
                save_confusion_matrix(b["cm"], classes, f"LOSO {test_sid} {name}", str(cm_path))


        print(f"[LOSO] {test_sid} done.")

    df = pd.DataFrame(per_subj_rows)
    out_csv = outdir / "loso_benchmark_all.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    # Rank by median macro-F1 across subjects
    pivot = df.pivot_table(index="test_subject", columns="clf", values="f1_macro", aggfunc="median")
    rank = pivot.median(axis=0).sort_values(ascending=False)
    print("\nMedian macro-F1 by classifier across subjects:\n", rank)

    # Simple significance check vs best (Wilcoxon on per-subject F1)
    best = rank.index[0]
    pvals = {}
    for name in rank.index[1:]:
        try:
            a = pivot[best].dropna()
            b = pivot[name].dropna()
            common = a.index.intersection(b.index)
            stat, p = wilcoxon(a.loc[common], b.loc[common], alternative="greater")  # best > name
            pvals[name] = p
        except Exception:
            pvals[name] = np.nan
    out_p = outdir / "loso_benchmark_wilcoxon.csv"
    pd.Series(pvals, name=f"p(best={best} > clf)").to_csv(out_p)
    print(f"[OK] Wilcoxon p-values vs best -> {out_p}")

if __name__ == "__main__":
    main()
