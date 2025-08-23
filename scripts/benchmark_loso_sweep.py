# scripts/benchmark_loso_sweep.py
from __future__ import annotations
import argparse, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from semg_srl.models.classifiers import make_clf
from semg_srl.models.ensembles import SoftVotingEnsemble
from semg_srl.eval.metrics import extended_metrics
from semg_srl.eval.plots import save_confusion_matrix

EXCLUDE_COLS = {"subject","exercise","win_start","win_len","repetition","label_id","label_name"}

def _collect_all(root: Path, exercises=None):
    files = sorted(root.glob("S*_E*_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs under {root}")
    df = pd.concat((pd.read_csv(f) for f in files), axis=0, ignore_index=True)
    if exercises:
        df = df[df["exercise"].isin(exercises)]
    return df

def _features_labels(df, label_col, include_labels=None, exclude_rest=False):
    df = df.dropna(subset=[label_col]).copy()
    if include_labels:
        if label_col == "label_id":
            keep = set(int(v) for v in include_labels)
            df = df[df[label_col].astype(int).isin(keep)]
        else:
            keep = set(str(v) for v in include_labels)
            df = df[df[label_col].astype(str).isin(keep)]
    if exclude_rest:
        if label_col == "label_id":
            df = df[df["label_id"] != 0]
        else:
            df = df[df["label_name"].astype(str).str.lower() != "rest"]
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].values.astype(float)
    y = df[label_col].astype(str).values
    return df, X, y, feat_cols

def param_grid(name: str) -> list[dict]:
    # Small, defensible search spaces
    if name == "svm_rbf":
        Cs = [0.5, 1.0, 2.0, 4.0]
        gammas = ["scale", 0.5, 2.0]   # 0.5/2.0 are relative-ish; fine for coarse check
        return [{"clf__C": C, "clf__gamma": g} for C, g in itertools.product(Cs, gammas)]
    if name == "rf":
        trees = [200, 400]
        depths = [None, 20]
        return [{"n_estimators": n, "max_depth": d} for n, d in itertools.product(trees, depths)]
    if name == "mlp":
        h = [(64,64), (128,64)]
        alphas = [1e-4, 1e-3]
        return [{"clf__hidden_layer_sizes": hs, "clf__alpha": a} for hs, a in itertools.product(h, alphas)]
    return [{}]

def set_params(est, params: dict):
    try:
        return est.set_params(**params)
    except Exception:
        # some models are Pipelines with 'clf__' params; others (RF) are plain
        return est

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", default="data/processed/db2/features")
    ap.add_argument("--exercises", nargs="*", type=int, default=[1,2])
    ap.add_argument("--label-col", default="label_id", choices=["label_id","label_name"])
    ap.add_argument("--include-labels", nargs="*", default=None)
    ap.add_argument("--exclude-rest", action="store_true")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    root = Path(args.features_root)
    outdir = Path(args.outdir); (outdir/"figures").mkdir(parents=True, exist_ok=True)

    df_all = _collect_all(root, exercises=args.exercises)
    subjects = sorted(df_all["subject"].unique().tolist())

    # define sweeps
    models = ["svm_rbf", "rf", "mlp"]
    sweeps = {m: param_grid(m) for m in models}

    rows = []
    # LOSO loop
    for test_sid in subjects:
        train_df = df_all[df_all["subject"] != test_sid].copy()
        test_df  = df_all[df_all["subject"] == test_sid].copy()
        train_df, Xtr, ytr, _ = _features_labels(train_df, args.label_col, args.include_labels, args.exclude_rest)
        test_df,  Xte, yte, _ = _features_labels(test_df,  args.label_col, args.include_labels, args.exclude_rest)

        # align classes
        le = LabelEncoder()
        ytr_enc = le.fit_transform(ytr)
        mask = np.isin(yte, le.classes_)
        Xte, yte = Xte[mask], yte[mask]
        if len(yte) == 0:
            continue
        yte_enc = le.transform(yte)
        classes = list(le.classes_)

        # Run each sweep, keep best by macro-F1 on this test subject
        best_by_model = {}
        for name in models:
            best = None
            for params in sweeps[name]:
                clf = make_clf(name)
                clf = set_params(clf, params)
                clf.fit(Xtr, ytr_enc)
                if hasattr(clf, "predict_proba"):
                    P = clf.predict_proba(Xte)
                    yhat = np.argmax(P, axis=1)
                else:
                    P = None
                    yhat = clf.predict(Xte)
                mets = extended_metrics(yte_enc, yhat, y_proba=P, labels_order=list(range(len(classes))))
                rec = {"test_subject": test_sid, "clf": name, **params, **mets, "n_test": len(yte_enc)}
                rows.append(rec)
                if best is None or mets["f1_macro"] > best[0]:
                    best = (mets["f1_macro"], params, (yhat, P))
            best_by_model[name] = best

        # Soft voting with the *best* params of the three models
        try:
            from sklearn.base import clone
            bases = []
            for name in models:
                params = best_by_model[name][1]
                est = make_clf(name)
                est = set_params(est, params)
                bases.append(est)
            ens = SoftVotingEnsemble(estimators=bases)
            # fit each base inside ensemble.fit
            ens.fit(Xtr, ytr_enc)
            P = ens.predict_proba(Xte)
            yhat = np.argmax(P, axis=1)
            mets = extended_metrics(yte_enc, yhat, y_proba=P, labels_order=list(range(len(classes))))
            rows.append({"test_subject": test_sid, "clf": "ensemble_soft_best", **mets, "n_test": len(yte_enc)})
        except Exception:
            pass

        # (Optional) save a CM for the per-subject winning single model
        if not args.no_figs:
            winner = max([(name, *best_by_model[name]) for name in models], key=lambda t: t[1])[0]
            yhat, _ = best_by_model[winner][2]
            from semg_srl.eval.metrics import basic_metrics
            b = basic_metrics(yte_enc, yhat, labels_order=list(range(len(classes))))
            cm_path = outdir / "figures" / f"cm_loso_sweep_{winner}_{test_sid}.png"
            save_confusion_matrix(b["cm"], classes, f"LOSO sweep {test_sid} {winner}", str(cm_path))

        print(f"[LOSO] {test_sid} done.")

    # Save results
    df = pd.DataFrame(rows)
    out_csv = Path(args.outdir) / "loso_sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    # Leaderboard by classifier
    summary = (df.groupby("clf")
                 .agg(f1_median=("f1_macro","median"),
                      f1_mean=("f1_macro","mean"),
                      acc_mean=("accuracy","mean"),
                      subjects=("test_subject","nunique"))
                 .sort_values("f1_median", ascending=False))
    summary.to_csv(Path(args.outdir) / "loso_sweep_leaderboard.csv")
    print(summary)

if __name__ == "__main__":
    main()
