import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from semg_srl.models.classifiers import make_clf
from semg_srl.models.ensembles import SoftVotingEnsemble
from semg_srl.eval.metrics import extended_metrics
from semg_srl.eval.uncertainty import bootstrap_ci
from semg_srl.eval.plots import save_confusion_matrix

EXCLUDE_COLS = {"subject","exercise","win_start","win_len","repetition","label_id","label_name"}

def _load_subject_frames(root: Path, subject: str, exercises=None):
    pats = list(root.glob(f"{subject}_E*_*.csv"))
    dfs = []
    for p in pats:
        if exercises:
            try: ex = int(str(p.name).split("_E")[1].split("_")[0])
            except Exception: ex = None
            if ex not in exercises: continue
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
    ap.add_argument("--features-root", default="data/processed/db2/features")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--exercises", nargs="*", type=int, default=None)
    ap.add_argument("--label-col", default="label_id", choices=["label_id","label_name"])
    ap.add_argument("--include-labels", nargs="*", default=None)  # numeric codes or names
    ap.add_argument("--exclude-rest", action="store_true")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    root = Path(args.features_root)
    outdir = Path(args.outdir); (outdir/"figures").mkdir(parents=True, exist_ok=True)

    # Load & filter
    df = _load_subject_frames(root, args.subject, exercises=args.exercises)
    y = df[args.label_col]
    df = df[~y.isna()].copy()
    y = df[args.label_col]

    if args.include_labels:
        if args.label_col == "label_id":
            keep = set(int(v) for v in args.include_labels)
            df = df[df[args.label_col].astype(int).isin(keep)].copy()
        else:
            keep = set(int(v) for v in args.include_labels)
            df = df[df[args.label_col].astype(str).isin(keep)].copy()

    if args.exclude_rest:
        if args.label_col == "label_id":
            df = df[df["label_id"] != 0]
        else:
            df = df[df["label_name"].astype(str).str.lower() != "rest"]

    # X / y
    X, feat_cols = _select_features(df)
    le = LabelEncoder()
    y_enc = le.fit_transform(df[args.label_col].astype(str))
    classes = list(le.classes_)

    # CV splitter
    groups = df["repetition"] if "repetition" in df.columns else None
    if groups is not None and groups.notna().any():
        splitter = GroupKFold(n_splits=min(args.splits, len(np.unique(groups))))
        split_iter = splitter.split(X, y_enc, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=min(args.splits, np.unique(y_enc).size), shuffle=True, random_state=42)
        split_iter = splitter.split(X, y_enc)

    # Define classifier zoo
    clf_names = ["lda", "svm_rbf", "svm_linear", "rf", "mlp", "logreg", "ensemble_soft"]
    preds = {n: [] for n in clf_names}
    probas = {n: [] for n in clf_names}
    trues = []

    for fold, (tr, te) in enumerate(split_iter, 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y_enc[tr], y_enc[te]
        trues.append(yte)

        # base models
        models = {
            "lda": make_clf("lda"),
            "svm_rbf": make_clf("svm_rbf"),
            "svm_linear": make_clf("svm_linear"),
            "rf": make_clf("rf"),
            "mlp": make_clf("mlp"),
            "logreg": make_clf("logreg"),
        }

        # fit bases
        for n, m in models.items():
            m.fit(Xtr, ytr)
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(Xte)
                probas[n].append(p)
                preds[n].append(np.argmax(p, axis=1))
            else:
                yhat = m.predict(Xte)
                preds[n].append(yhat)
                probas[n].append(None)

        # soft voting from prob-capable bases
        ens_bases = [models["svm_rbf"], models["rf"], models["mlp"]]
        if all(hasattr(m, "predict_proba") for m in ens_bases):
            from semg_srl.models.ensembles import SoftVotingEnsemble
            ens = SoftVotingEnsemble(estimators=ens_bases, weights=None)
            ens.fit(Xtr, ytr)
            p = ens.predict_proba(Xte)
            probas["ensemble_soft"].append(p)
            preds["ensemble_soft"].append(np.argmax(p, axis=1))
        else:
            preds["ensemble_soft"].append(np.full_like(yte, fill_value=-1))
            probas["ensemble_soft"].append(None)

    # Concatenate folds
    y_true = np.concatenate(trues, axis=0)
    summary_rows = []
    for name in clf_names:
        y_pred = np.concatenate(preds[name], axis=0)
        P = np.concatenate(probas[name], axis=0) if probas[name][0] is not None else None

        # metrics
        mets = extended_metrics(y_true, y_pred, y_proba=P, labels_order=list(range(len(classes))))
        # bootstrap CIs (accuracy & f1)
        acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, lambda a,b: (a==b).mean())
        f1_lo, f1_hi   = bootstrap_ci(y_true, y_pred, lambda a,b: __import__("sklearn.metrics").metrics.f1_score(a,b,average="macro",zero_division=0))
        mets["acc_ci95"] = f"[{acc_lo:.3f}, {acc_hi:.3f}]"
        mets["f1_ci95"]  = f"[{f1_lo:.3f}, {f1_hi:.3f}]"

        # save CM
        from semg_srl.eval.metrics import basic_metrics
        b = basic_metrics(y_true, y_pred, labels_order=list(range(len(classes))))
        cm_path = outdir / "figures" / f"cm_within_bench_{args.subject}_{name}.png"
        save_confusion_matrix(b["cm"], classes, f"{args.subject} {name} (within-subject)", str(cm_path))

        row = {"subject": args.subject, "clf": name}
        row.update(mets)
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows).sort_values(by="f1_macro", ascending=False)
    out_csv = outdir / f"within_benchmark_{args.subject}.csv"
    df_sum.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")
    print(df_sum[["clf","accuracy","acc_ci95","f1_macro","f1_ci95","balanced_accuracy","kappa","mcc","top2_acc","top3_acc","log_loss"]])

if __name__ == "__main__":
    main()
