# scripts/fatigue_proxy_classifier.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

# Reuse your plotting util if present
try:
    from semg_srl.eval.plots import save_confusion_matrix
except Exception:
    save_confusion_matrix = None

EXCLUDE_COLS = {
    "subject","exercise","win_start","win_len","repetition","label_id","label_name"
}

def _collect(root: Path, exercises=None):
    files = sorted(root.glob("S*_E*_*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), axis=0, ignore_index=True)
    if exercises:
        df = df[df["exercise"].isin(exercises)]
    return df

def _filter(df, label_col, include_labels=None, exclude_rest=False):
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
            df = df[df["label_id"].astype(int) != 0]
        else:
            df = df[df["label_name"].astype(str).str.lower() != "rest"]
    return df

def _make_proxy_labels(df, fresh_frac=0.20, fatig_frac=0.20, min_windows=30, label_col="label_id"):
    """Within each (subject, exercise, label, repetition):
       - first fresh_frac → class 0 (Fresh)
       - last  fatig_frac → class 1 (Fatigued)
       drop middle windows.
    """
    rows = []
    grp_keys = ["subject","exercise",label_col,"repetition"]
    for keys, g in df.groupby(grp_keys):
        g = g.sort_values("win_start")
        n = len(g)
        if n < min_windows:
            continue
        nf = max(1, int(np.ceil(fresh_frac * n)))
        nt = max(1, int(np.floor(fatig_frac * n)))
        fresh_idx = g.index[:nf]
        fatig_idx = g.index[-nt:]
        tag = pd.Series(index=g.index, dtype="float64")
        tag.loc[fresh_idx] = 0.0
        tag.loc[fatig_idx] = 1.0
        # keep only tagged rows
        gg = g.loc[tag.dropna().index].copy()
        gg["fatigue_proxy"] = tag.dropna().astype(int).values
        rows.append(gg)
    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["fatigue_proxy"])
    out = pd.concat(rows, axis=0, ignore_index=True)
    return out

def _features_matrix(df):
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "fatigue_proxy"]
    X = df[feat_cols].values.astype(float)
    return X, feat_cols

def _make_clf(name: str):
    if name == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=13)
    if name == "svm_rbf":
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", probability=True, class_weight="balanced", C=2.0, gamma="scale"))
    raise ValueError(f"Unknown clf {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", default="data/processed/db2/features")
    ap.add_argument("--exercises", nargs="*", type=int, default=[1,2])
    ap.add_argument("--label-col", default="label_id", choices=["label_id","label_name"])
    ap.add_argument("--include-labels", nargs="*", default=None)
    ap.add_argument("--exclude-rest", action="store_true")

    ap.add_argument("--fresh-frac", type=float, default=0.20)
    ap.add_argument("--fatig-frac", type=float, default=0.20)
    ap.add_argument("--min-windows", type=int, default=30)

    ap.add_argument("--subjects", nargs="*", default=None,
                    help="If given, limit LOSO to these test subjects (e.g., S01 S12 S33).")

    ap.add_argument("--clf", default="logreg", choices=["logreg","rf","svm_rbf"])
    ap.add_argument("--outdir", default="reports/fatigue_proxy")
    args = ap.parse_args()

    root = Path(args.features_root)
    outdir = Path(args.outdir); (outdir/"figures").mkdir(parents=True, exist_ok=True)

    # 1) Load + filter + proxy labels
    df_all = _collect(root, exercises=args.exercises)
    df_all = _filter(df_all, args.label_col, args.include_labels, args.exclude_rest)
    df_all = _make_proxy_labels(df_all,
                                fresh_frac=args.fresh_frac,
                                fatig_frac=args.fatig_frac,
                                min_windows=args.min_windows,
                                label_col=args.label_col)

    if df_all.empty:
        print("[ERR] No data after proxy labeling. Try lowering --min-windows or fractions.")
        return

    # 2) LOSO on fatigue_proxy ∈ {0,1}
    subjects = sorted(df_all["subject"].unique().tolist())
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]

    per_subj = []
    for test_sid in subjects:
        tr = df_all[df_all["subject"] != test_sid].copy()
        te = df_all[df_all["subject"] == test_sid].copy()
        if te.empty or tr.empty: 
            continue

        # Remove any classes absent in train (rare, but safe)
        ytr = tr["fatigue_proxy"].astype(int).values
        yte = te["fatigue_proxy"].astype(int).values
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            print(f"[SKIP] {test_sid}: binary classes not present in train/test")
            continue

        Xtr, feat_cols = _features_matrix(tr)
        Xte, _ = _features_matrix(te)

        clf = _make_clf(args.clf)
        clf.fit(Xtr, ytr)

        # proba if available, else decision_function → sigmoid-ish
        if hasattr(clf, "predict_proba"):
            P = clf.predict_proba(Xte)[:,1]
        else:
            try:
                d = clf.decision_function(Xte)
                # scale to (0,1)
                P = 1.0 / (1.0 + np.exp(-d))
            except Exception:
                P = None

        yhat = clf.predict(Xte)

        acc = accuracy_score(yte, yhat)
        f1m = f1_score(yte, yhat, average="macro", zero_division=0.0)
        f1b = f1_score(yte, yhat, average=None, zero_division=0.0)
        auc = roc_auc_score(yte, P) if P is not None else np.nan

        per_subj.append({
            "test_subject": test_sid,
            "clf": args.clf,
            "accuracy": acc,
            "f1_macro": f1m,
            "f1_fresh": float(f1b[0]),
            "f1_fatigued": float(f1b[1]),
            "roc_auc": auc,
            "n_test": int(len(yte))
        })

        # CM
        if save_confusion_matrix is not None:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(yte, yhat, labels=[0,1])
            cm_path = outdir/"figures"/f"cm_fatigue_proxy_{args.clf}_{test_sid}.png"
            save_confusion_matrix(cm, ["Fresh","Fatigued"], f"Fatigue Proxy — {test_sid} ({args.clf})", str(cm_path))
            print(f"[FIG] {cm_path}")

        # ROC
        if P is not None:
            fpr, tpr, _ = roc_curve(yte, P)
            plt.figure(figsize=(4,4))
            plt.plot(fpr, tpr, lw=2)
            plt.plot([0,1],[0,1], "k--", lw=1)
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {test_sid} ({args.clf})")
            plt.tight_layout()
            rpath = outdir/"figures"/f"roc_fatigue_proxy_{args.clf}_{test_sid}.png"
            plt.savefig(rpath, dpi=200)
            plt.close()
            print(f"[FIG] {rpath}")

        print(f"[LOSO] {test_sid}: acc={acc:.3f}, f1={f1m:.3f}, auc={auc if not np.isnan(auc) else float('nan'):.3f}")

    # Write summary
    dfm = pd.DataFrame(per_subj)
    out_csv = outdir / "fatigue_proxy_loso_metrics.csv"
    dfm.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    if not dfm.empty:
        med = dfm.groupby("clf")["f1_macro"].median().sort_values(ascending=False)
        print("\nMedian macro-F1 across subjects:\n", med)

if __name__ == "__main__":
    main()
