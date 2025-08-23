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

# Display names for the 5 selected codes
LABEL_MAP = {
    "6":  "Fist",
    "7":  "Index",
    "10": "Pronation",
    "13": "Wrist Flexion",
    "14": "Wrist Extension",
}

def _map_label(c):
    """
    Robust mapper: handles ints, strings, floats like '6.0', and stray whitespace.
    Falls back to the original token if not in the map.
    """
    s = str(c).strip()
    # try float→int→str (covers '6.0', '6.000')
    try:
        i = int(float(s))
        key = str(i)
        if key in LABEL_MAP:
            return LABEL_MAP[key]
    except Exception:
        pass
    # direct string key (covers '6')
    return LABEL_MAP.get(s, s)



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
    ap.add_argument("--no-figs", action="store_true", help="Skip saving confusion matrices (faster).")

    # NEW (single, canonical definitions)
    ap.add_argument("--only-clf", default=None,
                choices=["lda","svm_rbf","svm_linear","rf","mlp","logreg","ensemble_soft"],
                help="Run just one classifier (fast CM generation).")
    ap.add_argument("--subjects", nargs="*", default=None,
                help="Limit LOSO to these test subjects (e.g., S01 S12 S33).")

    args = ap.parse_args()



    root = Path(args.features_root)
    outdir = Path(args.outdir); (outdir/"figures").mkdir(parents=True, exist_ok=True)

    df_all = _collect_all(root, exercises=args.exercises)
    subjects = sorted(df_all["subject"].unique().tolist())
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]


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

        # === BEGIN PATCH: model selection & fitting ===
        run_base_models = (args.only_clf is None) or (args.only_clf != "ensemble_soft")
        run_ensemble    = (args.only_clf is None) or (args.only_clf == "ensemble_soft")

        if run_base_models:
            base_model_names = ["lda", "svm_rbf", "svm_linear", "rf", "mlp", "logreg"]
            models = {name: make_clf(name) for name in base_model_names}

            # Restrict to a single BASE model only if requested AND valid
            if args.only_clf in models:
                models = {args.only_clf: models[args.only_clf]}

            for name, m in models.items():
                m.fit(Xtr, ytr_enc)

                # Predict
                if hasattr(m, "predict_proba"):
                    P = m.predict_proba(Xte)
                    yhat = np.argmax(P, axis=1)
                else:
                    P = None
                    yhat = m.predict(Xte)

                # Metrics + row
                mets = extended_metrics(yte_enc, yhat, y_proba=P, labels_order=list(range(len(classes))))
                per_subj_rows.append({
                    "test_subject": test_sid, "clf": name, **mets, "n_test": len(yte_enc)
                })

                # Optional CM
            if not args.no_figs:
                from semg_srl.eval.metrics import basic_metrics
                b = basic_metrics(yte_enc, yhat, labels_order=list(range(len(classes))))
                display_classes = [_map_label(c) for c in classes]
                cm_path = outdir / "figures" / f"cm_loso_bench_ensemble_soft_{test_sid}.png"
                print(f"[DBG] classes raw: {list(classes)}")
                print(f"[DBG] classes mapped: {display_classes}")
                print(f"[FIG] saving -> {cm_path}")
                save_confusion_matrix(b["cm"], display_classes, f"LOSO {test_sid} ensemble_soft", str(cm_path))

        # Ensemble (prob-capable bases)
        if run_ensemble:
            # Build three bases fresh; ensure SVM has probability=True for ensemble
            svm = make_clf("svm_rbf")
            try:
                svm.set_params(clf__probability=True)
            except Exception:
                pass
            rf  = make_clf("rf")
            mlp = make_clf("mlp")

            ens = SoftVotingEnsemble([svm, rf, mlp])
            ens.fit(Xtr, ytr_enc)

            P = ens.predict_proba(Xte)
            yhat = np.argmax(P, axis=1)

            mets = extended_metrics(yte_enc, yhat, y_proba=P, labels_order=list(range(len(classes))))
            per_subj_rows.append({
                "test_subject": test_sid, "clf": "ensemble_soft", **mets, "n_test": len(yte_enc)
            })

            if not args.no_figs:
                from semg_srl.eval.metrics import basic_metrics
                b = basic_metrics(yte_enc, yhat, labels_order=list(range(len(classes))))
                display_classes = [_map_label(c) for c in classes]
                cm_path = outdir / "figures" / f"cm_loso_bench_ensemble_soft_{test_sid}.png"
                print(f"[DBG] classes raw: {list(classes)}")
                print(f"[DBG] classes mapped: {display_classes}")
                print(f"[FIG] saving -> {cm_path}")
                save_confusion_matrix(b["cm"], display_classes, f"LOSO {test_sid} ensemble_soft", str(cm_path))

            # === END PATCH ===


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
