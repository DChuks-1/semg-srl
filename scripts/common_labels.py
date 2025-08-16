from pathlib import Path
import pandas as pd

def main():
    root = Path("data/processed/db2/features")
    files = sorted(root.glob("S*_E*_*.csv"))
    if not files:
        print("[ERR] No feature CSVs found.")
        return
    per_subj = {}
    for f in files:
        df = pd.read_csv(f, usecols=["subject","label_name"])
        df = df.dropna(subset=["label_name"])
        subj = df["subject"].iloc[0]
        per_subj.setdefault(subj, set()).update(df["label_name"].astype(str).unique().tolist())
    subjects = sorted(per_subj.keys())
    common = set.intersection(*map(set, per_subj.values()))
    # remove 'rest' if present; we’ll exclude it anyway
    common = {c for c in common if c.lower() != "rest"}
    out = sorted(common)
    print("[INFO] Subjects:", subjects)
    print("[INFO] Common label set across all subjects (excluding 'rest'):\n", out)
    Path("reports").mkdir(exist_ok=True, parents=True)
    with open("reports/common_labels.txt","w",encoding="utf-8") as f:
        for c in out:
            f.write(f"{c}\n")
    print("[OK] wrote reports/common_labels.txt")

if __name__ == "__main__":
    main()
