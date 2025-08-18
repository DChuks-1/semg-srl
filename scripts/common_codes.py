from pathlib import Path
import pandas as pd
import numpy as np

def main():
    root = Path("data/processed/db2/features")
    files = sorted(root.glob("S*_E*_*.csv"))
    if not files:
        print("[ERR] No feature CSVs found. Build features first.")
        return
    per_subj = {}
    for f in files:
        df = pd.read_csv(f, usecols=["subject","label_id"])
        df = df.dropna(subset=["label_id"])
        subj = df["subject"].iloc[0]
        per_subj.setdefault(subj, set()).update(df["label_id"].astype(int).unique().tolist())
    subjects = sorted(per_subj.keys())
    common = set.intersection(*map(set, per_subj.values()))
    # drop rest if 0
    common = {c for c in common if c != 0}
    out = sorted(common)
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/common_codes.txt","w",encoding="utf-8") as f:
        for c in out:
            f.write(f"{c}\n")
    print("[INFO] Subjects:", subjects)
    print("[OK] wrote reports/common_codes.txt:", out)

if __name__ == "__main__":
    main()
