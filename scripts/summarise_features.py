from pathlib import Path
import pandas as pd

def main():
    root = Path("data/processed/db2/features")
    files = sorted(root.glob("S*_E*_*.csv"))
    rows = []
    for f in files:
        df = pd.read_csv(f, usecols=["subject","exercise","label_name"], dtype={"label_name":"string"})
        df = df.dropna(subset=["label_name"])
        for subj, ex, name, cnt in df.value_counts(["subject","exercise","label_name"]).reset_index(name="count").itertuples(index=False):
            rows.append({"subject":subj, "exercise":ex, "label_name":str(name), "count":int(cnt)})
    out = pd.DataFrame(rows).sort_values(["label_name","subject","exercise"])
    out_path = Path("reports/labels_summary.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} with {len(out)} rows")
    print(out.groupby("label_name")["count"].sum().sort_values(ascending=False).head(20))

if __name__ == "__main__":
    main()
