from pathlib import Path
import pandas as pd
import yaml
from collections import defaultdict

def main():
    root = Path("data/processed/db2/features")
    files = sorted(root.glob("S*_E*_*.csv"))
    if not files:
        print("[ERR] No feature CSVs found. Build features first.")
        return

    codes_by_ex = defaultdict(set)
    n_files, n_used, n_skipped = 0, 0, 0

    for f in files:
        n_files += 1
        # Infer exercise id from filename "..._E<ex>_..."
        try:
            ex = int(f.name.split("_E")[1].split("_")[0])
        except Exception:
            ex = 1

        # Peek columns first
        try:
            head = pd.read_csv(f, nrows=0)
            cols = set(head.columns)
        except Exception as e:
            print(f"[WARN] Skipping {f} (cannot read header): {e}")
            n_skipped += 1
            continue

        if "label_id" in cols:
            df = pd.read_csv(f, usecols=["label_id"])
            df = df.dropna(subset=["label_id"])
            if not df.empty:
                for c in df["label_id"].astype(int).unique():
                    codes_by_ex[ex].add(int(c))
                n_used += 1
        else:
            print(f"[WARN] {f} has no 'label_id' column; skipping. (Rebuild features to include label_id.)")
            n_skipped += 1

    if not codes_by_ex:
        print("[ERR] No files with 'label_id' found. Rebuild features and try again.")
        return

    # Build template: ensure rest (0) is present; include all seen codes
    out = {}
    for ex, codes in sorted(codes_by_ex.items()):
        m = {0: "rest"}
        for c in sorted(codes):
            if c == 0:
                continue
            m[int(c)] = f"code_{int(c)}"
        out[f"E{ex}"] = m

    Path("reports").mkdir(parents=True, exist_ok=True)
    out_path = Path("reports/labels_db2_template.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=True, allow_unicode=True)

    print(f"[OK] wrote {out_path}")
    print(f"[INFO] files total={n_files}, used={n_used}, skipped={n_skipped}")
    print("[HINT] For complete coverage, rebuild features so every CSV includes 'label_id'.")

if __name__ == "__main__":
    main()
