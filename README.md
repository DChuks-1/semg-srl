# semg-srl
Dual-function sEMG pipeline for gesture classification and fatigue monitoring (simulation-friendly).



## Quickstart
```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest
```
## Checkpoint 1 (5-class DB2 baseline)
- **Data**: NinaPro DB2, 5 classes (6,7,13,14,10).
- **Preprocessing**: 20–450 Hz BP, 50 Hz notch, 200 ms @ 50% overlap; transition-trim ±100 ms; ≥80% label majority.
- **Features**: RMS/MAV/WL/ZC/SSC + MDF/MPF.
- **Eval**: Within-subject (GroupKFold) + LOSO; macro-F1 primary + extended metrics; soft-voting ensemble.

### How to reproduce
```bash
# Windows PowerShell
python scripts/build_db2_features.py --outdir data/processed/db2/features --trim-ms 100 --min-majority 0.8
$codes = 6,7,13,14,10
python scripts/benchmark_within.py --features-root data/processed/db2/features --subject S01 --exercises 1 2 --exclude-rest --label-col label_id --include-labels $codes
python scripts/benchmark_loso.py --features-root data/processed/db2/features --exercises 1 2 --exclude-rest --label-col label_id --include-labels $codes --no-figs
python scripts/report_leaderboard.py
```
## 🔁 How to Reproduce Results & Figures

This repo contains a compact pipeline to benchmark sEMG gesture recognition on a 5-class subset
(`Fist, Index, Pronation, Wrist Flexion, Wrist Extension`) using NinaPro DB2 features and to
produce publication-ready figures. All commands are Windows PowerShell–friendly.

### 1) Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .[dev]
```
