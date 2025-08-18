import os
from pathlib import Path
import pytest

from semg_srl.io.ninapro_db2 import load_db2_subject, scan_unique_codes

DATA_ROOT = Path("data/raw/db2")

pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="DB2 data not present locally")

def test_scan_and_load_subject():
    # Pick the first configured subject from datasets.yaml
    import yaml
    with open("configs/datasets.yaml", "r") as f:
        subjects = yaml.safe_load(f)["db2"]["subjects"]
    sid = subjects[0]
    codes = scan_unique_codes(sid)
    assert isinstance(codes, dict) and len(codes) >= 1

    data = load_db2_subject(sid)
    # check at least one exercise loaded with emg and labels
    assert any("emg" in v and v["emg"].ndim == 2 for v in data.values())
