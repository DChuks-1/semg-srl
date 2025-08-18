from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.io import loadmat
import h5py
import yaml
import re
import os

def _read_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _try_loadmat_v5(path: Path) -> Optional[dict]:
    try:
        return loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except Exception:
        return None

def _try_loadmat_v73(path: Path) -> Optional[dict]:
    try:
        out = {}
        with h5py.File(str(path), "r") as f:
            def _to_np(name: str):
                if name in f:
                    arr = f[name][()]
                    arr = np.array(arr)
                    # MATLAB stores column-major; many variables come transposed
                    # We correct orientation for common signals below.
                    return arr
                return None
            for key in ("emg", "sEMG", "stimulus", "restimulus", "repetition", "rerepetition", "subject"):
                out[key] = _to_np(key)
        return out
    except Exception:
        return None

def _load_mat_any(path: Path) -> dict:
    d = _try_loadmat_v5(path)
    if d is not None:
        return d
    d = _try_loadmat_v73(path)
    if d is not None:
        return d
    raise IOError(f"Could not read MAT file (v5/v7.3) at {path}")

def _standardise_keys(d: dict) -> dict:
    # Normalise possible key variants
    emg = d.get("emg", d.get("sEMG", None))
    stim = d.get("restimulus", d.get("stimulus", None))
    rep  = d.get("rerepetition", d.get("repetition", None))
    subj = d.get("subject", None)

    # Ensure 2D EMG: (n_samples, n_channels)
    if emg is None:
        raise KeyError("No 'emg' or 'sEMG' variable found in MAT file.")
    emg = np.array(emg)
    # Common cases: (channels, samples) or (samples, channels)
    if emg.ndim == 2 and emg.shape[0] < emg.shape[1]:
        # If channels < samples and axis0 is smaller, assume (channels, samples) -> transpose
        emg = emg.T
    elif emg.ndim != 2:
        emg = np.atleast_2d(emg)
        if emg.shape[0] < emg.shape[1]:
            emg = emg.T

    # Stimulus & repetition to 1D vectors of length n_samples (where possible)
    def _to_1d(x):
        if x is None:
            return None
        arr = np.array(x).squeeze()
        return arr

    stim = _to_1d(stim)
    rep  = _to_1d(rep)
    subj = _to_1d(subj)

    return {"emg": emg, "stimulus": stim, "repetition": rep, "subject": subj}

def _guess_subject_id(mat_path: Path) -> str:
    # Try pattern from filename or parent folder (S01/S1/etc.)
    m = re.search(r"(S\d{1,2})", mat_path.stem, re.IGNORECASE)
    if m:
        s = m.group(1).upper()
        return f"S{int(s[1:]):02d}"
    m = re.search(r"(S\d{1,2})", str(mat_path.parent), re.IGNORECASE)
    if m:
        s = m.group(1).upper()
        return f"S{int(s[1:]):02d}"
    return "SXX"

def _guess_exercise_id(mat_path: Path) -> Optional[int]:
    m = re.search(r"E(\d)", mat_path.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None

def find_subject_files(root: str | Path, subject_id: str, exercises: List[int]) -> List[Path]:
    root = Path(root)
    pats = []
    # Typical patterns
    pats.append(root.glob(f"{subject_id}/*.mat"))        # data/raw/db2/S01/*.mat
    num = int(subject_id[1:])
    pats.append(root.glob(f"{subject_id}/S{num}_E*.mat"))
    pats.append(root.glob(f"**/{subject_id}_E*.mat"))
    pats.append(root.glob(f"**/S{num}_E*.mat"))
    files = set()
    for p in pats:
        for f in p:
            ex = _guess_exercise_id(f)
            if ex in exercises or ex is None:
                files.add(f.resolve())
    return sorted(files)

def load_db2_subject(subject_id: str,
                     cfg_ds_path: str = "configs/datasets.yaml",
                     cfg_lbl_path: str = "configs/labels_db2.yaml",
                     exercise_filter: Optional[List[int]] = None) -> Dict[int, dict]:
    """
    Load all exercises for one subject into memory.
    Returns: { exercise_id: { 'emg': (N,C), 'stimulus': (N,), 'repetition': (N,), 'labels': (N,) } }
    """
    ds_cfg = _read_yaml(cfg_ds_path)["db2"]
    lbl_cfg = _read_yaml(cfg_lbl_path)
    root = ds_cfg["root"]
    exercises = exercise_filter if exercise_filter is not None else ds_cfg.get("exercises", [1,2,3])

    out = {}
    files = find_subject_files(root, subject_id, exercises)
    if not files:
        raise FileNotFoundError(f"No MAT files found for {subject_id} under {root}")

    for f in files:
        d = _standardise_keys(_load_mat_any(f))
        ex = _guess_exercise_id(f)
        if ex is None:
            # If file name lacks E#, try to infer from label mapping coverage
            ex = 1
        # Map integer codes -> names (if mapping exists)
        mapping = lbl_cfg.get(f"E{ex}", {})
        labels_named = None
        if d["stimulus"] is not None:
            labels_named = np.array([mapping.get(int(x), f"code_{int(x)}") for x in d["stimulus"]])
        out[ex] = {
            "path": str(f),
            "emg": d["emg"],
            "stimulus": d["stimulus"],
            "repetition": d["repetition"],
            "labels_named": labels_named
        }
    return out

def scan_unique_codes(subject_id: str,
                      cfg_ds_path: str = "configs/datasets.yaml",
                      exercise_filter: Optional[List[int]] = None) -> Dict[int, List[int]]:
    """Return observed unique label codes per exercise (ignores None)."""
    ds_cfg = _read_yaml(cfg_ds_path)["db2"]
    root = ds_cfg["root"]
    exercises = exercise_filter if exercise_filter is not None else ds_cfg.get("exercises", [1,2,3])
    files = find_subject_files(root, subject_id, exercises)
    codes = {}
    for f in files:
        ex = _guess_exercise_id(f) or 1
        d = _standardise_keys(_load_mat_any(f))
        if d["stimulus"] is None:
            continue
        uniq = sorted(set(int(v) for v in np.array(d["stimulus"]).ravel() if not np.isnan(v)))
        codes.setdefault(ex, set()).update(uniq)
    return {ex: sorted(list(vals)) for ex, vals in codes.items()}
