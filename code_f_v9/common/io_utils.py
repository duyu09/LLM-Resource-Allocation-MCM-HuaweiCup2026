from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

import joblib
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: str | Path, required: Iterable[str] | None = None, **kwargs) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required input does not exist: {p}")
    df = pd.read_csv(p, **kwargs)
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing required columns: {missing}")
    return df


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def write_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_joblib(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(obj, p)
    return p


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"Object of type {type(x)} is not JSON serializable")
