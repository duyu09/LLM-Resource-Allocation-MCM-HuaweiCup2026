from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    """Recreate one generated-output directory so stale parquet/audit files cannot survive."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def canonical_model_key(value) -> str:
    """Conservative normalization only; no fuzzy model matching is done in preprocessing."""
    if pd.isna(value):
        return ""
    s = unicodedata.normalize("NFKC", str(value)).strip().lower()
    s = s.replace("\\", "/")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"/+", "/", s)
    return s.strip("/")


def safe_log10(series: pd.Series) -> pd.Series:
    x = numeric(series).astype(float)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    mask = x > 0
    out.loc[mask] = np.log10(x.loc[mask])
    return out


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a normal CSV only. utf-8-sig opens cleanly in Windows Excel."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


class ChunkCSVWriter:
    """Streaming CSV writer for large A1-A3 and C8 outputs."""
    def __init__(self, path: Path):
        ensure_dir(path.parent)
        self.path = path
        if self.path.exists():
            self.path.unlink()
        self.columns = None
        self.first = True

    def write(self, df: pd.DataFrame):
        if df.empty:
            return
        if self.columns is None:
            self.columns = list(df.columns)
        else:
            missing = [c for c in self.columns if c not in df.columns]
            extra = [c for c in df.columns if c not in self.columns]
            if missing or extra:
                raise ValueError(f"Chunk columns changed. missing={missing}, extra={extra}")
            df = df[self.columns]
        df.to_csv(
            self.path,
            mode="w" if self.first else "a",
            header=self.first,
            index=False,
            encoding="utf-8-sig" if self.first else "utf-8",
        )
        self.first = False
