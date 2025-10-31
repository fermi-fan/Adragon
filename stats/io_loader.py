# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_table(path: str | Path) -> pd.DataFrame:
    """
    读取 csv/parquet 表格（最小实现；后续可加 FITS）。
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported format: {suf}")
