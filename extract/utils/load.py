import os
from random import sample

import pandas as pd
from typing import Optional, Union, Iterable


def read_upload(
    file: str,
    columns: dict[str, str],
    sheet: Optional[Union[str, int]] = None,
    *,
    strip_whitespace: bool = True,
    strip_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:

    ext = os.path.splitext(file)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(file, sheet_name=(0 if sheet is None else sheet))
    else:
        df = pd.read_csv(file, encoding="utf-8")


    out = pd.DataFrame({out_name: df[real_col] for out_name, real_col in columns.items()})

    # 清理前后空格（仅字符串值）
    if strip_whitespace:
        targets = list(strip_cols) if strip_cols else list(out.columns)
        for c in targets:
            if c in out.columns:
                out[c] = out[c].map(lambda x: x.strip() if isinstance(x, str) else x)

    return out
