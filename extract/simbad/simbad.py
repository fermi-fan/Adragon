#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SIMBAD batch fetch via Simbad.query_objects()
---------------------------------------------
- 每个参数 f（如 "coordinates","dim","morphtype","parallax","propermotions","sp","velocity","flux(allfluxes)"）
  执行：s = Simbad(); s.add_votable_fields(f); s.query_objects(names)
- f == "coordinates"：原样保存
- 其他参数：删除列 [ra, dec, coo_err_maj, coo_err_min, coo_err_angle, coo_wavelength, coo_bibcode]
- 若存在 'matched_id'、'object_number_id' 列则删除
- 可选保存格式：CSV / XLSX / 两者皆可
"""

from __future__ import annotations
from typing import Sequence, List
import os, time
import pandas as pd
from astroquery.simbad import Simbad
from requests.exceptions import ReadTimeout, ConnectTimeout
from http.client import RemoteDisconnected
from tqdm import tqdm

# ============ 常量配置 ============
COO_COLS_TO_DROP = [
    "ra", "dec",
    "coo_err_maj", "coo_err_min", "coo_err_angle",
    "coo_wavelength", "coo_bibcode",
]

TQDM_OUTER = dict(
    desc="Parameter groups", unit="group", position=0, leave=True,
    dynamic_ncols=False, mininterval=1.5, smoothing=0.9,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
)

# ============ 辅助函数 ============
def _safe_query(call, *, retries: int = 1, timeout: int = 75):
    """执行查询（带超时与重试）"""
    from astroquery.query import BaseQuery
    last = None
    for k in range(retries + 1):
        try:
            BaseQuery.TIMEOUT = timeout
            return call()
        except (ReadTimeout, ConnectTimeout, RemoteDisconnected, Exception) as e:
            last = e
            time.sleep(0.8 + 0.4 * k)
    raise last

def _tbl_to_df(tbl) -> pd.DataFrame:
    """Astropy Table → pandas DataFrame"""
    if tbl is None:
        return pd.DataFrame()
    df = tbl.to_pandas()
    for c in df.columns:
        if df[c].dtype == object and len(df[c]) and isinstance(df[c].iloc[0], (bytes, bytearray)):
            df[c] = df[c].apply(lambda x: x.decode(errors="ignore") if isinstance(x, (bytes, bytearray)) else x)
    df.columns = [str(c).replace(" ", "_") for c in df.columns]
    return df

# ============ 主体函数 ============
def query_objects_single_parameter_batch(
    names: Sequence[str],
    f: str,
    *,
    timeout_seconds: int = 75,
    retries: int = 1,
) -> pd.DataFrame:
    """用 Simbad.query_objects() 一次性批量请求"""
    names = [str(n).strip() for n in names if str(n).strip()]
    if not names:
        return pd.DataFrame()

    s = Simbad()
    s.add_votable_fields(f)
    tbl = _safe_query(lambda: s.query_objects(names),
                      retries=retries, timeout=timeout_seconds)
    df = _tbl_to_df(tbl)
    if df.empty:
        return df

    # 去掉 matched_id 与 object_number_id
    for col in ("matched_id", "object_number_id"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # 非 coordinates 删除坐标列
    if f != "coordinates":
        df = df.drop(columns=[c for c in COO_COLS_TO_DROP if c in df.columns], errors="ignore")

    return df

def run_and_save(
    names: Sequence[str],
    parameters: Sequence[str],
    out_dir: str = "./simbad_out",
    *,
    save_formats: Sequence[str] = ("csv", "xlsx"),  # 可选保存格式
    timeout_seconds: int = 75,
    retries: int = 1,
):
    os.makedirs(out_dir, exist_ok=True)
    for f in tqdm(parameters, **TQDM_OUTER):
        df = query_objects_single_parameter_batch(
            names, f,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )

        fname = f.replace("/", "_")
        base = os.path.join(out_dir, fname)

        if "csv" in save_formats:
            df.to_csv(base + ".csv", index=False)
        if "xlsx" in save_formats:
            with pd.ExcelWriter(base + ".xlsx", engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False)

        print(f"✅ Saved [{f}] -> {', '.join(save_formats)} (rows={len(df)})", flush=True)

# ============ 示例运行 ============
if __name__ == "__main__":
    import load  # 你的读取函数

    # 从 Excel 读取名字
    names = load.read_upload("./simbad_test.xlsx", columns={'name': 'Name'})['name'].tolist()

    params = [
        "coordinates",
        "dim",
        "morphtype",
        "parallax",
        "propermotions",
        "sp",
        "velocity",
        "flux(allfluxes)"
    ]

    run_and_save(
        names, params,
        out_dir="./simbad_out",
        save_formats=("xlsx"),
        timeout_seconds=75,
        retries=1,
    )
