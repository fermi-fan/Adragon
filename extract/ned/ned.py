#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Iterable, Dict, List, Optional, Tuple
import os, time, socket, random, re, sys
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.ned import Ned
from requests.exceptions import ReadTimeout, ConnectTimeout, ConnectionError as RequestsConnectionError
from http.client import RemoteDisconnected
from tqdm import tqdm

# 无 GUI 环境后端
import matplotlib
if not os.environ.get("DISPLAY") and sys.platform != "win32" and os.environ.get("MPLBACKEND", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# =============== 常量 ===============
VALID_TABLES = {'positions','redshifts','diameters','photometry','object_notes'}
COLOR_MEAS  = "#2a77c7"
COLOR_LIMIT = "#7fb3ff"
ECOLOR      = "#6e7f91"
POINT_ALPHA = 0.9
ARROW_ALPHA = 0.95

# =============== 基础工具 ===============
def _drop_no_col(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['No.'], errors='ignore')

def _prepend_name(df: pd.DataFrame, name_val: str) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, 'Name', name_val, allow_duplicates=True)
    return df

def _to_skycoord_icrs(ra, dec) -> Optional[SkyCoord]:
    if ra is None or dec is None:
        return None
    if isinstance(ra, u.Quantity) and isinstance(dec, u.Quantity):
        if ra.unit.is_equivalent(u.deg) and dec.unit.is_equivalent(u.deg):
            return SkyCoord(ra.to(u.deg), dec.to(u.deg), frame='icrs')
        if ra.unit.is_equivalent(u.hourangle) and dec.unit.is_equivalent(u.deg):
            return SkyCoord(ra, dec, frame='icrs')
    def norm(s):
        s = str(s).strip()
        return s.translate({ord('°'):'d', ord('º'):'d', ord('′'):'m', ord('’'):'m', ord('″'):'s'})
    sra, sdec = norm(ra), norm(dec)
    try:
        return SkyCoord(sra, sdec, unit=(u.hourangle, u.deg), frame='icrs')
    except Exception:
        pass
    try:
        return SkyCoord(float(sra)*u.deg, float(sdec)*u.deg, frame='icrs')
    except Exception:
        return None

def _first_row(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[[0]].copy() if len(df) else df.copy()

def _select_diameters_max_major_axis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or ('NED Major Axis' not in df.columns):
        return pd.DataFrame()
    def to_float(x):
        try:
            if hasattr(x, 'to') and hasattr(x, 'value'):
                return float(x.to(u.arcsec).value) if x.unit.is_equivalent(u.arcsec) else float(x.value)
            return float(x)
        except Exception:
            return np.nan
    vals = df['NED Major Axis'].apply(to_float)
    if vals.isna().all():
        return pd.DataFrame()
    return df.loc[[vals.idxmax()]].copy()

def _filter_photometry_jy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or ('NED Units' not in df.columns):
        return pd.DataFrame()
    return df[df['NED Units'].astype(str).str.strip().eq('Jy')].copy()

# =============== positions 处理（第一行 + 度制/银道，6位小数） ===============
def _process_positions(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {'RA', 'DEC'}.issubset(df.columns):
        return pd.DataFrame()
    one = df.iloc[[0]].copy()
    sc = _to_skycoord_icrs(one.iloc[0]['RA'], one.iloc[0]['DEC'])
    if sc is None:
        return pd.DataFrame()
    one['RA[deg]']  = round(sc.ra.deg, 6)
    one['DEC[deg]'] = round(sc.dec.deg, 6)
    gal = sc.galactic
    one['Lon[deg]'] = round(float(gal.l.deg), 6)
    one['Lat[deg]'] = round(float(gal.b.deg), 6)
    one = _prepend_name(one, name)
    return one

def _process_table(name: str, table: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if table == 'positions':
        return _process_positions(name, df)
    if table == 'redshifts':
        out = _first_row(df).copy()
        out = _prepend_name(out, name)
        return out
    if table == 'diameters':
        out = _select_diameters_max_major_axis(df)
        if out.empty: return out
        out = _prepend_name(out, name)
        return out
    if table == 'photometry':
        out = _filter_photometry_jy(df)
        if out.empty: return out
        out = _prepend_name(out, name)
        return out
    if table == 'object_notes':
        out = df.copy()
        out = _prepend_name(out, name)
        return out
    return pd.DataFrame()

# =============== SED 绘图（你的风格） ===============
def _sanitize_filename(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')

def _parse_uncertainty(s: str) -> Tuple[str, Optional[float]]:
    if s is None: return ("none", None)
    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none"}: return ("none", None)
    if s.startswith("+/-"):
        try: return ("sym", float(s[3:].strip()))
        except: return ("none", None)
    if s.startswith("<"):
        try: return ("lt", float(s[1:].strip()))
        except: return ("none", None)
    if s.startswith(">"):
        try: return ("gt", float(s[1:].strip()))
        except: return ("none", None)
    try: return ("num", float(s))
    except: return ("none", None)

def _prepare_photometry_for_sed(df: pd.DataFrame) -> pd.DataFrame:
    need_cols = {'Frequency', 'Flux Density'}
    if not need_cols.issubset(df.columns):
        return pd.DataFrame()
    d = df.copy()
    d['Frequency']    = pd.to_numeric(d['Frequency'], errors='coerce')
    d['Flux Density'] = pd.to_numeric(d['Flux Density'], errors='coerce')  # 允许 NaN（用于限值）
    d = d.dropna(subset=['Frequency'])
    cols = ['Frequency', 'Flux Density'] + (['NED Uncertainty'] if 'NED Uncertainty' in d.columns else [])
    return d[cols].sort_values('Frequency')

def _split_points(df: pd.DataFrame):
    meas, uppers, lowers = [], [], []
    for _, r in df.iterrows():
        nu = float(r["Frequency"])
        fnu = r["Flux Density"]
        kind, val = _parse_uncertainty(r.get("NED Uncertainty"))
        if np.isfinite(fnu):
            if kind in {"sym", "num"} and val is not None:
                meas.append((nu, float(fnu), float(val)))
            else:
                meas.append((nu, float(fnu), np.nan))
        else:
            if kind == "lt" and val is not None:
                uppers.append((nu, float(val)))
            elif kind == "gt" and val is not None:
                lowers.append((nu, float(val)))
    det = pd.DataFrame(meas, columns=["Frequency", "Flux", "Err"])
    up  = pd.DataFrame(uppers, columns=["Frequency", "Limit"])
    low = pd.DataFrame(lowers, columns=["Frequency", "Limit"])
    return det, up, low

def _y_compute(det: pd.DataFrame, up: pd.DataFrame, low: pd.DataFrame, yunit: str):
    scale = 1.0
    ylabel = r"$\nu F_{\nu}$ [Jy$\cdot$Hz]"
    if yunit.lower() == "cgs":
        scale = 1e-23
        ylabel = r"$\nu F_{\nu}$ [erg cm$^{-2}$ s$^{-1}$]"
    if not det.empty:
        det["Y"]    = det["Frequency"] * det["Flux"] * scale
        det["Yerr"] = det["Frequency"] * det["Err"] * scale
    if not up.empty:
        up["Y"] = up["Frequency"] * up["Limit"] * scale
    if not low.empty:
        low["Y"] = low["Frequency"] * low["Limit"] * scale
    return ylabel

def _expand_decades(lo: float, hi: float, k: int):
    lo = max(float(lo), np.finfo(float).tiny)
    hi = float(hi) if float(hi) > lo else lo * 10.0
    f = 10.0 ** max(0, int(k))
    return max(lo / f, np.finfo(float).tiny), hi * f

def _draw_limit_arrow(ax, x, y, direction: str,
                      head_dex: float = 0.18,
                      lw: float = 1.0,
                      cap_lw: float = 1.5,
                      cap_dex: float = 0.18,   # 横线总宽度（dex），与调用端一致
                      color: str = COLOR_LIMIT, alpha: float = ARROW_ALPHA, zorder: int = 4):
    # 垂向箭头
    if direction.lower() == "down":
        y_head = y / (10 ** head_dex)
        y_cap  = y
    else:
        y_head = y * (10 ** head_dex)
        y_cap  = y
    arrow = FancyArrowPatch(
        posA=(x, y_cap), posB=(x, y_head),
        arrowstyle='->', mutation_scale=10,
        linewidth=lw, color=color, alpha=alpha,
        shrinkA=0, shrinkB=0, zorder=zorder
    )
    ax.add_patch(arrow)
    # 横线：log 对称，保证箭杆在正中
    r = 10 ** (cap_dex / 2.0)
    x_left, x_right = x / r, x * r
    cap = Line2D([x_left, x_right], [y_cap, y_cap],
                 lw=cap_lw, color=color, alpha=alpha, zorder=zorder)
    ax.add_line(cap)

def _sed_draw(ax, det: pd.DataFrame, up: pd.DataFrame, low: pd.DataFrame,
              jitter_dex: float, arrow_dex: float, arrow_lw: float, cap_lw: float, cap_dex: float):
    # 测量点
    if not det.empty:
        x = det["Frequency"].to_numpy(float)
        y = det["Y"].to_numpy(float)
        if jitter_dex > 0:
            logx = np.log10(x)
            rng = np.random.default_rng(20251028)
            x = np.power(10.0, logx + (rng.random(len(x)) - 0.5) * 2.0 * jitter_dex)
        if det["Yerr"].notna().any():
            ax.errorbar(x, y, yerr=det["Yerr"].to_numpy(float),
                        fmt="o", ms=4.2, lw=1.0, elinewidth=1.0, capsize=2.5,
                        color=COLOR_MEAS, ecolor=ECOLOR, alpha=POINT_ALPHA, label="Measured")
        else:
            ax.plot(x, y, "o", ms=4.2, color=COLOR_MEAS, alpha=POINT_ALPHA, label="Measured")

    # 上限
    drew_up = False
    if not up.empty:
        xu = up["Frequency"].to_numpy(float)
        yu = up["Y"].to_numpy(float)
        if jitter_dex > 0:
            logx = np.log10(xu); rng = np.random.default_rng(20251028)
            xu = np.power(10.0, logx + (rng.random(len(xu)) - 0.5) * 2.0 * jitter_dex)
        for xi, yi in zip(xu, yu):
            _draw_limit_arrow(ax, xi, yi, direction="down",
                              head_dex=arrow_dex, lw=arrow_lw,
                              cap_lw=cap_lw, cap_dex=cap_dex)
        drew_up = True

    # 下限
    drew_low = False
    if not low.empty:
        xl = low["Frequency"].to_numpy(float)
        yl = low["Y"].to_numpy(float)
        if jitter_dex > 0:
            logx = np.log10(xl); rng = np.random.default_rng(20251028)
            xl = np.power(10.0, logx + (rng.random(len(xl)) - 0.5) * 2.0 * jitter_dex)
        for xi, yi in zip(xl, yl):
            _draw_limit_arrow(ax, xi, yi, direction="up",
                              head_dex=arrow_dex, lw=arrow_lw,
                              cap_lw=cap_lw, cap_dex=cap_dex)
        drew_low = True

    # 图例：用箭头符号的代理图元，保证可见
    handles, labels = ax.get_legend_handles_labels()
    if drew_up:
        handles.append(Line2D([0],[0], linestyle='None',
                              marker=r'$\downarrow$', markersize=14,
                              color=COLOR_LIMIT, alpha=ARROW_ALPHA))
        labels.append("Upper limit")
    if drew_low:
        handles.append(Line2D([0],[0], linestyle='None',
                              marker=r'$\uparrow$', markersize=14,
                              color=COLOR_LIMIT, alpha=ARROW_ALPHA))
        labels.append("Lower limit")
    if handles:
        ax.legend(handles, labels)

def _plot_sed_from_df(name: str, df_jy: pd.DataFrame, outdir: str,
                      yunit: str = "HzJy", extend_decades: int = 2,
                      jitter_dex: float = 0.01,
                      arrow_dex: float = 0.8, arrow_lw: float = 1.2,
                      cap_lw: float = 0.8, cap_dex: float = 0.18,
                      title_suffix: str = "NED Photometry") -> Optional[str]:
    d = _prepare_photometry_for_sed(df_jy)
    if d.empty:
        return None
    det, up, low = _split_points(d)
    ylabel = _y_compute(det, up, low, yunit)

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\nu$ [Hz]"); ax.set_ylabel(ylabel)

    _sed_draw(ax, det, up, low,
              jitter_dex=jitter_dex,
              arrow_dex=arrow_dex, arrow_lw=arrow_lw,
              cap_lw=cap_lw, cap_dex=cap_dex)

    # 自动范围
    xvals = d["Frequency"].to_numpy(float)
    xvals = xvals[np.isfinite(xvals) & (xvals > 0)]
    xlo, xhi = (1.0, 10.0) if xvals.size == 0 else (np.nanmin(xvals), np.nanmax(xvals))
    xlo, xhi = _expand_decades(xlo, xhi, extend_decades)

    y_arrays = []
    if not det.empty: y_arrays.append(det["Y"].to_numpy(float))
    if not up.empty:  y_arrays.append(up["Y"].to_numpy(float))
    if not low.empty: y_arrays.append(low["Y"].to_numpy(float))
    if y_arrays:
        yvals = np.concatenate(y_arrays)
        yvals = yvals[np.isfinite(yvals) & (yvals > 0)]
        ylo, yhi = (np.nanmin(yvals), np.nanmax(yvals))
    else:
        ylo, yhi = (1.0, 10.0)
    ylo, yhi = _expand_decades(ylo, yhi, extend_decades)

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)

    # 刻度风格
    def _auto_log_ticks(vmin: float, vmax: float, n: int) -> List[float]:
        pmin = int(np.floor(np.log10(max(vmin, np.finfo(float).tiny))))
        pmax = int(np.ceil(np.log10(vmax)))
        grid = np.linspace(pmin, pmax, num=max(2, n))
        return list(np.power(10.0, grid))
    ax.set_xticks(_auto_log_ticks(xlo, xhi, 5))
    ax.set_yticks(_auto_log_ticks(ylo, yhi, 5))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    ax.grid(True, which="major", ls="-", alpha=0.25)
    ax.grid(True, which="minor", ls="--", alpha=0.15)
    for s in ax.spines.values(): s.set_linewidth(1.4)
    ax.tick_params(length=6, width=1.2, which="major")
    ax.tick_params(length=3, width=1.0, which="minor")
    ax.set_title(f"{name} — {title_suffix}")

    os.makedirs(os.path.join(outdir, "sed"), exist_ok=True)
    fpath = os.path.join(outdir, "sed", f"{_sanitize_filename(name)}_sed_finearrow.png")
    plt.tight_layout(); fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fpath

# =============== 带重试的 NED 请求 ===============
def _ned_get_table_with_retry(
    name: str,
    table: str,
    timeout_sec: float,
    max_retries: int,
    backoff_base: float,
    backoff_jitter: float,
    print_log: bool = False,
):
    attempt = 0
    while True:
        attempt += 1
        try:
            return Ned.get_table(name, table=table, timeout=timeout_sec)
        except (ReadTimeout, ConnectTimeout, RequestsConnectionError, RemoteDisconnected, socket.timeout) as e:
            if attempt > max_retries:
                if print_log:
                    print(f"  ! GIVE UP {name}/{table}: {type(e).__name__}: {e}")
                raise
            sleep_s = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, backoff_jitter)
            if print_log:
                print(f"  ! RETRY {name}/{table} (attempt {attempt}/{max_retries}) after {sleep_s:.2f}s due to {type(e).__name__}")
            time.sleep(sleep_s)

# =============== 主函数：表优先提取 + 绘图 ===============
def fetch_ned_tables(
    names: Iterable[str],
    tables: Iterable[str],
    outdir: str,
    throttle_sec: float = 0.0,
    random_throttle: bool = False,
    print_log: bool = True,
    timeout_sec: float = 60,
    max_retries: int = 2,
    backoff_base: float = 1.5,
    backoff_jitter: float = 0.5,
    keep_not_found: bool = True,
    plot_phot_sed: bool = True,
    sed_energy_unit: str = 'HzJy',   # 'HzJy' 或 'cgs'
    extend_decades: int = 2,
    jitter_dex: float = 0.01,
    arrow_dex: float = 0.8,
    arrow_lw: float = 1.2,
    cap_lw: float = 0.8,
    cap_dex: float = 0.18,
) -> Dict[str, str]:
    tables = [t.strip().lower() for t in tables]
    for t in tables:
        if t not in VALID_TABLES:
            raise ValueError(f"Unknown table: {t}; valid: {sorted(VALID_TABLES)}")

    names_list = list(names)
    os.makedirs(outdir, exist_ok=True)
    saved: Dict[str, str] = {}

    for t in tables:
        if print_log:
            print(f"\n===== [TABLE] {t} =====")
        parts: List[pd.DataFrame] = []

        for name in tqdm(names_list, desc=f"[{t}] Fetching", unit="src", total=len(names_list)):
            try:
                tbl = _ned_get_table_with_retry(
                    name=name, table=t,
                    timeout_sec=timeout_sec,
                    max_retries=max_retries,
                    backoff_base=backoff_base,
                    backoff_jitter=backoff_jitter,
                    print_log=False,
                )

                if throttle_sec > 0:
                    pause = throttle_sec + (random.uniform(0, 0.5) if random_throttle else 0.0)
                    time.sleep(pause)

                if tbl is None or len(tbl) == 0:
                    if keep_not_found:
                        parts.append(pd.DataFrame([{'Name': name, 'Status': 'NOT_FOUND'}]))
                    continue

                df = tbl.to_pandas() if hasattr(tbl, 'to_pandas') else pd.DataFrame(tbl)
                df = _drop_no_col(df)

                dfp = _process_table(name, t, df)

                if dfp.empty:
                    if keep_not_found:
                        parts.append(pd.DataFrame([{'Name': name, 'Status': 'NOT_FOUND'}]))
                    continue

                # 画 SED（仅 photometry）
                if t == 'photometry' and plot_phot_sed:
                    try:
                        _plot_sed_from_df(
                            name=name, df_jy=dfp, outdir=outdir,
                            yunit=sed_energy_unit,
                            extend_decades=extend_decades,
                            jitter_dex=jitter_dex,
                            arrow_dex=arrow_dex, arrow_lw=arrow_lw,
                            cap_lw=cap_lw, cap_dex=cap_dex,
                            title_suffix='NED Photometry'
                        )
                    except Exception as _e:
                        if print_log:
                            tqdm.write(f"  ! SED plot failed for {name}: {_e}")

                parts.append(dfp)

            except Exception as e:
                if keep_not_found:
                    parts.append(pd.DataFrame([{'Name': name, 'Status': 'NOT_FOUND'}]))
                if print_log:
                    tqdm.write(f"  ! ERROR {name}/{t}: {e}")

        if parts:
            outpath = os.path.join(outdir, f"{t}.csv")
            pd.concat(parts, ignore_index=True).to_csv(outpath, index=False)
            saved[t] = outpath
            if print_log:
                print(f"[OK] Saved {t}: {outpath} (rows={sum(len(p) for p in parts)})")
        else:
            if print_log:
                print(f"[WARN] No rows for '{t}' → skip saving.")

    return saved

# =============== 示例入口（可改成你自己的 names/tables） ===============
if __name__ == "__main__":
    try:
        import load
        demo_names = load.read_upload('../5BZCAT_test.xlsx', sheet='Sheet3',
                                      columns={'name':'Name'})['name'].tolist()
    except Exception as e:
        print(f"[WARN] load.read_upload failed: {e}")
        demo_names = ['NGC 224', '5BZQJ0005+0524']

    demo_tables = ['photometry','positions','diameters']
    saved_files = fetch_ned_tables(
        names=demo_names,
        tables=demo_tables,
        outdir='./ned_out3',
        throttle_sec=1.0,
        random_throttle=True,
        timeout_sec=60.0,
        max_retries=1,
        keep_not_found=True,
        plot_phot_sed=True,
        sed_energy_unit='HzJy',   # 'HzJy' 或 'cgs'
        print_log=True
    )
    print("Saved:", saved_files)
