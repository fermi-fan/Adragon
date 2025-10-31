# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import optimize

@dataclass
class BoxCoxResult:
    lam: float
    shift: float    # 平移至正域的量
    y: np.ndarray   # 对“可观测值”变换后的数组（便于直接出图）

def _boxcox_core(xpos: np.ndarray, lam: float) -> np.ndarray:
    if abs(lam) < 1e-12:
        return np.log(xpos)
    return (np.power(xpos, lam) - 1.0) / lam

def fit_transform(x: np.ndarray, allow_shift: bool=True) -> BoxCoxResult:
    """
    为>0数据寻找 λ（以偏度/峰度目标近似正态最优），必要时做平移。
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("fit_transform needs finite data.")
    shift = 0.0
    if allow_shift:
        xmin = np.nanmin(x)
        if xmin <= 0:
            shift = (1.0 - xmin) + 1e-6
    x_pos = x + shift

    def objective(lam):
        y = _boxcox_core(x_pos, lam)
        y = (y - y.mean()) / (y.std(ddof=1) + 1e-12)
        skew = (y**3).mean()
        kurt = (y**4).mean() - 3.0
        return skew*skew + kurt*kurt

    lam_opt = optimize.brent(objective, brack=(-2.0, 2.0))
    y = _boxcox_core(x_pos, lam_opt)
    return BoxCoxResult(lam=lam_opt, shift=shift, y=y)

def transform_scalar(x: float, bc: BoxCoxResult) -> float:
    return _boxcox_core(np.array([x + bc.shift]), bc.lam)[0]

def inverse_transform(y: np.ndarray, bc: BoxCoxResult) -> np.ndarray:
    lam, shift = bc.lam, bc.shift
    if abs(lam) < 1e-12:
        x = np.exp(y)
    else:
        x = np.power(lam*y + 1.0, 1.0/lam)
    return x - shift


# ==== 示例开始 ====

import matplotlib.pyplot as plt
# 1. 生成偏态数据（对数正态分布）
np.random.seed(42)
x = np.random.lognormal(mean=1.5, sigma=0.4, size=1000)

# 2. 拟合 Box–Cox 变换
bc = fit_transform(x, allow_shift=True)
x_bc = bc.y

# 3. 可视化原始 vs 变换后数据
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(x, bins=30, alpha=0.7, color='steelblue')
ax[0].set_title("Original Data")
ax[1].hist(x_bc, bins=30, alpha=0.7, color='tomato')
ax[1].set_title("Box–Cox Transformed")
fig.suptitle(f"Box–Cox λ = {bc.lam:.3f}, shift = {bc.shift:.3f}")
fig.tight_layout()
plt.show()

# 4. 验证逆变换的精度
x_reconstructed = inverse_transform(x_bc, bc)
x_aligned = x + bc.shift
error = np.abs(x_aligned - x_reconstructed)
print(f"Max error: {np.max(error):.2e}")
print(f"Mean error: {np.mean(error):.2e}")