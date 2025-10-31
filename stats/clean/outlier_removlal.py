# src/stats/clean/outlier_removal.py

import numpy as np
from typing import Literal
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt


def remove_outliers(
    data: np.ndarray,
    method: Literal["zscore", "iqr", "lof", "mad"] = "zscore",
    threshold: float = 3.0,
    return_mask: bool = False,
    lof_n_neighbors: int = 5,
    lof_contamination: float | str = 'auto'
) -> np.ndarray:
    """
    从一维数据中剔除异常值。

    参数：
        data: 原始数据（1D numpy 数组）
        method: 可选 "zscore", "iqr", "lof", "mad"
        threshold: 判定阈值（对于不同方法代表不同意义）
        return_mask: 若为 True，返回布尔掩码而非数据
        lof_n_neighbors: LOF 的邻居数（小于样本数）
        lof_contamination: LOF 中假设的异常比例（'auto' 或 0~1）

    返回：
        - 清洗后的数据（或掩码）
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Only 1D data supported")
    mask = np.full_like(data, fill_value=True, dtype=bool)

    if method == "zscore":
        mean = np.nanmean(data)
        std = np.nanstd(data)
        z = np.abs((data - mean) / (std + 1e-12))
        mask = z <= threshold

    elif method == "iqr":
        q1, q3 = np.nanpercentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (data >= lower) & (data <= upper)

    elif method == "mad":
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - median)) + 1e-12
        modified_z = 0.6745 * np.abs(data - median) / mad
        mask = modified_z <= threshold

    elif method == "lof":
        n = len(data)
        n_neighbors = min(lof_n_neighbors, n - 1)
        data_2d = data.reshape(-1, 1)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=lof_contamination
        )
        lof_fit = lof.fit_predict(data_2d)
        mask = lof_fit == 1

    else:
        raise ValueError("Unknown method. Use 'zscore', 'iqr', 'mad' or 'lof'.")

    return mask if return_mask else data[mask]


def plot_outliers(data: np.ndarray, mask: np.ndarray, title: str = "Outlier Detection"):
    """可视化数据与异常值"""
    data = np.asarray(data)
    mask = np.asarray(mask)
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(data))[mask], data[mask], label="Normal", alpha=0.8)
    ax.scatter(np.arange(len(data))[~mask], data[~mask], label="Outlier", c="red", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 示例
if __name__ == "__main__":
    x = np.array([1, 2, 2, 3, 100, 3, 2, 1, 5, 3])
    for method in ["zscore", "iqr", "mad", "lof"]:
        mask = remove_outliers(x, method=method, return_mask=True)
        print(f"方法: {method}, 清洗后数据:", x[mask])
        plot_outliers(x, mask, title=f"Outlier Detection: {method}")
