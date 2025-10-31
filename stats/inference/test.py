# src/stats/inference/truncnorm_test.py

import numpy as np
from scipy.stats import truncnorm, kstest
from typing import Literal, Dict


def evaluate_truncnorm_fit(
    data: np.ndarray,
    mu: float,
    sigma: float,
    a: float,
    b: float,
    B: int = 1000,
    alpha: float = 0.05,
    method: Literal["ks", "cvm"] = "ks",
    random_state: int = 42,
) -> Dict[str, float | bool]:
    """
    用 bootstrap 和朴素方法检验截断正态分布拟合质量。

    参数：
        data: 观测数据（一维数组，截断在 [a, b] 内）
        mu, sigma: 拟合得到的参数
        a, b: 截断区间
        B: bootstrap 重采样次数
        alpha: 显著性水平
        method: "ks"（Kolmogorov–Smirnov）或 "cvm"（Cramér–von Mises）
        random_state: 随机种子

    返回：
        dict，包括检验统计量、bootstrap p 值、朴素 p 值、是否拒绝拟合假设
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    n = data.size
    if n < 5:
        raise ValueError("Insufficient data for testing")

    # 构造理论截断正态分布
    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma
    dist = truncnorm(a_std, b_std, loc=mu, scale=sigma)

    # 排序后的观测值和理论 CDF
    x_sorted = np.sort(data)
    cdf_obs = dist.cdf(x_sorted)
    ecdf = np.arange(1, n + 1) / n

    if method == "ks":
        stat_obs = np.max(np.abs(ecdf - cdf_obs))
        # 朴素 KS p 值（仅供参考）
        p_naive = kstest(data, dist.cdf).pvalue
    elif method == "cvm":
        stat_obs = np.sum((cdf_obs - ecdf) ** 2) + 1 / (12 * n)
        p_naive = np.nan  # 无现成实现
    else:
        raise ValueError("Unknown method. Use 'ks' or 'cvm'.")

    # bootstrap 样本 KS/CvM 分布
    rng = np.random.default_rng(random_state)
    stats_boot = []
    for _ in range(B):
        sample = dist.rvs(size=n, random_state=rng)
        sample.sort()
        cdf_sample = dist.cdf(sample)
        ecdf_sample = np.arange(1, n + 1) / n
        if method == "ks":
            stat = np.max(np.abs(ecdf_sample - cdf_sample))
        else:  # cvm
            stat = np.sum((cdf_sample - ecdf_sample) ** 2) + 1 / (12 * n)
        stats_boot.append(stat)

    stats_boot = np.array(stats_boot)
    p_boot = (np.sum(stats_boot >= stat_obs) + 1) / (B + 1)
    reject = p_boot < alpha

    return {
        "method": method.upper(),
        "statistic": round(stat_obs, 5),
        "p_bootstrap": round(p_boot, 5),
        "p_naive": round(p_naive, 5) if not np.isnan(p_naive) else None,
        "alpha": alpha,
        "reject_null": reject
    }


# 示例用法（可删）
if __name__ == "__main__":
    np.random.seed(0)
    true_mu, true_sigma = 5, 1
    a, b = 2, 8
    n = 500
    data = truncnorm.rvs((a - true_mu)/true_sigma, (b - true_mu)/true_sigma, loc=true_mu, scale=true_sigma, size=n)

    print("==== KS 检验 ====")
    ks_result = evaluate_truncnorm_fit(data, mu=5.0, sigma=1.0, a=a, b=b, B=500, method="ks")
    for k, v in ks_result.items():
        print(f"{k:>16}: {v}")

    print("\n==== CvM 检验 ====")
    cvm_result = evaluate_truncnorm_fit(data, mu=5.0, sigma=1.0, a=a, b=b, B=500, method="cvm")
    for k, v in cvm_result.items():
        print(f"{k:>16}: {v}")
