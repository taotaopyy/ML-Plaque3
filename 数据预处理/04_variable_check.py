"""
步骤 4：离散变量与连续变量检查
================================

读取步骤 3 输出的处理后训练集 ``data/train_processed.csv``，对每个特征进行：

1. 自动判别变量类型：
       - 二元离散：唯一非空取值 ≤ 2，且全部为整数 / 0–1 编码；
       - 多分类离散：唯一非空取值在 3 ~ ``MAX_CAT_LEVELS`` 之间且为整数 / 字符串；
       - 连续变量：其他数值列（含 float 与高基数 int）。
2. 连续变量描述：n、缺失率、mean、std、min、Q1、median、Q3、max、
   skew、kurtosis、正态性检验（Shapiro，n>5000 时退化为 Kolmogorov-Smirnov）；
   并与结局 ``target_lesion_positive`` 做单变量比较：
       - 正态：独立样本 t 检验；非正态：Mann-Whitney U。
3. 离散变量描述：n、缺失率、唯一值数、众数及其频数，以及每个取值的频数 /
   占比；与结局做 χ²（任一期望频数<5 时退化为 Fisher 精确检验）。

输入：data/train_processed.csv
输出：
    data/var_check_continuous.csv
    data/var_check_discrete.csv
    data/var_check_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_PATH = DATA_DIR / "train_processed.csv"
OUT_CONT = DATA_DIR / "var_check_continuous.csv"
OUT_DISC = DATA_DIR / "var_check_discrete.csv"
OUT_SUMMARY = DATA_DIR / "var_check_summary.txt"

TARGET_COL = "target_lesion_positive"
ID_COL = "accession_id"

MAX_CAT_LEVELS = 10           # 取值数 ≤ 此阈值视为离散
NORMALITY_ALPHA = 0.05        # 正态性显著性阈值


# ---------------------------------------------------------------------------
def classify_columns(df: pd.DataFrame, target: str, id_col: str) -> tuple[list[str], list[str]]:
    """返回 (continuous_cols, discrete_cols)，跳过 id 与结局列。"""
    continuous, discrete = [], []
    for c in df.columns:
        if c in (target, id_col):
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        nunique = s.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(s)
        looks_integer = is_numeric and np.allclose(s, s.round())
        if not is_numeric:
            discrete.append(c)
        elif nunique <= 2 or (looks_integer and nunique <= MAX_CAT_LEVELS):
            discrete.append(c)
        else:
            continuous.append(c)
    return continuous, discrete


# ---------------------------------------------------------------------------
def normality_test(s: pd.Series) -> tuple[float, str]:
    """返回 (p-value, 检验名称)。"""
    s = s.dropna()
    if len(s) < 8:
        return np.nan, "n_too_small"
    if len(s) <= 5000:
        try:
            return float(stats.shapiro(s).pvalue), "shapiro"
        except Exception:
            pass
    # 大样本退化为 KS 与正态分布对比
    mu, sigma = s.mean(), s.std(ddof=1)
    if sigma == 0:
        return np.nan, "ks_zero_std"
    return float(stats.kstest((s - mu) / sigma, "norm").pvalue), "ks"


def continuous_vs_target(s: pd.Series, y: pd.Series, normal: bool) -> tuple[float, str]:
    """连续变量与二分类结局的单变量比较。"""
    g0 = s[y == 0].dropna()
    g1 = s[y == 1].dropna()
    if len(g0) < 3 or len(g1) < 3:
        return np.nan, "n_too_small"
    if normal:
        return float(stats.ttest_ind(g0, g1, equal_var=False).pvalue), "ttest"
    return float(stats.mannwhitneyu(g0, g1, alternative="two-sided").pvalue), "mannwhitney"


# ---------------------------------------------------------------------------
def describe_continuous(df: pd.DataFrame, cols: list[str], y: pd.Series) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c]
        clean = s.dropna()
        norm_p, norm_test = normality_test(clean)
        is_normal = (not np.isnan(norm_p)) and norm_p > NORMALITY_ALPHA
        target_p, target_test = continuous_vs_target(s, y, is_normal)
        rows.append({
            "variable": c,
            "type": "continuous",
            "n": int(clean.size),
            "missing_rate": float(s.isna().mean()),
            "mean":   float(clean.mean()),
            "std":    float(clean.std(ddof=1)) if clean.size > 1 else np.nan,
            "min":    float(clean.min()),
            "q1":     float(clean.quantile(0.25)),
            "median": float(clean.median()),
            "q3":     float(clean.quantile(0.75)),
            "max":    float(clean.max()),
            "skew":     float(clean.skew())     if clean.size > 2 else np.nan,
            "kurtosis": float(clean.kurtosis()) if clean.size > 3 else np.nan,
            "normality_test": norm_test,
            "normality_p":    norm_p,
            "is_normal":      bool(is_normal),
            "target_test":  target_test,
            "target_p":     target_p,
            "target_signif": (not np.isnan(target_p)) and target_p < 0.05,
        })
    return pd.DataFrame(rows).sort_values("target_p", na_position="last")


# ---------------------------------------------------------------------------
def discrete_vs_target(s: pd.Series, y: pd.Series) -> tuple[float, str]:
    valid = s.notna() & y.notna()
    ct = pd.crosstab(s[valid], y[valid])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, "constant"
    try:
        chi2, p, _, expected = stats.chi2_contingency(ct)
        if (expected < 5).any():
            # 期望频数过小，2x2 用 Fisher 精确，更大用 chi2 的近似
            if ct.shape == (2, 2):
                _, p = stats.fisher_exact(ct.values)
                return float(p), "fisher"
            return float(p), "chi2_lowexp"
        return float(p), "chi2"
    except Exception:
        return np.nan, "failed"


def describe_discrete(df: pd.DataFrame, cols: list[str], y: pd.Series) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c]
        clean = s.dropna()
        vc = clean.value_counts(dropna=True)
        mode_val = vc.index[0] if not vc.empty else np.nan
        mode_freq = int(vc.iloc[0]) if not vc.empty else 0
        target_p, target_test = discrete_vs_target(s, y)
        rows.append({
            "variable": c,
            "type": "binary" if clean.nunique() <= 2 else "categorical",
            "n": int(clean.size),
            "missing_rate": float(s.isna().mean()),
            "n_unique": int(clean.nunique()),
            "mode": str(mode_val),
            "mode_freq": mode_freq,
            "mode_pct": float(mode_freq / clean.size) if clean.size else np.nan,
            "value_counts": "; ".join(f"{k}={v}" for k, v in vc.items()),
            "target_test":  target_test,
            "target_p":     target_p,
            "target_signif": (not np.isnan(target_p)) and target_p < 0.05,
        })
    return pd.DataFrame(rows).sort_values("target_p", na_position="last")


# ---------------------------------------------------------------------------
def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到 {INPUT_PATH}，请先运行步骤 1–3。")
    df = pd.read_csv(INPUT_PATH)
    if TARGET_COL not in df.columns:
        raise KeyError(f"找不到结局列 {TARGET_COL}")

    y = df[TARGET_COL].astype(int)
    continuous_cols, discrete_cols = classify_columns(df, TARGET_COL, ID_COL)
    print(f"训练集 {df.shape}")
    print(f"[type] 连续变量 {len(continuous_cols)} 个 | 离散变量 {len(discrete_cols)} 个")

    cont_df = describe_continuous(df, continuous_cols, y)
    disc_df = describe_discrete(df, discrete_cols, y)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cont_df.to_csv(OUT_CONT, index=False, encoding="utf-8-sig")
    disc_df.to_csv(OUT_DISC, index=False, encoding="utf-8-sig")

    summary_lines = []
    summary_lines.append(f"训练集形状: {df.shape}")
    summary_lines.append(f"结局阳性率: {y.mean():.2%}  阳性 {int(y.sum())} / 阴性 {int((y==0).sum())}")
    summary_lines.append("")
    summary_lines.append(f"连续变量数: {len(continuous_cols)}")
    summary_lines.append(f"  其中通过正态性检验 (p>{NORMALITY_ALPHA}): "
                         f"{int(cont_df['is_normal'].sum())}")
    summary_lines.append(f"  与结局 p<0.05 的连续变量数: "
                         f"{int(cont_df['target_signif'].sum())}")
    sig_cont = cont_df.loc[cont_df['target_signif'], ['variable', 'target_p', 'target_test']].head(20)
    if not sig_cont.empty:
        summary_lines.append("  Top 20 显著连续变量:")
        for _, r in sig_cont.iterrows():
            summary_lines.append(f"    {r['variable']:<45s} p={r['target_p']:.2e} ({r['target_test']})")
    summary_lines.append("")
    summary_lines.append(f"离散变量数: {len(discrete_cols)}")
    summary_lines.append(f"  与结局 p<0.05 的离散变量数: "
                         f"{int(disc_df['target_signif'].sum())}")
    sig_disc = disc_df.loc[disc_df['target_signif'], ['variable', 'target_p', 'target_test']].head(20)
    if not sig_disc.empty:
        summary_lines.append("  Top 20 显著离散变量:")
        for _, r in sig_disc.iterrows():
            summary_lines.append(f"    {r['variable']:<45s} p={r['target_p']:.2e} ({r['target_test']})")

    summary = "\n".join(summary_lines)
    OUT_SUMMARY.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print()
    print(f"[完成] 连续变量统计 -> {OUT_CONT}")
    print(f"[完成] 离散变量统计 -> {OUT_DISC}")
    print(f"[完成] 概要         -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
