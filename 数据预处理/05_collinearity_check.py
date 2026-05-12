"""
步骤 5：共线性检查
================================

读取步骤 3 输出的处理后训练集 ``data/train_processed.csv``，对特征矩阵做：

1. **相关性矩阵**：连续 + 离散数值化后的全部特征计算 Pearson 与 Spearman
   相关系数，导出完整矩阵以及 ``|r| ≥ HIGH_CORR_THRESHOLD`` 的高共线性变量对。
2. **方差膨胀因子 (VIF)**：仅在连续变量上计算，
       VIF_i = 1 / (1 - R_i^2)
   其中 R_i^2 由其余连续变量回归 X_i 得到。VIF > 10 通常视为存在严重共线性。
3. **降维候选清单**：对每对高相关变量，保留与结局相关性更强的一个，
   将另一变量记入 ``drop_candidates``；并把 VIF > 10 的变量也加入候选。
   注：清单仅作建议，不会自动删除任何特征，需研究者结合临床意义复核。

输入：data/train_processed.csv
输出：
    data/collinearity_pearson.csv     完整 Pearson 矩阵
    data/collinearity_spearman.csv    完整 Spearman 矩阵
    data/collinearity_high_pairs.csv  高相关变量对
    data/collinearity_vif.csv         连续变量 VIF
    data/collinearity_drop_candidates.csv  建议删除候选
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_PATH = DATA_DIR / "train_processed.csv"

OUT_PEARSON  = DATA_DIR / "collinearity_pearson.csv"
OUT_SPEARMAN = DATA_DIR / "collinearity_spearman.csv"
OUT_PAIRS    = DATA_DIR / "collinearity_high_pairs.csv"
OUT_VIF      = DATA_DIR / "collinearity_vif.csv"
OUT_DROP     = DATA_DIR / "collinearity_drop_candidates.csv"

TARGET_COL = "target_lesion_positive"
ID_COL = "accession_id"

HIGH_CORR_THRESHOLD = 0.80
VIF_THRESHOLD = 10.0
MAX_CAT_LEVELS = 10  # 与脚本 04 保持一致


# ---------------------------------------------------------------------------
def classify_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    continuous, discrete = [], []
    for c in df.columns:
        if c in (TARGET_COL, ID_COL):
            continue
        s = df[c].dropna()
        if s.empty or not pd.api.types.is_numeric_dtype(s):
            if not s.empty:
                discrete.append(c)
            continue
        nunique = s.nunique()
        looks_integer = np.allclose(s, s.round())
        if nunique <= 2 or (looks_integer and nunique <= MAX_CAT_LEVELS):
            discrete.append(c)
        else:
            continuous.append(c)
    return continuous, discrete


# ---------------------------------------------------------------------------
def high_corr_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """提取上三角中 |r| ≥ threshold 的变量对。"""
    cols = corr.columns
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    triu = corr.where(mask)
    pairs = (
        triu.stack()
            .reset_index()
            .rename(columns={"level_0": "var1", "level_1": "var2", 0: "r"})
    )
    pairs["abs_r"] = pairs["r"].abs()
    pairs = pairs[pairs["abs_r"] >= threshold].sort_values("abs_r", ascending=False)
    return pairs.reset_index(drop=True)


# ---------------------------------------------------------------------------
def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """连续变量 VIF。会自动剔除完全共线（VIF=inf）的列。"""
    X = df[cols].dropna()
    if X.empty or len(cols) < 2:
        return pd.DataFrame(columns=["variable", "VIF"])
    X_c = add_constant(X, has_constant="add")
    rows = []
    for i, name in enumerate(X_c.columns):
        if name == "const":
            continue
        try:
            vif = variance_inflation_factor(X_c.values, i)
        except Exception:
            vif = np.nan
        rows.append({"variable": name, "VIF": float(vif)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


# ---------------------------------------------------------------------------
def build_drop_candidates(pairs: pd.DataFrame, target_corr: pd.Series,
                          vif_df: pd.DataFrame) -> pd.DataFrame:
    """对每对高相关变量：保留与结局 |corr| 更大的一个，记录另一个为可删候选。
    VIF > 阈值的变量也加入候选。"""
    drops: dict[str, str] = {}
    for _, row in pairs.iterrows():
        v1, v2 = row["var1"], row["var2"]
        c1 = abs(target_corr.get(v1, 0))
        c2 = abs(target_corr.get(v2, 0))
        loser = v2 if c1 >= c2 else v1
        keeper = v1 if loser == v2 else v2
        if loser not in drops:
            drops[loser] = f"与 {keeper} 高度相关 (|r|={row['abs_r']:.3f})，且 |corr_with_y| 较小"
    if not vif_df.empty:
        for _, row in vif_df.iterrows():
            if row["VIF"] > VIF_THRESHOLD and row["variable"] not in drops:
                drops[row["variable"]] = f"VIF={row['VIF']:.2f} (>{VIF_THRESHOLD})"
    return pd.DataFrame(
        [{"variable": k, "reason": v} for k, v in drops.items()]
    ).sort_values("variable")


# ---------------------------------------------------------------------------
def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到 {INPUT_PATH}，请先运行步骤 1–3。")
    df = pd.read_csv(INPUT_PATH)
    if TARGET_COL not in df.columns:
        raise KeyError(f"找不到结局列 {TARGET_COL}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    continuous, discrete = classify_columns(df)
    feature_cols = continuous + [c for c in discrete if pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in feature_cols if c not in (TARGET_COL, ID_COL)]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    print(f"训练集 {df.shape} | 连续 {len(continuous)} | 离散数值 {len(feature_cols)-len(continuous)}")

    # 1. 相关性矩阵 ------------------------------------------------------
    pearson  = X.corr(method="pearson")
    spearman = X.corr(method="spearman")
    pearson.to_csv(OUT_PEARSON,  encoding="utf-8-sig")
    spearman.to_csv(OUT_SPEARMAN, encoding="utf-8-sig")
    print(f"[corr] Pearson / Spearman 矩阵已输出 ({pearson.shape[0]}×{pearson.shape[1]})")

    pairs = high_corr_pairs(spearman, HIGH_CORR_THRESHOLD)
    pairs.to_csv(OUT_PAIRS, index=False, encoding="utf-8-sig")
    print(f"[corr] |Spearman r| ≥ {HIGH_CORR_THRESHOLD} 的变量对：{len(pairs)} 组")
    if not pairs.empty:
        print(pairs.head(15).to_string(index=False))

    # 2. VIF ------------------------------------------------------------
    vif = compute_vif(df, continuous)
    vif.to_csv(OUT_VIF, index=False, encoding="utf-8-sig")
    print(f"\n[VIF] 连续变量 VIF 计算完成 ({len(vif)})；VIF>{VIF_THRESHOLD} 的有 "
          f"{int((vif['VIF'] > VIF_THRESHOLD).sum())} 个")
    if not vif.empty:
        print(vif.head(15).to_string(index=False))

    # 3. 删除候选 -------------------------------------------------------
    y = df[TARGET_COL].astype(float)
    target_corr = X.apply(lambda s: s.corr(y))
    drop_df = build_drop_candidates(pairs, target_corr, vif)
    drop_df.to_csv(OUT_DROP, index=False, encoding="utf-8-sig")
    print(f"\n[drop] 共线性导致的候选删除变量：{len(drop_df)} 个")
    if not drop_df.empty:
        print(drop_df.head(20).to_string(index=False))

    print()
    print(f"[完成] Pearson 矩阵      -> {OUT_PEARSON}")
    print(f"[完成] Spearman 矩阵     -> {OUT_SPEARMAN}")
    print(f"[完成] 高相关变量对      -> {OUT_PAIRS}")
    print(f"[完成] VIF 表            -> {OUT_VIF}")
    print(f"[完成] 删除候选清单      -> {OUT_DROP}")


if __name__ == "__main__":
    main()
