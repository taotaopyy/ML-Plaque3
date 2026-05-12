"""
步骤 3：基于训练集的局部预处理
================================

所有"需要学习参数"的预处理操作严格只在训练集上拟合（fit），然后用同一组
参数应用（transform）到测试集，以避免信息泄漏。

具体包括：
1. 异常值检测与处理：在训练集上用 Tukey IQR 法计算连续变量的上下限
       lower = Q1 - 1.5 * IQR
       upper = Q3 + 1.5 * IQR
   对训练集和测试集采用同一组上下限做 winsorize（截断到边界）。
2. 缺失值评估与插补：
       - 剔除训练集缺失率 > 30% 的特征（同步从测试集删除）；
       - 数值列用训练集中位数插补；
       - 类别列用训练集众数插补。
3. 数据分布转换：
       - 训练集中绝对偏度 |skew| > 1 且全部非负的连续变量执行 log1p 转换；
       - 同样的变量列表同步应用到测试集。
4. 多分类变量 one-hot 编码：
       - 在训练集上拟合编码器（保留全部类别），编码器同步用于测试集；
       - 二元 0/1 变量不再做 one-hot。

输入：
    data/train.csv
    data/test.csv

输出：
    data/train_processed.csv
    data/test_processed.csv
    data/preprocess_params.json   （保存所有拟合得到的参数，便于审计 / 复现）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_IN = DATA_DIR / "train.csv"
TEST_IN  = DATA_DIR / "test.csv"
TRAIN_OUT = DATA_DIR / "train_processed.csv"
TEST_OUT  = DATA_DIR / "test_processed.csv"
PARAMS_OUT = DATA_DIR / "preprocess_params.json"

TARGET_COL = "target_lesion_positive"
ID_COL = "accession_id"

# 已被脚本 01 硬编码为 0/1 的二元变量，不再视为多分类
BINARY_COLUMNS = {
    "sex", "history_hypertension", "history_diabetes_new", "history_smoking",
    "history_pci", "history_cabg", "target_lesion_positive",
    "positive_remodeling_flag", "low_attenuation_plaque_flag",
    "napkin_ring_sign_flag", "spotty_calcification_flag",
    "High-Risk Plaque Flag", "pcat_hu_high_risk_flag",
    "新发糖尿病", "history_diabetes_new_py",
}

# 已知的多分类变量（文本或多档位数值）——这些将做 one-hot
MULTICLASS_COLUMNS = [
    "artery_vessel", "lesion_segment", "contact_segment",
    "lesion_stenosis_grade", "plaque_type",
    "ffrct_risk_class", "plaque_feature_count",
    "lipid_response_grade", "baseline_ldl_cat",
]

MISSING_THRESHOLD = 0.30   # 缺失率 > 30% 的特征剔除
SKEW_THRESHOLD = 1.0       # |skew| > 1 视为严重偏态


# ---------------------------------------------------------------------------
def drop_high_missing(
    train: pd.DataFrame, test: pd.DataFrame, threshold: float, protect: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    miss_rate = train.isna().mean()
    dropped = [c for c in miss_rate.index
               if miss_rate[c] > threshold and c not in protect]
    if dropped:
        print(f"[missing] 剔除缺失率 >{threshold:.0%} 的列 {len(dropped)} 个：{dropped}")
    train = train.drop(columns=dropped)
    test  = test.drop(columns=[c for c in dropped if c in test.columns])
    return train, test, dropped


# ---------------------------------------------------------------------------
def iqr_bounds(train: pd.DataFrame, numeric_cols: list[str]) -> dict[str, dict[str, float]]:
    bounds = {}
    for c in numeric_cols:
        s = train[c].dropna()
        if len(s) < 10 or s.nunique() <= 2:
            continue
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        bounds[c] = {"lower": float(q1 - 1.5 * iqr),
                     "upper": float(q3 + 1.5 * iqr)}
    return bounds


def winsorize(df: pd.DataFrame, bounds: dict[str, dict[str, float]]) -> pd.DataFrame:
    for c, b in bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(lower=b["lower"], upper=b["upper"])
    return df


# ---------------------------------------------------------------------------
def fit_impute(
    train: pd.DataFrame, numeric_cols: list[str], cat_cols: list[str]
) -> dict[str, float | str]:
    params: dict[str, float | str] = {}
    for c in numeric_cols:
        if c in train.columns:
            params[c] = float(train[c].median())
    for c in cat_cols:
        if c in train.columns:
            mode = train[c].mode(dropna=True)
            if not mode.empty:
                params[c] = mode.iloc[0]
    return params


def apply_impute(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    for c, v in params.items():
        if c in df.columns:
            df[c] = df[c].fillna(v)
    return df


# ---------------------------------------------------------------------------
def select_skewed_cols(train: pd.DataFrame, numeric_cols: list[str], threshold: float) -> list[str]:
    skewed = []
    for c in numeric_cols:
        s = train[c].dropna()
        if len(s) < 30:
            continue
        if (s < 0).any():
            continue  # log 转换要求非负
        if abs(s.skew()) > threshold:
            skewed.append(c)
    return skewed


def apply_log1p(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))
    return df


# ---------------------------------------------------------------------------
def fit_onehot(train: pd.DataFrame, cols: list[str]) -> tuple[OneHotEncoder, list[str]]:
    use_cols = [c for c in cols if c in train.columns]
    if not use_cols:
        return None, []
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(train[use_cols].astype(str).fillna("NA"))
    return enc, use_cols


def apply_onehot(df: pd.DataFrame, enc: OneHotEncoder, cols: list[str]) -> pd.DataFrame:
    if enc is None or not cols:
        return df
    mat = enc.transform(df[cols].astype(str).fillna("NA"))
    new_names = enc.get_feature_names_out(cols)
    onehot_df = pd.DataFrame(mat, columns=new_names, index=df.index)
    df = df.drop(columns=cols).reset_index(drop=True)
    onehot_df = onehot_df.reset_index(drop=True)
    return pd.concat([df, onehot_df], axis=1)


# ---------------------------------------------------------------------------
def main() -> None:
    train = pd.read_csv(TRAIN_IN)
    test  = pd.read_csv(TEST_IN)
    print(f"训练集: {train.shape}  测试集: {test.shape}")

    protected = {TARGET_COL, ID_COL}

    # 2.1 缺失率筛选 -------------------------------------------------------
    train, test, dropped_cols = drop_high_missing(train, test, MISSING_THRESHOLD, protected)

    # 区分数值列 / 类别列（依据 dtype 与已知名单）
    multiclass_cols = [c for c in MULTICLASS_COLUMNS if c in train.columns]
    numeric_cols = [c for c in train.columns
                    if c not in protected
                    and c not in multiclass_cols
                    and pd.api.types.is_numeric_dtype(train[c])]
    cat_cols = [c for c in train.columns
                if c not in protected
                and c not in numeric_cols
                and c not in multiclass_cols]   # 剩下的字符列若有，则按类别填众数

    print(f"[cols] 数值变量 {len(numeric_cols)}  多分类变量 {len(multiclass_cols)}  "
          f"其他类别变量 {len(cat_cols)}")

    # 1. 异常值检测 -------------------------------------------------------
    bounds = iqr_bounds(train, numeric_cols)
    print(f"[outlier] 在训练集上为 {len(bounds)} 个连续变量学习 IQR 上下限")
    train = winsorize(train, bounds)
    test  = winsorize(test,  bounds)

    # 2.2 缺失值插补 -----------------------------------------------------
    impute_params = fit_impute(train, numeric_cols, multiclass_cols + cat_cols)
    print(f"[impute] 训练集上拟合 {len(impute_params)} 个特征的插补值（数值=中位数，类别=众数）")
    train = apply_impute(train, impute_params)
    test  = apply_impute(test,  impute_params)

    # 3. 偏态变量 log 转换 -----------------------------------------------
    skewed_cols = select_skewed_cols(train, numeric_cols, SKEW_THRESHOLD)
    print(f"[skew] |skew|>{SKEW_THRESHOLD} 且非负的连续变量 {len(skewed_cols)} 个，做 log1p 转换")
    train = apply_log1p(train, skewed_cols)
    test  = apply_log1p(test,  skewed_cols)

    # 4. 多分类 one-hot --------------------------------------------------
    encoder, oh_cols = fit_onehot(train, multiclass_cols)
    print(f"[onehot] 对 {len(oh_cols)} 个多分类变量做 one-hot：{oh_cols}")
    train = apply_onehot(train, encoder, oh_cols)
    test  = apply_onehot(test,  encoder, oh_cols)

    # 对齐列（双重保险，防止测试集出现训练集没有的列）
    test = test.reindex(columns=train.columns, fill_value=0)

    # 输出 ------------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
    test.to_csv(TEST_OUT,  index=False, encoding="utf-8-sig")
    print(f"\n[完成] 训练集 -> {TRAIN_OUT}  shape={train.shape}")
    print(f"[完成] 测试集 -> {TEST_OUT}   shape={test.shape}")

    # 参数审计
    params = {
        "dropped_high_missing": dropped_cols,
        "iqr_bounds": bounds,
        "impute_values": {k: (v if isinstance(v, (int, float, str)) else str(v))
                          for k, v in impute_params.items()},
        "log1p_columns": skewed_cols,
        "onehot_columns": oh_cols,
        "onehot_categories": (
            {c: list(map(str, cats)) for c, cats in zip(oh_cols, encoder.categories_)}
            if encoder is not None else {}
        ),
    }
    with open(PARAMS_OUT, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"[完成] 参数记录 -> {PARAMS_OUT}")


if __name__ == "__main__":
    main()
