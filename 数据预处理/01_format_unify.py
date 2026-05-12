"""
步骤 1：统一格式与编码一致化
================================

功能：
1. 单位统一：基于常见检验项目的临床参考范围，自动识别并转换
   不同中心 / 不同时期可能存在的单位差异（如肌酐 mg/dL ↔ μmol/L、
   总胆固醇 mg/dL ↔ mmol/L 等）。
2. 日期格式化：自动识别日期型列并统一为 YYYY-MM-DD 字符串。
3. 基础分类变量硬编码：将已知且固定的二元分类变量
   （性别、高血压史、糖尿病史、吸烟史、PCI 史、CABG 史等）
   统一映射为 1 / 0；同时清理列名两端空白、剔除完全为空的列。

输入：all_variable.csv
输出：data/all_variable_step1.csv
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "all_variable.csv"
OUTPUT_DIR = ROOT / "data"
OUTPUT_PATH = OUTPUT_DIR / "all_variable_step1.csv"


# ---------------------------------------------------------------------------
# 1. 单位统一
# ---------------------------------------------------------------------------
# 中位数若处于"另一单位"的合理量级，则视为单位不一致，整列做换算。
# (factor, expected_low, expected_high) ：当中位数 < expected_low 或
# > expected_high 时，整列乘以 factor 进行换算。
UNIT_RULES = {
    # 肌酐：µmol/L（常见 40–110） ↔ mg/dL（0.5–1.4）。mg/dL × 88.4 ≈ µmol/L
    "baseline_creatinine": {"factor": 88.4, "low": 20, "high": 800},
    "followup_creatinine": {"factor": 88.4, "low": 20, "high": 800},
    # 尿酸：µmol/L（200–450） ↔ mg/dL（3–7）。mg/dL × 59.48 ≈ µmol/L
    "baseline_uric_acid":  {"factor": 59.48, "low": 80, "high": 800},
    "followup_uric_acid":  {"factor": 59.48, "low": 80, "high": 800},
    # 总胆固醇：mmol/L（3–6） ↔ mg/dL（120–240）。mg/dL / 38.67 ≈ mmol/L
    "baseline_total_cholesterol": {"factor": 1 / 38.67, "low": 1.5, "high": 12},
    "followup_total_cholesterol": {"factor": 1 / 38.67, "low": 1.5, "high": 12},
    # LDL：mmol/L（1–5） ↔ mg/dL（40–200）。mg/dL / 38.67 ≈ mmol/L
    "baseline_ldl": {"factor": 1 / 38.67, "low": 0.3, "high": 10},
    "followup_ldl": {"factor": 1 / 38.67, "low": 0.3, "high": 10},
    # HDL：mmol/L（0.5–2.5） ↔ mg/dL（20–100）
    "baseline_hdl_cholesterol": {"factor": 1 / 38.67, "low": 0.2, "high": 5},
    "followup_hdl_cholesterol": {"factor": 1 / 38.67, "low": 0.2, "high": 5},
    # 甘油三酯：mmol/L（0.5–4） ↔ mg/dL（50–400）。mg/dL / 88.57 ≈ mmol/L
    "baseline_triglycerides": {"factor": 1 / 88.57, "low": 0.2, "high": 15},
    "followup_triglycerides": {"factor": 1 / 88.57, "low": 0.2, "high": 15},
}


def unify_units(df: pd.DataFrame) -> pd.DataFrame:
    """根据中位数所处量级判断是否需要换算单位。"""
    for col, rule in UNIT_RULES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.dropna().empty:
            continue
        median = series.median()
        if median < rule["low"] or median > rule["high"]:
            df[col] = series * rule["factor"]
            print(f"[unit] {col}: 中位数={median:g} -> ×{rule['factor']:.4g}")
        else:
            df[col] = series
    return df


# ---------------------------------------------------------------------------
# 2. 日期格式化
# ---------------------------------------------------------------------------
DATE_NAME_HINT = re.compile(r"(date|time|日期|时间)", re.IGNORECASE)


def _looks_like_date(series: pd.Series) -> bool:
    """轻量级判断一个 object 列是否包含可解析的日期字符串。"""
    if series.dtype != "object":
        return False
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.8


def format_dates(df: pd.DataFrame) -> pd.DataFrame:
    """识别日期列并统一格式为 YYYY-MM-DD。"""
    for col in df.columns:
        if DATE_NAME_HINT.search(col) or _looks_like_date(df[col]):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                df[col] = parsed.dt.strftime("%Y-%m-%d")
                print(f"[date] {col} -> YYYY-MM-DD")
    return df


# ---------------------------------------------------------------------------
# 3. 基础分类变量硬编码
# ---------------------------------------------------------------------------
SEX_MAP = {
    "男": 1, "M": 1, "Male": 1, "male": 1, 1: 1, "1": 1,
    "女": 0, "F": 0, "Female": 0, "female": 0, 0: 0, "0": 0,
}

YESNO_MAP = {
    "有": 1, "是": 1, "Y": 1, "y": 1, "Yes": 1, "yes": 1, "True": 1, True: 1, 1: 1, "1": 1,
    "无": 0, "否": 0, "N": 0, "n": 0, "No":  0, "no":  0, "False": 0, False: 0, 0: 0, "0": 0,
}

BINARY_COLUMNS = [
    "sex",
    "history_hypertension",
    "history_diabetes_new",
    "history_smoking",
    "history_pci",
    "history_cabg",
    "target_lesion_positive",
    "positive_remodeling_flag",
    "low_attenuation_plaque_flag",
    "napkin_ring_sign_flag",
    "spotty_calcification_flag",
    "High-Risk Plaque Flag",
    "pcat_hu_high_risk_flag",
    "新发糖尿病",
    "history_diabetes_new_py",
]


def _map_binary(series: pd.Series, mapping: dict) -> pd.Series:
    """将文本/数值映射为 0/1，保留缺失值。"""
    def _convert(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, str):
            v = v.strip()
        return mapping.get(v, np.nan)
    return series.map(_convert).astype("Float64")


def encode_basic_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """硬编码已知的二元分类变量。"""
    if "sex" in df.columns:
        df["sex"] = _map_binary(df["sex"], SEX_MAP)
        print("[encode] sex: 男=1, 女=0")

    for col in BINARY_COLUMNS:
        if col == "sex" or col not in df.columns:
            continue
        df[col] = _map_binary(df[col], YESNO_MAP)
        print(f"[encode] {col}: 有/是=1, 无/否=0")

    return df


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """剔除完全为空的列、去除列名首尾空白。"""
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    print(f"原始数据：{df.shape[0]} 行 × {df.shape[1]} 列")

    df = clean_columns(df)
    df = unify_units(df)
    df = format_dates(df)
    df = encode_basic_categoricals(df)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[完成] 写出 {OUTPUT_PATH}  ({df.shape[0]} × {df.shape[1]})")


if __name__ == "__main__":
    main()
