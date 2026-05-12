"""
建模 - 步骤 1：特征工程
================================

按论文 / 表格定义构建建模所需的特征表，包括：

A. **固定输入变量** (fixed)：始终进入最终模型；
B. **RFE 候选变量** (rfe_candidates)：交由后续 RFE / RFECV 自动筛选；
C. **衍生特征**：
   1) 基于 4 个高危斑块标志（positive_remodeling / low_attenuation /
      napkin_ring / spotty_calcification），生成
        extreme_risk_ge2 / extreme_risk_ge3 / extreme_risk_ge4，
      用于定义"极高危斑块"。
   2) 基于 ffrct_value 设界，三分类（>0.8=低危=0, [0.7,0.8)=中=1, <0.7=高=2）
      → ffrct_class3。
   3) 基于 PCAT 平均 HU 设界 −70：mean_hu_lt_minus70（<-70 = 1，否则 0）。

注：原始数据中部分列名与论文/表格不完全一致，本脚本做了如下映射：
  - history_diabetes        ->  history_diabetes_new
  - mean_hu                 ->  mean_hu（pcat_hu）
  - maximum_luminal_area    ->  minimum_luminal_area_mm3
  - followup_triglycerides  ->  diff_triglycerides     （表格备注：仅作为基线差值）
  - followup_lipoprotein_a  ->  diff_lipoprotein_a     （同上）
原表中不存在的 ``cda_admission_number`` / ``stent_present`` 自动跳过并提示。

输入：data/all_variable_step1.csv  （步骤 1 单位/编码统一后的数据）
输出：
  data/features_engineered.csv
  data/feature_manifest.json   —— 记录最终用到的固定 / RFE 候选 / 衍生特征
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_PATH = DATA_DIR / "all_variable_step1.csv"
OUT_FEATURES = DATA_DIR / "features_engineered.csv"
OUT_MANIFEST = DATA_DIR / "feature_manifest.json"

TARGET_COL = "target_lesion_positive"
ID_COL = "accession_id"


# ---------------------------------------------------------------------------
# 列名映射：表格名 -> 数据中实际列名
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "history_diabetes":          "history_diabetes_new",
    "mean_hu":                   "mean_hu（pcat_hu）",
    "maximum_luminal_area_mm3":  "minimum_luminal_area_mm3",
    "followup_triglycerides":    "diff_triglycerides",
    "followup_lipoprotein_a":    "diff_lipoprotein_a",
}

# 表格里有但数据中不存在的列（会跳过并打印提示）
KNOWN_MISSING = {"cda_admission_number", "stent_present"}

FIXED_FEATURES_RAW = [
    "history_diabetes",       # -> history_diabetes_new
    "history_smoking",
    "baseline_triglycerides",
    "baseline_lipoprotein_a",
    "baseline_hba1c",
    "positive_remodeling_flag",
    "low_attenuation_plaque_flag",
    "napkin_ring_sign_flag",
    "spotty_calcification_flag",
    "plaque_feature_count",
    "High-Risk Plaque Flag",
    "contact_volume_mm3",
    "ffrct_value",
    "ffrct_risk_class",
]

RFE_CANDIDATES_RAW = [
    "history_cabg",
    "cda_admission_number",
    "baseline_ctnt", "baseline_nt_pro_bnp", "baseline_creatinine",
    "baseline_uric_acid", "baseline_ck", "baseline_ck_mb",
    "baseline_crp", "baseline_albumin", "baseline_alt",
    "baseline_wbc", "baseline_neutrophil_pct", "baseline_lymphocyte_pct",
    "baseline_platelet_count", "baseline_pdw",
    "baseline_total_cholesterol", "baseline_hdl_cholesterol",
    "baseline_lvef",
    "followup_hdl_cholesterol",
    "followup_triglycerides",   # -> diff_triglycerides
    "followup_lipoprotein_a",   # -> diff_lipoprotein_a
    "followup_hba1c",
    "plaque_type",              # 多分类，会在建模阶段 one-hot
    "stent_present",
    "stenosis_percent",
    "maximum_luminal_area_mm3", # -> minimum_luminal_area_mm3
    "maximum_diameter_stenosis_percent",
    "calcified_volume_mm3", "calcified_volume_ratio",
    "non_calcified_volume_mm3",
    "low_attenuation_volume_mm3", "low_attenuation_volume_ratio",
    "fibrous_fatty_volume_mm3", "fibrous_fatty_volume_ratio",
    "fibrotic_volume_mm3",
    "whole_lesion_volume_mm3",
    "lumen_volume_mm3", "vessel_volume_mm3",
    "mean_hu",                  # -> mean_hu（pcat_hu）
    "std_hu",
]

HIGH_RISK_FLAGS = [
    "positive_remodeling_flag",
    "low_attenuation_plaque_flag",
    "napkin_ring_sign_flag",
    "spotty_calcification_flag",
]


# ---------------------------------------------------------------------------
def resolve(name: str, df: pd.DataFrame) -> str | None:
    """把表格中的名字解析为数据中的实际列名；不存在则返回 None。"""
    actual = COLUMN_ALIASES.get(name, name)
    if actual in df.columns:
        return actual
    if name in df.columns:  # 兜底
        return name
    return None


def derive_extreme_risk(df: pd.DataFrame) -> pd.DataFrame:
    """基于 4 个高危标志构造 ≥2 / ≥3 / ≥4 三个二值特征。"""
    flag_cols = [c for c in HIGH_RISK_FLAGS if c in df.columns]
    if len(flag_cols) < 2:
        print("[warn] 高危斑块标志列不足，跳过 extreme_risk_* 衍生")
        return df
    cnt = df[flag_cols].fillna(0).sum(axis=1)
    df["extreme_risk_ge2"] = (cnt >= 2).astype(int)
    df["extreme_risk_ge3"] = (cnt >= 3).astype(int)
    df["extreme_risk_ge4"] = (cnt >= 4).astype(int)
    print(f"[derive] extreme_risk_ge2/3/4 已生成（基于 {len(flag_cols)} 个标志）")
    print(f"         ge2 阳性率 {df['extreme_risk_ge2'].mean():.2%} | "
          f"ge3 {df['extreme_risk_ge3'].mean():.2%} | "
          f"ge4 {df['extreme_risk_ge4'].mean():.2%}")
    return df


def derive_ffrct_class3(df: pd.DataFrame) -> pd.DataFrame:
    """ffrct 三分类：>=0.8 (0=低危), [0.7,0.8) (1=中), <0.7 (2=高)。"""
    if "ffrct_value" not in df.columns:
        return df
    v = df["ffrct_value"]
    cls = pd.Series(np.where(v >= 0.8, 0, np.where(v >= 0.7, 1, 2)), index=df.index)
    cls = cls.where(v.notna())
    df["ffrct_class3"] = cls
    print(f"[derive] ffrct_class3 已生成（>=0.8=0, [0.7,0.8)=1, <0.7=2）"
          f"  分布：{df['ffrct_class3'].value_counts(dropna=False).to_dict()}")
    return df


def derive_mean_hu_cat(df: pd.DataFrame) -> pd.DataFrame:
    """PCAT mean_hu < -70 → 高危 = 1。"""
    src = COLUMN_ALIASES.get("mean_hu", "mean_hu")
    if src not in df.columns:
        return df
    df["mean_hu_lt_minus70"] = (df[src] < -70).astype(int)
    df["mean_hu_lt_minus70"] = df["mean_hu_lt_minus70"].where(df[src].notna())
    print(f"[derive] mean_hu_lt_minus70 已生成 "
          f"（阳性率 {df['mean_hu_lt_minus70'].mean():.2%}）")
    return df


# ---------------------------------------------------------------------------
def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"未找到 {INPUT_PATH}，请先运行 数据预处理/01_format_unify.py")

    df = pd.read_csv(INPUT_PATH)
    print(f"原始数据 {df.shape}")

    # 1. 衍生特征 -------------------------------------------------------
    df = derive_extreme_risk(df)
    df = derive_ffrct_class3(df)
    df = derive_mean_hu_cat(df)

    # 2. 解析最终特征列表 -----------------------------------------------
    fixed_resolved, missing_fixed = [], []
    for name in FIXED_FEATURES_RAW:
        actual = resolve(name, df)
        if actual:
            fixed_resolved.append(actual)
        else:
            missing_fixed.append(name)

    # 衍生的固定特征
    for c in ["extreme_risk_ge2", "extreme_risk_ge3", "extreme_risk_ge4",
              "ffrct_class3"]:
        if c in df.columns:
            fixed_resolved.append(c)

    rfe_resolved, missing_rfe = [], []
    for name in RFE_CANDIDATES_RAW:
        if name in KNOWN_MISSING:
            missing_rfe.append(name)
            continue
        actual = resolve(name, df)
        if actual:
            rfe_resolved.append(actual)
        else:
            missing_rfe.append(name)

    # 衍生的 RFE 候选
    if "mean_hu_lt_minus70" in df.columns:
        rfe_resolved.append("mean_hu_lt_minus70")

    # 去重并保留顺序
    seen = set()
    fixed_resolved = [c for c in fixed_resolved if not (c in seen or seen.add(c))]
    seen = set(fixed_resolved)  # RFE 候选不能与固定特征重复
    rfe_resolved = [c for c in rfe_resolved if not (c in seen or seen.add(c))]

    print(f"\n固定特征 (fixed)         共 {len(fixed_resolved)} 个")
    print(f"RFE 候选 (rfe_candidates) 共 {len(rfe_resolved)} 个")
    if missing_fixed:
        print(f"[warn] 固定特征中缺失：{missing_fixed}")
    if missing_rfe:
        print(f"[warn] RFE 候选中缺失：{missing_rfe}")

    # 重命名：把 mean_hu（pcat_hu）改成更易处理的 mean_hu
    rename_map = {}
    if "mean_hu（pcat_hu）" in fixed_resolved + rfe_resolved:
        rename_map["mean_hu（pcat_hu）"] = "mean_hu"

    keep_cols = [TARGET_COL]
    if ID_COL in df.columns:
        keep_cols.append(ID_COL)
    keep_cols += fixed_resolved + rfe_resolved
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in df.columns]

    out = df[keep_cols].rename(columns=rename_map)
    fixed_final = [rename_map.get(c, c) for c in fixed_resolved]
    rfe_final   = [rename_map.get(c, c) for c in rfe_resolved]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")
    manifest = {
        "target": TARGET_COL,
        "id_column": ID_COL,
        "fixed_features": fixed_final,
        "rfe_candidates": rfe_final,
        "derived_features": [c for c in ["extreme_risk_ge2", "extreme_risk_ge3",
                                          "extreme_risk_ge4", "ffrct_class3",
                                          "mean_hu_lt_minus70"]
                              if c in out.columns],
        "missing_in_data": {
            "fixed": missing_fixed,
            "rfe":   missing_rfe,
        },
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 特征表        -> {OUT_FEATURES}  shape={out.shape}")
    print(f"[完成] 特征清单 JSON -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
