"""
建模 - 步骤 5：随机森林模型 + RFECV 特征筛选 + 可解释性分析
================================================================

流程：
1. 读取 ``data/all_variable_step1.csv``（步骤 1 编码统一后的数据）。
2. 特征工程：
   a) 列名映射（history_diabetes → history_diabetes_new 等）；
   b) 衍生固定特征（extreme_risk_ge2/ge3、ffrct_class3）；
   c) 衍生 RFE 候选特征：体积比、狭窄/FFRct 交互、易损性复合指标、
      实验室指标比值（NLR、TG/HDL 等）、斑块负荷、log 变换。
3. 按 accession_id 分组分层抽样（8:2），防止同一受试者跨集泄漏。
4. 多分类列 one-hot、缺失值中位数插补。
5. 在 RFE 候选集合上跑 RFECV（基学习器 = RF，5 折分层 CV，AUC）。
6. 最终特征 = 固定特征 ∪ RFE 选出的特征，训练最终 RF 模型。
7. 评估：5 折 CV AUC + 测试集 AUC / Accuracy / Precision / Sensitivity /
   Specificity / F1 / Brier（阈值 0.5 与 Youden 最优两套）。
8. 可解释性分析：
   a) Gini 特征重要性
   b) Permutation Importance
   c) SHAP summary plot
   d) SHAP 依赖图（Top-5 特征）
   e) 测试集 ROC 曲线
   f) Calibration 曲线

输入：
  data/all_variable_step1.csv

输出（均在 data/ 目录下）：
  rf_model_train.csv / rf_model_test.csv
  rf_selected_features.json / rf_rfe_ranking.csv
  rf_model_results.json
  rf_roc_curve.png
  rf_feature_importance_gini.png / rf_feature_importance_permutation.png
  rf_shap_summary.png / rf_shap_dependence_*.png
  rf_calibration_curve.png
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold, StratifiedShuffleSplit, cross_val_score,
)
from sklearn.pipeline import Pipeline

import shap

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_PATH = DATA_DIR / "all_variable_step1.csv"

RANDOM_STATE = 42
CV_FOLDS = 5
MIN_FEATURES_RFE = 5
TARGET_COL = "target_lesion_positive"
ID_COL = "accession_id"

COLUMN_ALIASES = {
    "history_diabetes":          "history_diabetes_new",
    "mean_hu":                   "mean_hu（pcat_hu）",
    "maximum_luminal_area_mm3":  "minimum_luminal_area_mm3",
    "followup_triglycerides":    "diff_triglycerides",
    "followup_lipoprotein_a":    "diff_lipoprotein_a",
}
KNOWN_MISSING = {"cda_admission_number", "stent_present"}

MULTICLASS_COLUMNS = {"plaque_type"}

HIGH_RISK_FLAGS = [
    "positive_remodeling_flag",
    "low_attenuation_plaque_flag",
    "napkin_ring_sign_flag",
    "spotty_calcification_flag",
]

FIXED_FEATURES_RAW = [
    "history_diabetes", "history_smoking",
    "baseline_triglycerides", "baseline_lipoprotein_a", "baseline_hba1c",
    "positive_remodeling_flag", "low_attenuation_plaque_flag",
    "napkin_ring_sign_flag", "spotty_calcification_flag",
    "plaque_feature_count", "High-Risk Plaque Flag",
    "contact_volume_mm3", "ffrct_value", "ffrct_risk_class",
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
    "followup_triglycerides",
    "followup_lipoprotein_a",
    "followup_hba1c",
    "plaque_type",
    "stent_present",
    "stenosis_percent",
    "maximum_luminal_area_mm3",
    "maximum_diameter_stenosis_percent",
    "calcified_volume_mm3", "calcified_volume_ratio",
    "non_calcified_volume_mm3",
    "low_attenuation_volume_mm3", "low_attenuation_volume_ratio",
    "fibrous_fatty_volume_mm3", "fibrous_fatty_volume_ratio",
    "fibrotic_volume_mm3",
    "whole_lesion_volume_mm3",
    "lumen_volume_mm3", "vessel_volume_mm3",
    "mean_hu",
    "std_hu",
]


# ===================================================================
# Helpers
# ===================================================================

def resolve(name: str, df: pd.DataFrame) -> str | None:
    actual = COLUMN_ALIASES.get(name, name)
    if actual in df.columns:
        return actual
    if name in df.columns:
        return name
    return None


def stratified_group_split(df, target, group, test_size, seed):
    if group and group in df.columns:
        g = df.groupby(group)[target].max().reset_index()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=seed)
        (tr_idx, te_idx), = sss.split(np.zeros(len(g)), g[target].astype(int))
        tr_ids = set(g.iloc[tr_idx][group])
        te_ids = set(g.iloc[te_idx][group])
        return (df[df[group].isin(tr_ids)].copy(),
                df[df[group].isin(te_ids)].copy())
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=seed)
    (tr_idx, te_idx), = sss.split(np.zeros(len(df)), df[target].astype(int))
    return df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()


def onehot_fit_apply(train, test, cols):
    new_cols: list[str] = []
    cols = [c for c in cols if c in train.columns]
    for c in cols:
        cats = sorted(train[c].dropna().astype(str).unique())
        for cat in cats:
            new_name = f"{c}__{cat}"
            train[new_name] = (train[c].astype(str) == cat).astype(int)
            test[new_name] = (
                (test[c].astype(str) == cat).astype(int)
                if c in test.columns else 0
            )
            new_cols.append(new_name)
        train.drop(columns=[c], inplace=True)
        if c in test.columns:
            test.drop(columns=[c], inplace=True)
    return train, test, new_cols


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "AUC":         float(roc_auc_score(y_true, y_prob)),
        "Accuracy":    float(accuracy_score(y_true, y_pred)),
        "Precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "F1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "Brier":       float(brier_score_loss(y_true, y_prob)),
        "Threshold":   float(threshold),
    }


def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


# ===================================================================
# Feature Engineering
# ===================================================================

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create derived features; return (df, list_of_new_column_names)."""
    derived = []

    # ── Derived fixed features ────────────────────────────────────────
    flag_cols = [c for c in HIGH_RISK_FLAGS if c in df.columns]
    if len(flag_cols) >= 2:
        cnt = df[flag_cols].fillna(0).sum(axis=1)
        df["extreme_risk_ge2"] = (cnt >= 2).astype(int)
        df["extreme_risk_ge3"] = (cnt >= 3).astype(int)
        derived += ["extreme_risk_ge2", "extreme_risk_ge3"]

    if "ffrct_value" in df.columns:
        v = df["ffrct_value"]
        df["ffrct_class3"] = np.where(v >= 0.8, 0, np.where(v >= 0.7, 1, 2))
        df["ffrct_class3"] = df["ffrct_class3"].where(v.notna()).astype(float)
        derived.append("ffrct_class3")

    mean_hu_col = COLUMN_ALIASES.get("mean_hu", "mean_hu")
    if mean_hu_col in df.columns:
        df["mean_hu_lt_minus70"] = (df[mean_hu_col] < -70).astype(int)
        df["mean_hu_lt_minus70"] = df["mean_hu_lt_minus70"].where(
            df[mean_hu_col].notna()
        )
        derived.append("mean_hu_lt_minus70")

    # ── RFE candidate derived features ────────────────────────────────

    # 1) Volume ratios (clinically meaningful plaque composition)
    ratio_pairs = [
        ("low_attenuation_volume_mm3", "whole_lesion_volume_mm3", "la_whole_ratio"),
        ("non_calcified_volume_mm3",   "whole_lesion_volume_mm3", "nc_whole_ratio"),
        ("fibrous_fatty_volume_mm3",   "whole_lesion_volume_mm3", "ff_whole_ratio"),
        ("fibrotic_volume_mm3",        "whole_lesion_volume_mm3", "fib_whole_ratio"),
        ("non_calcified_volume_mm3",   "calcified_volume_mm3",   "nc_calc_ratio"),
        ("lumen_volume_mm3",           "vessel_volume_mm3",       "lumen_vessel_ratio"),
        ("low_attenuation_volume_mm3", "non_calcified_volume_mm3", "la_nc_ratio"),
        ("whole_lesion_volume_mm3",    "vessel_volume_mm3",       "lesion_vessel_ratio"),
    ]
    for num, den, name in ratio_pairs:
        if num in df.columns and den in df.columns:
            df[name] = df[num] / (df[den] + 1e-6)
            derived.append(name)

    # 2) Stenosis interactions
    if "stenosis_percent" in df.columns:
        df["stenosis_sq"] = df["stenosis_percent"] ** 2
        derived.append("stenosis_sq")
        if "minimum_luminal_area_mm3" in df.columns:
            df["stenosis_inv_lumen"] = df["stenosis_percent"] / (
                df["minimum_luminal_area_mm3"] + 1e-6
            )
            derived.append("stenosis_inv_lumen")

    if "ffrct_value" in df.columns and "stenosis_percent" in df.columns:
        df["ffrct_stenosis"] = (1 - df["ffrct_value"]) * df["stenosis_percent"]
        derived.append("ffrct_stenosis")

    # 3) Vulnerability composite
    if len(flag_cols) >= 2:
        df["vulnerability_score"] = df[flag_cols].fillna(0).sum(axis=1)
        derived.append("vulnerability_score")
        if "stenosis_percent" in df.columns:
            df["vuln_x_stenosis"] = (
                df["vulnerability_score"] * df["stenosis_percent"]
            )
            derived.append("vuln_x_stenosis")

    # 4) Soft / vulnerable plaque total volume
    soft_cols = [c for c in ["low_attenuation_volume_mm3",
                             "fibrous_fatty_volume_mm3"] if c in df.columns]
    if soft_cols:
        df["total_vulnerable_vol"] = df[soft_cols].fillna(0).sum(axis=1)
        derived.append("total_vulnerable_vol")

    # 5) Lab-value interactions
    if "baseline_neutrophil_pct" in df.columns and \
       "baseline_lymphocyte_pct" in df.columns:
        df["nlr"] = df["baseline_neutrophil_pct"] / (
            df["baseline_lymphocyte_pct"] + 1e-6
        )
        derived.append("nlr")

    if "baseline_triglycerides" in df.columns and \
       "baseline_hdl_cholesterol" in df.columns:
        df["tg_hdl_ratio"] = df["baseline_triglycerides"] / (
            df["baseline_hdl_cholesterol"] + 1e-6
        )
        derived.append("tg_hdl_ratio")

    if "baseline_total_cholesterol" in df.columns and \
       "baseline_hdl_cholesterol" in df.columns:
        df["tc_hdl_ratio"] = df["baseline_total_cholesterol"] / (
            df["baseline_hdl_cholesterol"] + 1e-6
        )
        derived.append("tc_hdl_ratio")

    if "baseline_crp" in df.columns and "baseline_wbc" in df.columns:
        df["crp_x_wbc"] = df["baseline_crp"].fillna(0) * \
                           df["baseline_wbc"].fillna(0)
        derived.append("crp_x_wbc")

    # 6) Plaque burden features (directly from data, derived from volumes)
    for c in ["non_calcified_plaque_burden", "low_attenuation_plaque_burden",
              "whole_lesion_plaque_burden", "fibrous_fatty_plaque_burden",
              "remodeling_index", "maximum_area_stenosis_percent"]:
        if c in df.columns and c not in derived:
            derived.append(c)

    # 7) Log-transforms for highly skewed lab values
    skewed = ["baseline_ctnt", "baseline_nt_pro_bnp", "baseline_crp",
              "low_attenuation_volume_mm3", "non_calcified_volume_mm3",
              "whole_lesion_volume_mm3"]
    for c in skewed:
        if c in df.columns:
            log_c = f"log_{c}"
            df[log_c] = np.log1p(df[c].fillna(0).clip(lower=0))
            derived.append(log_c)

    return df, derived


# ===================================================================
# Plotting
# ===================================================================

def plot_roc(y_true, y_prob, auc_val, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, lw=2, label=f"RF  AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("Test-set ROC — Random Forest")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gini_importance(model, features, path, top_n=25):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, len(idx) * 0.35)))
    ax.barh(range(len(idx)), imp[idx][::-1], color="#3b82f6")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([features[i] for i in idx][::-1], fontsize=9)
    ax.set_xlabel("Gini Importance")
    ax.set_title(f"RF Feature Importance (Gini, top {len(idx)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_permutation_importance(result, features, path, top_n=25):
    sorted_idx = result.importances_mean.argsort()[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_idx) * 0.35)))
    ax.barh(
        range(len(sorted_idx)),
        result.importances_mean[sorted_idx][::-1],
        xerr=result.importances_std[sorted_idx][::-1],
        color="#10b981",
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([features[i] for i in sorted_idx][::-1], fontsize=9)
    ax.set_xlabel("Decrease in AUC")
    ax.set_title(f"Permutation Importance (top {len(sorted_idx)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(y_true, y_prob, path, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(prob_pred, prob_true, "s-", label="RF")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5,
            label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve — Random Forest")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shap_summary(shap_values, X, path):
    plt.figure(figsize=(10, max(6, len(X.columns) * 0.32)))
    shap.summary_plot(shap_values, X, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_shap_dependence(shap_values, X, feature, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.dependence_plot(feature, shap_values.values, X, ax=ax, show=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    t_start = time.time()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"未找到 {INPUT_PATH}，请先运行 数据预处理/01_format_unify.py")

    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    print(f"原始数据 {df.shape}")

    # ── 1. Feature engineering ─────────────────────────────────────────
    df, derived_cols = engineer_features(df)
    print(f"[FE] 衍生特征 {len(derived_cols)} 个")

    # Resolve fixed features
    fixed_resolved, missing_fixed = [], []
    for name in FIXED_FEATURES_RAW:
        actual = resolve(name, df)
        if actual:
            fixed_resolved.append(actual)
        else:
            missing_fixed.append(name)

    for c in ["extreme_risk_ge2", "extreme_risk_ge3", "ffrct_class3"]:
        if c in df.columns:
            fixed_resolved.append(c)

    # Resolve RFE candidates
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

    for c in derived_cols:
        if c not in fixed_resolved and c not in rfe_resolved and c in df.columns:
            rfe_resolved.append(c)

    seen = set()
    fixed_resolved = [c for c in fixed_resolved
                      if not (c in seen or seen.add(c))]
    seen_all = set(fixed_resolved)
    rfe_resolved = [c for c in rfe_resolved
                    if not (c in seen_all or seen_all.add(c))]

    print(f"\n固定特征 (fixed)          共 {len(fixed_resolved)} 个")
    print(f"RFE 候选 (rfe_candidates) 共 {len(rfe_resolved)} 个")
    if missing_fixed:
        print(f"[warn] 固定特征中缺失：{missing_fixed}")
    if missing_rfe:
        print(f"[warn] RFE 候选中缺失：{missing_rfe}")

    # ── 2. Train / Test split ──────────────────────────────────────────
    train, test = stratified_group_split(
        df, TARGET_COL, ID_COL, 0.2, RANDOM_STATE
    )
    print(f"\n训练集 {train.shape} (事件率 {train[TARGET_COL].mean():.2%})")
    print(f"测试集 {test.shape} (事件率 {test[TARGET_COL].mean():.2%})")

    # ── 3. One-hot for multi-class candidates ──────────────────────────
    multi_cols = [c for c in rfe_resolved if c in MULTICLASS_COLUMNS]
    train, test, onehot_cols = onehot_fit_apply(train, test, multi_cols)
    rfe_resolved = [c for c in rfe_resolved if c not in multi_cols] + onehot_cols
    if onehot_cols:
        print(f"[onehot] {multi_cols} -> {onehot_cols}")

    # ── 4. Median imputation ──────────────────────────────────────────
    all_feat = fixed_resolved + rfe_resolved
    all_feat = [c for c in dict.fromkeys(all_feat) if c in train.columns]
    imputer = SimpleImputer(strategy="median")
    train[all_feat] = imputer.fit_transform(train[all_feat])
    test[all_feat]  = imputer.transform(test[all_feat])

    y_train = train[TARGET_COL].astype(int).values
    y_test  = test[TARGET_COL].astype(int).values

    # ── 5. RFECV with Random Forest ───────────────────────────────────
    rfe_estimator = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        class_weight="balanced", n_jobs=1,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    rfe_cols = [c for c in rfe_resolved if c in train.columns]
    print(f"\n[RFE] 在 {len(rfe_cols)} 个候选上跑 RFECV（RF, {CV_FOLDS}-fold, AUC）...")
    rfecv = RFECV(
        estimator=rfe_estimator, step=1, cv=cv, scoring="roc_auc",
        min_features_to_select=max(1, MIN_FEATURES_RFE),
        n_jobs=CV_FOLDS,
    )
    rfecv.fit(train[rfe_cols], y_train)

    ranking = pd.DataFrame({
        "feature": rfe_cols,
        "rank": rfecv.ranking_,
        "selected": rfecv.support_,
    }).sort_values(["rank", "feature"])
    ranking.to_csv(DATA_DIR / "rf_rfe_ranking.csv", index=False,
                   encoding="utf-8-sig")

    selected_rfe = [c for c, keep in zip(rfe_cols, rfecv.support_) if keep]
    cv_best_auc_rfe = float(rfecv.cv_results_["mean_test_score"].max())
    print(f"[RFE] 选出 {len(selected_rfe)} / {len(rfe_cols)} 个候选特征")
    print(f"      {selected_rfe}")
    print(f"[RFE] 最佳 CV AUC = {cv_best_auc_rfe:.4f} (@ k={rfecv.n_features_})")

    final_features = fixed_resolved + selected_rfe
    final_features = [c for c in dict.fromkeys(final_features)
                      if c in train.columns]
    print(f"\n最终特征 ({len(final_features)}) = "
          f"固定 {len(fixed_resolved)} + RFE {len(selected_rfe)}")

    # ── 6. Save train / test matrices ─────────────────────────────────
    keep_cols = (
        ([ID_COL] if ID_COL in train.columns else [])
        + [TARGET_COL] + final_features
    )
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in train.columns]
    train[keep_cols].to_csv(DATA_DIR / "rf_model_train.csv", index=False,
                            encoding="utf-8-sig")
    test[keep_cols].to_csv(DATA_DIR / "rf_model_test.csv", index=False,
                           encoding="utf-8-sig")

    X_train = train[final_features].copy()
    X_test  = test[final_features].copy()

    # ── 7. Train final RF model ───────────────────────────────────────
    rf_final = RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", rf_final),
    ])

    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=CV_FOLDS)
    print(f"\n[模型] RF {CV_FOLDS}-fold CV AUC = "
          f"{np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")

    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    m_default = evaluate(y_test, y_prob, 0.5)
    thr_opt = youden_threshold(y_test, y_prob)
    m_opt = evaluate(y_test, y_prob, thr_opt)

    print(f"\n=== 测试集指标 (阈值=0.5) ===")
    for k, v in m_default.items():
        print(f"  {k:<14s}: {v:.4f}")
    print(f"\n=== 测试集指标 (阈值=Youden {thr_opt:.3f}) ===")
    for k, v in m_opt.items():
        print(f"  {k:<14s}: {v:.4f}")

    # ── 8. Plots ──────────────────────────────────────────────────────
    print("\n[plot] ROC 曲线 ...")
    plot_roc(y_test, y_prob, m_default["AUC"],
             DATA_DIR / "rf_roc_curve.png")

    rf_model = pipe.named_steps["clf"]

    print("[plot] Gini 特征重要性 ...")
    plot_gini_importance(rf_model, final_features,
                         DATA_DIR / "rf_feature_importance_gini.png")

    print("[plot] Permutation Importance ...")
    perm_result = permutation_importance(
        pipe, X_test, y_test,
        scoring="roc_auc", n_repeats=20,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    plot_permutation_importance(perm_result, final_features,
                                DATA_DIR / "rf_feature_importance_permutation.png")

    print("[plot] Calibration 曲线 ...")
    plot_calibration(y_test, y_prob, DATA_DIR / "rf_calibration_curve.png")

    # ── 9. SHAP ───────────────────────────────────────────────────────
    print("[SHAP] 计算 TreeExplainer ...")
    X_test_imp = pd.DataFrame(
        pipe.named_steps["imputer"].transform(X_test),
        columns=final_features, index=X_test.index,
    )
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(X_test_imp)

    shap_positive = shap_values[..., 1]

    print("[SHAP] summary plot ...")
    plot_shap_summary(shap_positive, X_test_imp,
                      DATA_DIR / "rf_shap_summary.png")

    mean_abs_shap = np.abs(shap_positive.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:5]
    for rank, idx in enumerate(top_indices, 1):
        fname = final_features[idx]
        out_dep = DATA_DIR / f"rf_shap_dependence_{rank}_{fname}.png"
        print(f"[SHAP] 依赖图 #{rank}: {fname}")
        plot_shap_dependence(shap_positive, X_test_imp, fname, out_dep)

    # ── 10. Results JSON ──────────────────────────────────────────────
    gini_imp = dict(zip(final_features,
                        [float(x) for x in rf_model.feature_importances_]))
    perm_imp = dict(zip(final_features,
                        [float(x) for x in perm_result.importances_mean]))
    shap_imp = dict(zip(final_features,
                        [float(x) for x in mean_abs_shap]))

    results = {
        "model": "RandomForest",
        "parameters": {
            "n_estimators": 1000,
            "max_depth": 10,
            "min_samples_split": 8,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
        "data": {
            "total_samples": int(len(df)),
            "train_samples": int(len(train)),
            "test_samples": int(len(test)),
            "event_rate_train": float(y_train.mean()),
            "event_rate_test": float(y_test.mean()),
            "group_split": True,
            "group_column": ID_COL,
        },
        "fixed_features": fixed_resolved,
        "rfe_selected": selected_rfe,
        "final_features": final_features,
        "n_features": len(final_features),
        "rfe_best_cv_auc": cv_best_auc_rfe,
        "cv_auc_mean": float(np.mean(cv_auc)),
        "cv_auc_std": float(np.std(cv_auc)),
        "test_threshold_0.5": m_default,
        "test_threshold_youden": m_opt,
        "feature_importance_gini": gini_imp,
        "feature_importance_permutation": perm_imp,
        "feature_importance_shap_mean_abs": shap_imp,
    }

    sel_summary = {
        "target": TARGET_COL,
        "id_column": ID_COL,
        "fixed_features": fixed_resolved,
        "rfe_pool": rfe_cols,
        "rfe_selected": selected_rfe,
        "final_features": final_features,
        "rfe_best_n_features": int(len(selected_rfe)),
        "rfe_best_cv_auc": cv_best_auc_rfe,
        "source": "rfecv_rf",
    }
    with open(DATA_DIR / "rf_selected_features.json", "w",
              encoding="utf-8") as f:
        json.dump(sel_summary, f, ensure_ascii=False, indent=2)

    with open(DATA_DIR / "rf_model_results.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  RF 模型训练与可解释性分析完成  ({elapsed:.1f}s)")
    print(f"  最终特征数   = {len(final_features)}")
    print(f"  CV AUC       = {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")
    print(f"  Test AUC     = {m_default['AUC']:.4f}")
    print(f"{'='*60}")
    print(f"\n[完成] 训练矩阵      -> {DATA_DIR / 'rf_model_train.csv'}")
    print(f"[完成] 测试矩阵      -> {DATA_DIR / 'rf_model_test.csv'}")
    print(f"[完成] 选出特征 JSON  -> {DATA_DIR / 'rf_selected_features.json'}")
    print(f"[完成] RFE 排序       -> {DATA_DIR / 'rf_rfe_ranking.csv'}")
    print(f"[完成] 模型结果 JSON  -> {DATA_DIR / 'rf_model_results.json'}")
    print(f"[完成] ROC 曲线       -> {DATA_DIR / 'rf_roc_curve.png'}")
    print(f"[完成] Gini 重要性    -> {DATA_DIR / 'rf_feature_importance_gini.png'}")
    print(f"[完成] Perm 重要性    -> {DATA_DIR / 'rf_feature_importance_permutation.png'}")
    print(f"[完成] SHAP 汇总图    -> {DATA_DIR / 'rf_shap_summary.png'}")
    print(f"[完成] 校准曲线       -> {DATA_DIR / 'rf_calibration_curve.png'}")


if __name__ == "__main__":
    main()
