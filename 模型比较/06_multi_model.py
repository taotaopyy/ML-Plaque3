"""
建模 - 步骤 6：多基学习器 RFE 共识特征选择 + 7 模型对比 + 可解释性分析
======================================================================

目标：使用 LR / LightGBM / XGBoost 三个基学习器分别做 RFECV，取共识
（多数表决）得到 ~30 个最终特征（固定 14 + RFE ~16），然后在该特征集上
对比 KNN / LR / RF / XGBoost / AdaBoost / LightGBM / SVM 共 7 个模型，
输出最优模型的可解释性分析。

流程：
1. 读取 ``data/all_variable_step1.csv``（步骤 1 编码统一后的数据）。
2. 特征工程：列名映射、衍生固定特征、衍生 RFE 候选特征。
3. 按 accession_id 分组分层 8:2 抽样。
4. 多分类列 one-hot → 缺失值中位数插补。
5. 三基学习器 RFECV（LR / LightGBM / XGBoost），各自在 RFE 候选集上
   做 5 折分层 CV，取多数表决得到共识特征集。
6. 最终特征 = 固定特征 ∪ 共识 RFE 特征。
7. 在最终特征上训练并对比 7 个模型（Pipeline: imputer + [scaler] + clf）。
8. 可解释性分析（基于最优模型）：
   - Feature Importance（Gini / coefficient / gain）
   - Permutation Importance
   - SHAP summary + 依赖图
   - ROC 曲线（所有模型叠加）
   - 校准曲线

输入：data/all_variable_step1.csv
输出（均在 data/ 目录下）：
  multi_model_train.csv / multi_model_test.csv
  multi_rfe_consensus.json / multi_rfe_per_learner.csv
  multi_model_comparison.csv / multi_model_comparison_optimal.csv
  multi_model_results.json
  multi_roc_comparison.png
  multi_best_feature_importance.png / multi_best_permutation_importance.png
  multi_best_shap_summary.png / multi_best_shap_dependence_*.png
  multi_best_calibration.png
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
from sklearn.ensemble import (
    AdaBoostClassifier, RandomForestClassifier,
)
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold, StratifiedShuffleSplit, cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import shap

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_PATH = DATA_DIR / "all_variable_step1.csv"

RANDOM_STATE = 42
CV_FOLDS = 5
MIN_FEATURES_RFE = 5
TARGET_COL = "target_lesion_positive"
ID_COL = "accession_id"
VOTE_THRESHOLD = 2

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
    "positive_remodeling_flag", "low_attenuation_plaque_flag",
    "napkin_ring_sign_flag", "spotty_calcification_flag",
]
SCALE_MODELS = {"KNN", "LR", "SVM"}

FIXED_FEATURES_RAW = [
    "history_diabetes", "history_smoking",
    "baseline_triglycerides", "baseline_lipoprotein_a", "baseline_hba1c",
    "positive_remodeling_flag", "low_attenuation_plaque_flag",
    "napkin_ring_sign_flag", "spotty_calcification_flag",
    "plaque_feature_count", "High-Risk Plaque Flag",
    "contact_volume_mm3", "ffrct_value", "ffrct_risk_class",
]

RFE_CANDIDATES_RAW = [
    "history_cabg", "cda_admission_number",
    "baseline_ctnt", "baseline_nt_pro_bnp", "baseline_creatinine",
    "baseline_uric_acid", "baseline_ck", "baseline_ck_mb",
    "baseline_crp", "baseline_albumin", "baseline_alt",
    "baseline_wbc", "baseline_neutrophil_pct", "baseline_lymphocyte_pct",
    "baseline_platelet_count", "baseline_pdw",
    "baseline_total_cholesterol", "baseline_hdl_cholesterol",
    "baseline_lvef",
    "followup_hdl_cholesterol",
    "followup_triglycerides", "followup_lipoprotein_a",
    "followup_hba1c",
    "plaque_type", "stent_present",
    "stenosis_percent", "maximum_luminal_area_mm3",
    "maximum_diameter_stenosis_percent",
    "calcified_volume_mm3", "calcified_volume_ratio",
    "non_calcified_volume_mm3",
    "low_attenuation_volume_mm3", "low_attenuation_volume_ratio",
    "fibrous_fatty_volume_mm3", "fibrous_fatty_volume_ratio",
    "fibrotic_volume_mm3", "whole_lesion_volume_mm3",
    "lumen_volume_mm3", "vessel_volume_mm3",
    "mean_hu", "std_hu",
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
            nm = f"{c}__{cat}"
            train[nm] = (train[c].astype(str) == cat).astype(int)
            test[nm] = ((test[c].astype(str) == cat).astype(int)
                        if c in test.columns else 0)
            new_cols.append(nm)
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
# Feature Engineering (same as 05_rf_model.py)
# ===================================================================

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    derived = []
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

    mhu = COLUMN_ALIASES.get("mean_hu", "mean_hu")
    if mhu in df.columns:
        df["mean_hu_lt_minus70"] = (df[mhu] < -70).astype(int).where(df[mhu].notna())
        derived.append("mean_hu_lt_minus70")

    ratio_pairs = [
        ("low_attenuation_volume_mm3", "whole_lesion_volume_mm3", "la_whole_ratio"),
        ("non_calcified_volume_mm3",   "whole_lesion_volume_mm3", "nc_whole_ratio"),
        ("fibrous_fatty_volume_mm3",   "whole_lesion_volume_mm3", "ff_whole_ratio"),
        ("fibrotic_volume_mm3",        "whole_lesion_volume_mm3", "fib_whole_ratio"),
        ("non_calcified_volume_mm3",   "calcified_volume_mm3",   "nc_calc_ratio"),
        ("lumen_volume_mm3",           "vessel_volume_mm3",       "lumen_vessel_ratio"),
        ("low_attenuation_volume_mm3", "non_calcified_volume_mm3","la_nc_ratio"),
        ("whole_lesion_volume_mm3",    "vessel_volume_mm3",       "lesion_vessel_ratio"),
    ]
    for num, den, name in ratio_pairs:
        if num in df.columns and den in df.columns:
            df[name] = df[num] / (df[den] + 1e-6)
            derived.append(name)

    if "stenosis_percent" in df.columns:
        df["stenosis_sq"] = df["stenosis_percent"] ** 2
        derived.append("stenosis_sq")
        if "minimum_luminal_area_mm3" in df.columns:
            df["stenosis_inv_lumen"] = df["stenosis_percent"] / (
                df["minimum_luminal_area_mm3"] + 1e-6)
            derived.append("stenosis_inv_lumen")

    if "ffrct_value" in df.columns and "stenosis_percent" in df.columns:
        df["ffrct_stenosis"] = (1 - df["ffrct_value"]) * df["stenosis_percent"]
        derived.append("ffrct_stenosis")

    if len(flag_cols) >= 2:
        df["vulnerability_score"] = df[flag_cols].fillna(0).sum(axis=1)
        derived.append("vulnerability_score")
        if "stenosis_percent" in df.columns:
            df["vuln_x_stenosis"] = df["vulnerability_score"] * df["stenosis_percent"]
            derived.append("vuln_x_stenosis")

    soft_cols = [c for c in ["low_attenuation_volume_mm3",
                             "fibrous_fatty_volume_mm3"] if c in df.columns]
    if soft_cols:
        df["total_vulnerable_vol"] = df[soft_cols].fillna(0).sum(axis=1)
        derived.append("total_vulnerable_vol")

    if "baseline_neutrophil_pct" in df.columns and \
       "baseline_lymphocyte_pct" in df.columns:
        df["nlr"] = df["baseline_neutrophil_pct"] / (
            df["baseline_lymphocyte_pct"] + 1e-6)
        derived.append("nlr")

    if "baseline_triglycerides" in df.columns and \
       "baseline_hdl_cholesterol" in df.columns:
        df["tg_hdl_ratio"] = df["baseline_triglycerides"] / (
            df["baseline_hdl_cholesterol"] + 1e-6)
        derived.append("tg_hdl_ratio")

    if "baseline_total_cholesterol" in df.columns and \
       "baseline_hdl_cholesterol" in df.columns:
        df["tc_hdl_ratio"] = df["baseline_total_cholesterol"] / (
            df["baseline_hdl_cholesterol"] + 1e-6)
        derived.append("tc_hdl_ratio")

    if "baseline_crp" in df.columns and "baseline_wbc" in df.columns:
        df["crp_x_wbc"] = df["baseline_crp"].fillna(0) * \
                           df["baseline_wbc"].fillna(0)
        derived.append("crp_x_wbc")

    for c in ["non_calcified_plaque_burden", "low_attenuation_plaque_burden",
              "whole_lesion_plaque_burden", "fibrous_fatty_plaque_burden",
              "remodeling_index", "maximum_area_stenosis_percent"]:
        if c in df.columns and c not in derived:
            derived.append(c)

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
# RFE base learners
# ===================================================================

def make_rfe_learners():
    return {
        "LR": (LogisticRegression(
            penalty="l2", solver="liblinear", max_iter=2000,
            class_weight="balanced", random_state=RANDOM_STATE), 1),
        "LightGBM": (LGBMClassifier(
            n_estimators=100, learning_rate=0.1, num_leaves=31,
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=1, verbose=-1), 1),
        "XGBoost": (XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            scale_pos_weight=5.5, eval_metric="auc", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=1), 1),
    }


# ===================================================================
# Model builder (7 models)
# ===================================================================

def build_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=1),
        "LR": LogisticRegression(
            penalty="l2", solver="liblinear", max_iter=2000,
            class_weight="balanced", random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced", n_jobs=1,
            random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=5.5, eval_metric="auc",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=1),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5,
            random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.05,
            num_leaves=31, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
        "SVM": SVC(C=1.0, kernel="rbf", probability=True,
                   class_weight="balanced", random_state=RANDOM_STATE),
    }


def make_pipeline(name, estimator):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name in SCALE_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


# ===================================================================
# Plotting
# ===================================================================

def plot_roc_all(roc_data, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc_val) in sorted(roc_data.items(),
                                             key=lambda kv: -kv[1][2]):
        ax.plot(fpr, tpr, label=f"{name}  AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Test-set ROC Comparison")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(values, names, xlabel, title, path, top_n=25):
    idx = np.argsort(values)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, len(idx) * 0.35)))
    ax.barh(range(len(idx)), np.array(values)[idx][::-1], color="#3b82f6")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx][::-1], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_perm_importance(result, features, path, top_n=25):
    si = result.importances_mean.argsort()[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, len(si) * 0.35)))
    ax.barh(range(len(si)),
            result.importances_mean[si][::-1],
            xerr=result.importances_std[si][::-1],
            color="#10b981")
    ax.set_yticks(range(len(si)))
    ax.set_yticklabels([features[i] for i in si][::-1], fontsize=9)
    ax.set_xlabel("Decrease in AUC")
    ax.set_title(f"Permutation Importance (top {len(si)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(y_true, y_prob, name, path, n_bins=10):
    pt, pp = calibration_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(pp, pt, "s-", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5,
            label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {name}")
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


def plot_shap_dep(shap_values, X, feature, path):
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
        df, TARGET_COL, ID_COL, 0.2, RANDOM_STATE)
    print(f"\n训练集 {train.shape} (事件率 {train[TARGET_COL].mean():.2%})")
    print(f"测试集 {test.shape} (事件率 {test[TARGET_COL].mean():.2%})")

    # ── 3. One-hot ─────────────────────────────────────────────────────
    multi_cols = [c for c in rfe_resolved if c in MULTICLASS_COLUMNS]
    train, test, onehot_cols = onehot_fit_apply(train, test, multi_cols)
    rfe_resolved = [c for c in rfe_resolved
                    if c not in multi_cols] + onehot_cols
    if onehot_cols:
        print(f"[onehot] {multi_cols} -> {onehot_cols}")

    # ── 4. Imputation ──────────────────────────────────────────────────
    all_feat = fixed_resolved + rfe_resolved
    all_feat = [c for c in dict.fromkeys(all_feat) if c in train.columns]
    imputer = SimpleImputer(strategy="median")
    train[all_feat] = imputer.fit_transform(train[all_feat])
    test[all_feat]  = imputer.transform(test[all_feat])
    y_train = train[TARGET_COL].astype(int).values
    y_test  = test[TARGET_COL].astype(int).values

    rfe_cols = [c for c in rfe_resolved if c in train.columns]

    # ── 5. Multi-learner RFECV ─────────────────────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    X_rfe = train[rfe_cols].copy()
    scaler = StandardScaler()
    X_rfe_std = pd.DataFrame(scaler.fit_transform(X_rfe),
                             columns=rfe_cols, index=train.index)

    ranking_table = pd.DataFrame({"feature": rfe_cols})
    rfe_summary = {}

    for learner_name, (est, step) in make_rfe_learners().items():
        t0 = time.time()
        print(f"\n[RFE] base = {learner_name}  在 {len(rfe_cols)} 候选上 "
              f"RFECV (5-fold, AUC) ...", flush=True)

        X_input = X_rfe_std if learner_name == "LR" else X_rfe

        rfecv = RFECV(
            estimator=est, step=step, cv=cv, scoring="roc_auc",
            min_features_to_select=max(1, MIN_FEATURES_RFE),
            n_jobs=CV_FOLDS,
        )
        rfecv.fit(X_input, y_train)

        scores = rfecv.cv_results_["mean_test_score"]
        best_k = int(rfecv.n_features_)
        best_auc = float(scores.max())
        selected = [f for f, keep in zip(rfe_cols, rfecv.support_) if keep]

        print(f"      [{time.time()-t0:.1f}s] 选出 {len(selected)} 个，"
              f"最佳 CV AUC = {best_auc:.4f} (@ k={best_k})")
        print(f"      {selected}")

        ranking_table[f"rank_{learner_name}"] = rfecv.ranking_
        ranking_table[f"selected_{learner_name}"] = rfecv.support_.astype(int)
        rfe_summary[learner_name] = {
            "best_n_features": best_k,
            "best_cv_auc": best_auc,
            "selected": selected,
        }

    # ── Consensus (average-rank approach, target ~13 RFE features) ────
    TARGET_RFE_COUNT = 13

    vote_cols = [c for c in ranking_table.columns if c.startswith("selected_")]
    rank_cols = [c for c in ranking_table.columns if c.startswith("rank_")]
    ranking_table["votes"] = ranking_table[vote_cols].sum(axis=1)
    ranking_table["avg_rank"] = ranking_table[rank_cols].mean(axis=1)
    ranking_table = ranking_table.sort_values(
        ["avg_rank", "feature"], ascending=[True, True])
    ranking_table.to_csv(DATA_DIR / "multi_rfe_per_learner.csv",
                         index=False, encoding="utf-8-sig")

    consensus_features = ranking_table.head(TARGET_RFE_COUNT)["feature"].tolist()

    print(f"\n[共识] 按平均 rank 排序，取前 {TARGET_RFE_COUNT} 个 RFE 特征：")
    for _, row in ranking_table.head(TARGET_RFE_COUNT).iterrows():
        print(f"    {row['feature']:<45s}  avg_rank={row['avg_rank']:.1f}  "
              f"votes={int(row['votes'])}")

    final_features = fixed_resolved + consensus_features
    final_features = [c for c in dict.fromkeys(final_features)
                      if c in train.columns]
    print(f"\n最终特征 ({len(final_features)}) = "
          f"固定 {len(fixed_resolved)} + 共识 {len(consensus_features)}")

    # ── Save consensus JSON ────────────────────────────────────────────
    consensus_json = {
        "target": TARGET_COL,
        "id_column": ID_COL,
        "fixed_features": fixed_resolved,
        "rfe_pool": rfe_cols,
        "per_learner": rfe_summary,
        "vote_threshold": VOTE_THRESHOLD,
        "consensus_features": consensus_features,
        "final_features": final_features,
    }
    with open(DATA_DIR / "multi_rfe_consensus.json", "w",
              encoding="utf-8") as f:
        json.dump(consensus_json, f, ensure_ascii=False, indent=2)

    # ── 6. Save train / test matrices ─────────────────────────────────
    keep_cols = (([ID_COL] if ID_COL in train.columns else [])
                 + [TARGET_COL] + final_features)
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in train.columns]
    train[keep_cols].to_csv(DATA_DIR / "multi_model_train.csv",
                            index=False, encoding="utf-8-sig")
    test[keep_cols].to_csv(DATA_DIR / "multi_model_test.csv",
                           index=False, encoding="utf-8-sig")

    X_train = train[final_features].copy()
    X_test  = test[final_features].copy()

    # ── 7. 7-model comparison ─────────────────────────────────────────
    rows_default, rows_optimal = [], []
    roc_data = {}
    best_name, best_auc_val, best_pipe = "", -1.0, None
    best_y_prob = None

    for name, est in build_models().items():
        t0 = time.time()
        print(f"\n[fit] {name} ...", flush=True)
        pipe = make_pipeline(name, est)
        try:
            cv_auc = cross_val_score(
                pipe, X_train, y_train, cv=cv,
                scoring="roc_auc", n_jobs=CV_FOLDS)
        except Exception as e:
            print(f"[warn] {name} CV 失败：{e}")
            cv_auc = np.array([np.nan])

        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        m_def = evaluate(y_test, y_prob, 0.5)
        thr = youden_threshold(y_test, y_prob)
        m_opt = evaluate(y_test, y_prob, thr)

        rows_default.append({"Model": name,
                              "CV_AUC_mean": float(np.nanmean(cv_auc)),
                              "CV_AUC_std":  float(np.nanstd(cv_auc)),
                              **m_def})
        rows_optimal.append({"Model": name,
                              "CV_AUC_mean": float(np.nanmean(cv_auc)),
                              "CV_AUC_std":  float(np.nanstd(cv_auc)),
                              **m_opt})

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, m_def["AUC"])

        print(f"  {name:<9s}  CV_AUC={np.nanmean(cv_auc):.4f}±"
              f"{np.nanstd(cv_auc):.4f}  Test_AUC={m_def['AUC']:.4f}  "
              f"Sens={m_def['Sensitivity']:.3f}  "
              f"Spec={m_def['Specificity']:.3f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

        if m_def["AUC"] > best_auc_val:
            best_auc_val = m_def["AUC"]
            best_name = name
            best_pipe = pipe
            best_y_prob = y_prob

    cmp_def = pd.DataFrame(rows_default).sort_values("AUC", ascending=False)
    cmp_opt = pd.DataFrame(rows_optimal).sort_values("AUC", ascending=False)
    cmp_def.to_csv(DATA_DIR / "multi_model_comparison.csv",
                   index=False, encoding="utf-8-sig")
    cmp_opt.to_csv(DATA_DIR / "multi_model_comparison_optimal.csv",
                   index=False, encoding="utf-8-sig")

    print(f"\n=== 测试集对比（阈值=0.5，按 AUC 排序） ===")
    print(cmp_def.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n=== 测试集对比（阈值=Youden 最优） ===")
    print(cmp_opt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── 8. Plots ──────────────────────────────────────────────────────
    print(f"\n[plot] ROC 曲线（所有模型） ...")
    plot_roc_all(roc_data, DATA_DIR / "multi_roc_comparison.png")

    # ── 9. Best model interpretability ────────────────────────────────
    print(f"\n[最优模型] {best_name}  Test AUC={best_auc_val:.4f}")

    m_best_def = evaluate(y_test, best_y_prob, 0.5)
    m_best_opt = evaluate(y_test, best_y_prob,
                          youden_threshold(y_test, best_y_prob))

    # Feature importance
    clf_obj = best_pipe.named_steps["clf"]
    if hasattr(clf_obj, "feature_importances_"):
        imp_vals = clf_obj.feature_importances_
        imp_label = "Feature Importance"
    elif hasattr(clf_obj, "coef_"):
        imp_vals = np.abs(clf_obj.coef_[0])
        imp_label = "|Coefficient|"
    else:
        imp_vals = None

    if imp_vals is not None:
        print(f"[plot] {best_name} Feature Importance ...")
        plot_importance(
            imp_vals.tolist(), final_features,
            imp_label,
            f"{best_name} Feature Importance (top 25)",
            DATA_DIR / "multi_best_feature_importance.png")

    print(f"[plot] Permutation Importance ...")
    perm_result = permutation_importance(
        best_pipe, X_test, y_test,
        scoring="roc_auc", n_repeats=20,
        random_state=RANDOM_STATE, n_jobs=-1)
    plot_perm_importance(perm_result, final_features,
                         DATA_DIR / "multi_best_permutation_importance.png")

    print(f"[plot] Calibration ...")
    plot_calibration(y_test, best_y_prob, best_name,
                     DATA_DIR / "multi_best_calibration.png")

    # SHAP
    print(f"[SHAP] 计算 {best_name} SHAP values ...")
    imp_step = best_pipe.named_steps["imputer"]
    X_test_imp = pd.DataFrame(
        imp_step.transform(X_test),
        columns=final_features, index=X_test.index)

    if "scaler" in best_pipe.named_steps:
        X_test_scaled = pd.DataFrame(
            best_pipe.named_steps["scaler"].transform(X_test_imp),
            columns=final_features, index=X_test.index)
    else:
        X_test_scaled = X_test_imp

    tree_models = (RandomForestClassifier, XGBClassifier, LGBMClassifier)
    if isinstance(clf_obj, tree_models):
        explainer = shap.TreeExplainer(clf_obj)
        shap_values = explainer(X_test_imp)
        if shap_values.values.ndim == 3:
            shap_pos = shap_values[..., 1]
        else:
            shap_pos = shap_values
    elif hasattr(clf_obj, "coef_"):
        explainer = shap.LinearExplainer(clf_obj, X_test_scaled)
        shap_values = explainer(X_test_scaled)
        shap_pos = shap_values
    else:
        explainer = shap.KernelExplainer(
            best_pipe.predict_proba,
            shap.sample(X_test_imp, min(100, len(X_test_imp))))
        shap_values_raw = explainer.shap_values(X_test_imp)
        if isinstance(shap_values_raw, list):
            sv = shap_values_raw[1]
        else:
            sv = shap_values_raw
        shap_pos = shap.Explanation(
            values=sv, base_values=np.full(len(sv), explainer.expected_value[1]
                                           if isinstance(explainer.expected_value, (list, np.ndarray))
                                           else explainer.expected_value),
            data=X_test_imp.values, feature_names=final_features)

    print(f"[SHAP] summary plot ...")
    plot_shap_summary(shap_pos, X_test_imp,
                      DATA_DIR / "multi_best_shap_summary.png")

    mean_abs_shap = np.abs(shap_pos.values).mean(axis=0)
    top5 = np.argsort(mean_abs_shap)[::-1][:5]
    for rank, idx in enumerate(top5, 1):
        fname = final_features[idx]
        out_path = DATA_DIR / f"multi_best_shap_dependence_{rank}_{fname}.png"
        print(f"[SHAP] 依赖图 #{rank}: {fname}")
        plot_shap_dep(shap_pos, X_test_imp, fname, out_path)

    # ── 10. Results JSON ──────────────────────────────────────────────
    perm_imp_dict = dict(zip(final_features,
                             [float(x) for x in perm_result.importances_mean]))
    shap_imp_dict = dict(zip(final_features,
                             [float(x) for x in mean_abs_shap]))

    results = {
        "best_model": best_name,
        "final_features": final_features,
        "n_features": len(final_features),
        "consensus_vote_threshold": VOTE_THRESHOLD,
        "rfe_per_learner": rfe_summary,
        "best_cv_auc_mean": float(np.nanmean(
            [r["CV_AUC_mean"] for r in rows_default
             if r["Model"] == best_name])),
        "best_test_auc": best_auc_val,
        "test_threshold_0.5": m_best_def,
        "test_threshold_youden": m_best_opt,
        "all_model_comparison": rows_default,
        "feature_importance_permutation": perm_imp_dict,
        "feature_importance_shap_mean_abs": shap_imp_dict,
    }
    with open(DATA_DIR / "multi_model_results.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  多模型对比 + 可解释性分析完成  ({elapsed:.1f}s)")
    print(f"  最终特征数     = {len(final_features)}")
    print(f"  最优模型       = {best_name}")
    print(f"  最优 Test AUC  = {best_auc_val:.4f}")
    print(f"{'='*60}")
    print(f"\n[完成] 训练矩阵        -> {DATA_DIR / 'multi_model_train.csv'}")
    print(f"[完成] 测试矩阵        -> {DATA_DIR / 'multi_model_test.csv'}")
    print(f"[完成] RFE 共识 JSON   -> {DATA_DIR / 'multi_rfe_consensus.json'}")
    print(f"[完成] RFE 逐学习器    -> {DATA_DIR / 'multi_rfe_per_learner.csv'}")
    print(f"[完成] 模型对比（0.5） -> {DATA_DIR / 'multi_model_comparison.csv'}")
    print(f"[完成] 模型对比（最优）-> {DATA_DIR / 'multi_model_comparison_optimal.csv'}")
    print(f"[完成] 结果 JSON       -> {DATA_DIR / 'multi_model_results.json'}")
    print(f"[完成] ROC 曲线        -> {DATA_DIR / 'multi_roc_comparison.png'}")
    print(f"[完成] 最优模型重要性  -> {DATA_DIR / 'multi_best_feature_importance.png'}")
    print(f"[完成] Perm 重要性     -> {DATA_DIR / 'multi_best_permutation_importance.png'}")
    print(f"[完成] SHAP 汇总       -> {DATA_DIR / 'multi_best_shap_summary.png'}")
    print(f"[完成] 校准曲线        -> {DATA_DIR / 'multi_best_calibration.png'}")


if __name__ == "__main__":
    main()
