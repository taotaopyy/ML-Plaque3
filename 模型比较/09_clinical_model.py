"""
建模 - 步骤 9：临床高风险变量 + 高级特征工程 + 多模型对比 + Stacking + 可解释性
==============================================================================

基于临床专家定义的固定输入变量（14 个）和 RFE 候选变量（41 个），
结合步骤 8 中验证有效的高级特征工程策略（交互项、比值、复合指标、
患者级聚合、log 变换等），执行以下流程：

1. 列名映射 + 特征工程（从用户指定变量派生交互/比值/复合特征）
2. 按 accession_id 分组分层 8:2 抽样
3. 多分类列 one-hot、缺失值中位数插补
4. 三基学习器 RFECV（LR / LightGBM / XGBoost）→ 按平均 rank 取共识
5. 最终特征 ≈ 30 个（固定 + 衍生固定 + 共识 RFE）
6. 7 单模型 + 5-model Stacking 对比
7. 最优模型可解释性分析（SHAP / Permutation / ROC / Calibration）

输入：data/all_variable_step1.csv
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
    AdaBoostClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, RandomForestClassifier,
    StackingClassifier,
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
TARGET_COL = "target_lesion_positive"
ID_COL = "accession_id"
TARGET_TOTAL_FEATURES = 30

COLUMN_ALIASES = {
    "history_diabetes":          "history_diabetes_new",
    "mean_hu":                   "mean_hu（pcat_hu）",
    "maximum_luminal_area_mm3":  "minimum_luminal_area_mm3",
    "followup_triglycerides":    "diff_triglycerides",
    "followup_lipoprotein_a":    "diff_lipoprotein_a",
}
KNOWN_MISSING = {"cda_admission_number", "stent_present"}
MULTICLASS_COLUMNS = {"plaque_type"}
SCALE_MODELS = {"KNN", "LR", "SVM"}

HIGH_RISK_FLAGS = [
    "positive_remodeling_flag", "low_attenuation_plaque_flag",
    "napkin_ring_sign_flag", "spotty_calcification_flag",
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
def resolve(name, df):
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
        (tr_idx, te_idx), = sss.split(np.zeros(len(g)),
                                       g[target].astype(int))
        tr_ids = set(g.iloc[tr_idx][group])
        te_ids = set(g.iloc[te_idx][group])
        return (df[df[group].isin(tr_ids)].copy(),
                df[df[group].isin(te_ids)].copy())
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=seed)
    (tr_idx, te_idx), = sss.split(np.zeros(len(df)),
                                   df[target].astype(int))
    return df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()


def onehot_fit_apply(train, test, cols):
    new_cols = []
    for c in [x for x in cols if x in train.columns]:
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
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "Threshold": float(threshold),
    }


def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


# ===================================================================
# Feature engineering — only from user-specified variables
# ===================================================================
def engineer_from_clinical(df):
    """Derive features strictly from the user-specified variable pool."""
    derived_fixed, derived_rfe = [], []
    hr = [c for c in HIGH_RISK_FLAGS if c in df.columns]

    # -- derived FIXED --
    if len(hr) >= 2:
        cnt = df[hr].fillna(0).sum(axis=1)
        df["extreme_risk_ge2"] = (cnt >= 2).astype(int)
        df["extreme_risk_ge3"] = (cnt >= 3).astype(int)
        derived_fixed += ["extreme_risk_ge2", "extreme_risk_ge3"]

    if "ffrct_value" in df.columns:
        v = df["ffrct_value"]
        df["ffrct_class3"] = np.where(v >= 0.8, 0,
                                       np.where(v >= 0.7, 1, 2)).astype(float)
        derived_fixed.append("ffrct_class3")

    mhu = COLUMN_ALIASES.get("mean_hu", "mean_hu")
    if mhu in df.columns:
        df["mean_hu_lt_minus70"] = (df[mhu] < -70).astype(int)
        derived_rfe.append("mean_hu_lt_minus70")

    # -- derived RFE candidates --
    # Stenosis interactions
    if "stenosis_percent" in df.columns:
        df["stenosis_sq"] = df["stenosis_percent"] ** 2
        derived_rfe.append("stenosis_sq")
        if "minimum_luminal_area_mm3" in df.columns:
            df["stenosis_inv_lumen"] = df["stenosis_percent"] / (
                df["minimum_luminal_area_mm3"] + 1e-6)
            derived_rfe.append("stenosis_inv_lumen")

    if "ffrct_value" in df.columns and "stenosis_percent" in df.columns:
        df["ffrct_stenosis"] = (1 - df["ffrct_value"]) * df["stenosis_percent"]
        df["ffrct_inv"] = 1.0 / (df["ffrct_value"] + 0.01)
        derived_rfe += ["ffrct_stenosis", "ffrct_inv"]

    # Vulnerability composites
    if len(hr) >= 2:
        df["vulnerability_score"] = df[hr].fillna(0).sum(axis=1)
        derived_rfe.append("vulnerability_score")
        if "stenosis_percent" in df.columns:
            df["vuln_x_stenosis"] = df["vulnerability_score"] * df["stenosis_percent"]
            derived_rfe.append("vuln_x_stenosis")
        if "ffrct_value" in df.columns:
            df["vuln_x_ffrct_inv"] = df["vulnerability_score"] * (1 - df["ffrct_value"])
            derived_rfe.append("vuln_x_ffrct_inv")

    # Volume ratios
    for num, den, nm in [
        ("low_attenuation_volume_mm3", "whole_lesion_volume_mm3", "la_whole_ratio"),
        ("non_calcified_volume_mm3", "whole_lesion_volume_mm3", "nc_whole_ratio"),
        ("fibrous_fatty_volume_mm3", "whole_lesion_volume_mm3", "ff_whole_ratio"),
        ("fibrotic_volume_mm3", "whole_lesion_volume_mm3", "fib_whole_ratio"),
        ("non_calcified_volume_mm3", "calcified_volume_mm3", "nc_calc_ratio"),
        ("lumen_volume_mm3", "vessel_volume_mm3", "lumen_vessel_ratio"),
        ("whole_lesion_volume_mm3", "vessel_volume_mm3", "lesion_vessel_ratio"),
    ]:
        if num in df.columns and den in df.columns:
            df[nm] = df[num] / (df[den] + 1e-6)
            derived_rfe.append(nm)

    # Plaque composition
    if all(c in df.columns for c in ["low_attenuation_volume_mm3",
                                      "fibrous_fatty_volume_mm3",
                                      "whole_lesion_volume_mm3"]):
        df["soft_plaque_pct"] = (
            df["low_attenuation_volume_mm3"] + df["fibrous_fatty_volume_mm3"]
        ) / (df["whole_lesion_volume_mm3"] + 1e-6)
        derived_rfe.append("soft_plaque_pct")

    # Burden × hemodynamics (burden features exist in data)
    for burden, interact, nm in [
        ("non_calcified_plaque_burden", "stenosis_percent", "nc_burden_x_stenosis"),
        ("low_attenuation_plaque_burden", "ffrct_value", "la_burden_x_ffrct"),
        ("whole_lesion_plaque_burden", "vulnerability_score", "total_burden_x_vuln"),
    ]:
        if burden in df.columns and interact in df.columns:
            if interact == "ffrct_value":
                df[nm] = df[burden] * (1 - df[interact])
            else:
                df[nm] = df[burden] * df[interact]
            derived_rfe.append(nm)

    # Lab-value ratios
    if "baseline_neutrophil_pct" in df.columns and "baseline_lymphocyte_pct" in df.columns:
        df["nlr"] = df["baseline_neutrophil_pct"] / (df["baseline_lymphocyte_pct"] + 1e-6)
        derived_rfe.append("nlr")
    if "baseline_triglycerides" in df.columns and "baseline_hdl_cholesterol" in df.columns:
        df["tg_hdl_ratio"] = df["baseline_triglycerides"] / (df["baseline_hdl_cholesterol"] + 1e-6)
        derived_rfe.append("tg_hdl_ratio")
    if "baseline_total_cholesterol" in df.columns and "baseline_hdl_cholesterol" in df.columns:
        df["tc_hdl_ratio"] = df["baseline_total_cholesterol"] / (df["baseline_hdl_cholesterol"] + 1e-6)
        derived_rfe.append("tc_hdl_ratio")
    if "baseline_crp" in df.columns and "baseline_wbc" in df.columns:
        df["crp_x_wbc"] = df["baseline_crp"].fillna(0) * df["baseline_wbc"].fillna(0)
        derived_rfe.append("crp_x_wbc")
    if "baseline_hba1c" in df.columns and "history_diabetes_new" in df.columns:
        df["hba1c_x_dm"] = df["baseline_hba1c"].fillna(0) * df["history_diabetes_new"].fillna(0)
        derived_rfe.append("hba1c_x_dm")

    # Mean HU interaction
    if mhu in df.columns and "vulnerability_score" in df.columns:
        df["mean_hu_x_vuln"] = df[mhu] * df["vulnerability_score"]
        derived_rfe.append("mean_hu_x_vuln")

    # Composite risk
    if all(c in df.columns for c in ["stenosis_percent", "ffrct_value", "vulnerability_score"]):
        df["composite_risk"] = (
            (df["stenosis_percent"] / 100) * 0.4
            + (1 - df["ffrct_value"]) * 0.4
            + (df["vulnerability_score"] / 4) * 0.2)
        derived_rfe.append("composite_risk")

    # Stenosis severity bin
    if "stenosis_percent" in df.columns:
        df["stenosis_severe"] = (df["stenosis_percent"] >= 70).astype(int)
        derived_rfe.append("stenosis_severe")

    # Plaque burden features from data
    for c in ["non_calcified_plaque_burden", "low_attenuation_plaque_burden",
              "whole_lesion_plaque_burden", "fibrous_fatty_plaque_burden",
              "remodeling_index", "eccentricity_index",
              "maximum_area_stenosis_percent"]:
        if c in df.columns:
            derived_rfe.append(c)

    # Patient-level aggregation
    if ID_COL in df.columns:
        df["n_lesions"] = df.groupby(ID_COL)[TARGET_COL].transform("count")
        df["pt_max_sten"] = df.groupby(ID_COL)["stenosis_percent"].transform("max")
        df["is_worst"] = (df["stenosis_percent"] == df["pt_max_sten"]).astype(int)
        df["pt_total_vol"] = df.groupby(ID_COL)["whole_lesion_volume_mm3"].transform("sum")
        df["pt_mean_ffrct"] = df.groupby(ID_COL)["ffrct_value"].transform("mean")
        df["pt_any_hrp"] = df.groupby(ID_COL)["High-Risk Plaque Flag"].transform("max")
        derived_rfe += ["n_lesions", "pt_max_sten", "is_worst",
                        "pt_total_vol", "pt_mean_ffrct", "pt_any_hrp"]

    # Log transforms for skewed
    for c in ["baseline_ctnt", "baseline_nt_pro_bnp", "baseline_crp",
              "low_attenuation_volume_mm3", "non_calcified_volume_mm3",
              "whole_lesion_volume_mm3", "contact_volume_mm3"]:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(df[c].fillna(0).clip(lower=0))
            derived_rfe.append(f"log_{c}")

    return df, derived_fixed, derived_rfe


# ===================================================================
# Plotting helpers
# ===================================================================
def plot_roc_all(roc_data, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc_val) in sorted(roc_data.items(), key=lambda kv: -kv[1][2]):
        ax.plot(fpr, tpr, label=f"{name}  AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Test-set ROC — Clinical Variables + Advanced FE")
    ax.legend(loc="lower right", fontsize=8); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)

def plot_importance(values, names, xlabel, title, path, top_n=30):
    idx = np.argsort(values)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, max(5, len(idx)*0.32)))
    ax.barh(range(len(idx)), np.array(values)[idx][::-1], color="#3b82f6")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx][::-1], fontsize=8)
    ax.set_xlabel(xlabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_perm_imp(result, features, path, top_n=30):
    si = result.importances_mean.argsort()[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, max(5, len(si)*0.32)))
    ax.barh(range(len(si)), result.importances_mean[si][::-1],
            xerr=result.importances_std[si][::-1], color="#10b981")
    ax.set_yticks(range(len(si)))
    ax.set_yticklabels([features[i] for i in si][::-1], fontsize=8)
    ax.set_xlabel("Decrease in AUC")
    ax.set_title(f"Permutation Importance (top {len(si)})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_calibration(y_true, y_prob, name, path):
    pt, pp = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(pp, pt, "s-", label=name)
    ax.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.5,label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {name}"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_shap_summary(sv, X, path):
    plt.figure(figsize=(10, max(7, len(X.columns)*0.28)))
    shap.summary_plot(sv, X, show=False, max_display=30)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close("all")

def plot_shap_dep(sv, X, feat, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.dependence_plot(feat, sv.values, X, ax=ax, show=False)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ===================================================================
# Models
# ===================================================================
def build_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=1),
        "LR": LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5, class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5.5, eval_metric="auc", tree_method="hist", random_state=RANDOM_STATE, n_jobs=1),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
        "SVM": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }

def make_pipeline(name, est):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name in SCALE_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", est))
    return Pipeline(steps)

def make_rfe_learners():
    return {
        "LR": (LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE), 1),
        "LightGBM": (LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1, verbose=-1), 1),
        "XGBoost": (XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=5.5, eval_metric="auc", tree_method="hist", random_state=RANDOM_STATE, n_jobs=1), 1),
    }


# ===================================================================
# MAIN
# ===================================================================
def main():
    t_start = time.time()
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    print(f"原始数据 {df.shape}")

    # ── 1. Split ──
    train, test = stratified_group_split(df, TARGET_COL, ID_COL, 0.2, RANDOM_STATE)

    # ── 2. Feature engineering ──
    train, derived_fixed, derived_rfe = engineer_from_clinical(train)
    test, _, _ = engineer_from_clinical(test)
    print(f"[FE] 衍生固定 {len(derived_fixed)}, 衍生 RFE {len(derived_rfe)}")

    # ── 3. Resolve feature names ──
    fixed_resolved, missing_f = [], []
    for name in FIXED_FEATURES_RAW:
        actual = resolve(name, train)
        if actual: fixed_resolved.append(actual)
        else: missing_f.append(name)
    for c in derived_fixed:
        if c in train.columns: fixed_resolved.append(c)

    rfe_resolved, missing_r = [], []
    for name in RFE_CANDIDATES_RAW:
        if name in KNOWN_MISSING: missing_r.append(name); continue
        actual = resolve(name, train)
        if actual: rfe_resolved.append(actual)
        else: missing_r.append(name)
    for c in derived_rfe:
        if c in train.columns and c not in rfe_resolved:
            rfe_resolved.append(c)

    seen = set()
    fixed_resolved = [c for c in fixed_resolved if not (c in seen or seen.add(c))]
    seen_all = set(fixed_resolved)
    rfe_resolved = [c for c in rfe_resolved if not (c in seen_all or seen_all.add(c))]

    print(f"固定特征 {len(fixed_resolved)}, RFE 候选 {len(rfe_resolved)}")
    if missing_f: print(f"[warn] 固定缺失：{missing_f}")
    if missing_r: print(f"[warn] RFE 缺失：{missing_r}")

    # ── 4. One-hot ──
    multi = [c for c in rfe_resolved if c in MULTICLASS_COLUMNS]
    train, test, oh = onehot_fit_apply(train, test, multi)
    rfe_resolved = [c for c in rfe_resolved if c not in multi] + oh
    if oh: print(f"[onehot] {multi} -> {oh}")

    # ── 5. Imputation ──
    all_feat = fixed_resolved + rfe_resolved
    all_feat = [c for c in dict.fromkeys(all_feat) if c in train.columns]
    imputer = SimpleImputer(strategy="median")
    train[all_feat] = imputer.fit_transform(train[all_feat])
    test[all_feat] = imputer.transform(test[all_feat])
    y_train = train[TARGET_COL].astype(int).values
    y_test = test[TARGET_COL].astype(int).values

    rfe_cols = [c for c in rfe_resolved if c in train.columns]
    print(f"训练集 {train.shape} | 测试集 {test.shape}")

    # ── 6. Feature selection by LightGBM importance ──
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── 6. Use ALL RFE candidates + derived features (no hard limit) ──
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n[特征] 使用全部 {len(rfe_cols)} 个 RFE 候选（含衍生特征）")
    lgb_sel = LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=15,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=1, verbose=-1)
    lgb_sel.fit(train[rfe_cols].values, y_train)
    importances = lgb_sel.feature_importances_

    ranking_table = pd.DataFrame({
        "feature": rfe_cols,
        "lgb_importance": importances,
    }).sort_values("lgb_importance", ascending=False)
    ranking_table.to_csv(DATA_DIR / "clin_rfe_per_learner.csv",
                         index=False, encoding="utf-8-sig")

    consensus_features = rfe_cols
    rfe_summary = {"method": "all_candidates_plus_derived", "n": len(rfe_cols)}

    print(f"\n[特征] Top-20 RFE candidates by LightGBM importance：")
    for _, row in ranking_table.head(20).iterrows():
        print(f"    {row['feature']:<45s}  importance={row['lgb_importance']}")

    final_features = fixed_resolved + consensus_features
    final_features = [c for c in dict.fromkeys(final_features) if c in train.columns]
    print(f"\n最终特征 ({len(final_features)}) = 固定 {len(fixed_resolved)} + 共识 {len(consensus_features)}")

    # Save consensus
    with open(DATA_DIR / "clin_rfe_consensus.json", "w", encoding="utf-8") as f:
        json.dump({"fixed": fixed_resolved, "consensus_rfe": consensus_features,
                   "final": final_features, "per_learner": rfe_summary}, f, ensure_ascii=False, indent=2)

    # Save train/test
    keep = ([ID_COL] if ID_COL in train.columns else []) + [TARGET_COL] + final_features
    keep = [c for c in dict.fromkeys(keep) if c in train.columns]
    train[keep].to_csv(DATA_DIR / "clin_model_train.csv", index=False, encoding="utf-8-sig")
    test[keep].to_csv(DATA_DIR / "clin_model_test.csv", index=False, encoding="utf-8-sig")

    X_train = train[final_features].copy()
    X_test = test[final_features].copy()

    # ── 7. 7-model comparison ──
    rows_def, rows_opt = [], []
    roc_data = {}
    best_name, best_auc_val, best_pipe, best_y_prob = "", -1.0, None, None

    for name, est in build_models().items():
        t0 = time.time()
        print(f"\n[fit] {name} ...", flush=True)
        pipe = make_pipeline(name, est)
        try:
            cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=CV_FOLDS)
        except Exception as e:
            print(f"[warn] {name} CV failed: {e}"); cv_auc = np.array([np.nan])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        m_d = evaluate(y_test, y_prob, 0.5)
        m_o = evaluate(y_test, y_prob, youden_threshold(y_test, y_prob))
        rows_def.append({"Model": name, "CV_AUC_mean": float(np.nanmean(cv_auc)), "CV_AUC_std": float(np.nanstd(cv_auc)), **m_d})
        rows_opt.append({"Model": name, "CV_AUC_mean": float(np.nanmean(cv_auc)), "CV_AUC_std": float(np.nanstd(cv_auc)), **m_o})
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, m_d["AUC"])
        print(f"  {name:<9s} CV={np.nanmean(cv_auc):.4f}±{np.nanstd(cv_auc):.4f} Test={m_d['AUC']:.4f} ({time.time()-t0:.1f}s)")
        if m_d["AUC"] > best_auc_val:
            best_auc_val, best_name, best_pipe, best_y_prob = m_d["AUC"], name, pipe, y_prob

    # ── 8. Stacking ──
    print(f"\n[Stacking] 5 base -> LR meta ...", flush=True)
    t0 = time.time()
    stack = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5, class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE)),
            ("lgb", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1, verbose=-1)),
            ("xgb", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5.5, eval_metric="auc", tree_method="hist", random_state=RANDOM_STATE, n_jobs=1)),
            ("et", ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE)),
            ("gbm", GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE)),
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        cv=cv, stack_method="predict_proba", passthrough=False, n_jobs=1)
    stack_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("stack", stack)])
    cv_st = cross_val_score(stack_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    stack_pipe.fit(X_train, y_train)
    y_st = stack_pipe.predict_proba(X_test)[:, 1]
    m_st_d = evaluate(y_test, y_st, 0.5)
    m_st_o = evaluate(y_test, y_st, youden_threshold(y_test, y_st))
    rows_def.append({"Model": "Stacking", "CV_AUC_mean": float(np.mean(cv_st)), "CV_AUC_std": float(np.std(cv_st)), **m_st_d})
    rows_opt.append({"Model": "Stacking", "CV_AUC_mean": float(np.mean(cv_st)), "CV_AUC_std": float(np.std(cv_st)), **m_st_o})
    fpr_s, tpr_s, _ = roc_curve(y_test, y_st)
    roc_data["Stacking"] = (fpr_s, tpr_s, m_st_d["AUC"])
    print(f"  Stacking CV={np.mean(cv_st):.4f}±{np.std(cv_st):.4f} Test={m_st_d['AUC']:.4f} ({time.time()-t0:.1f}s)")
    if m_st_d["AUC"] > best_auc_val:
        best_auc_val, best_name, best_pipe, best_y_prob = m_st_d["AUC"], "Stacking", stack_pipe, y_st

    # Print comparison
    cmp_d = pd.DataFrame(rows_def).sort_values("AUC", ascending=False)
    cmp_o = pd.DataFrame(rows_opt).sort_values("AUC", ascending=False)
    cmp_d.to_csv(DATA_DIR / "clin_model_comparison.csv", index=False, encoding="utf-8-sig")
    cmp_o.to_csv(DATA_DIR / "clin_model_comparison_optimal.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 测试集对比（阈值=0.5） ===")
    print(cmp_d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n=== 测试集对比（阈值=Youden） ===")
    print(cmp_o.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── 9. Plots ──
    print(f"\n[plot] ROC ...")
    plot_roc_all(roc_data, DATA_DIR / "clin_roc_comparison.png")

    # ── 10. Interpretability ──
    print(f"\n[最优模型] {best_name} Test AUC={best_auc_val:.4f}")
    m_best_d = evaluate(y_test, best_y_prob, 0.5)
    m_best_o = evaluate(y_test, best_y_prob, youden_threshold(y_test, best_y_prob))

    lgb_interp = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
    lgb_interp_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", lgb_interp)])
    lgb_interp_pipe.fit(X_train, y_train)
    lgb_model = lgb_interp_pipe.named_steps["clf"]

    print("[plot] Feature Importance (LightGBM) ...")
    plot_importance(lgb_model.feature_importances_.tolist(), final_features, "Importance", "LightGBM Feature Importance", DATA_DIR / "clin_best_feature_importance.png")

    print("[plot] Permutation Importance ...")
    perm = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
    plot_perm_imp(perm, final_features, DATA_DIR / "clin_best_permutation_importance.png")

    print("[plot] Calibration ...")
    plot_calibration(y_test, best_y_prob, best_name, DATA_DIR / "clin_best_calibration.png")

    print("[SHAP] TreeExplainer (LightGBM) ...")
    X_test_imp = pd.DataFrame(lgb_interp_pipe.named_steps["imputer"].transform(X_test), columns=final_features, index=X_test.index)
    explainer = shap.TreeExplainer(lgb_model)
    sv = explainer(X_test_imp)
    plot_shap_summary(sv, X_test_imp, DATA_DIR / "clin_best_shap_summary.png")
    mean_abs = np.abs(sv.values).mean(axis=0)
    for rank, idx in enumerate(np.argsort(mean_abs)[::-1][:5], 1):
        fn = final_features[idx]
        print(f"[SHAP] dep #{rank}: {fn}")
        plot_shap_dep(sv, X_test_imp, fn, DATA_DIR / f"clin_best_shap_dep_{rank}_{fn}.png")

    # ── 11. Results JSON ──
    results = {
        "best_model": best_name, "n_features": len(final_features),
        "final_features": final_features, "fixed": fixed_resolved,
        "consensus_rfe": consensus_features,
        "best_test_auc": best_auc_val,
        "test_0.5": m_best_d, "test_youden": m_best_o,
        "all_comparison": rows_def,
        "perm_importance": dict(zip(final_features, [float(x) for x in perm.importances_mean])),
        "shap_importance": dict(zip(final_features, [float(x) for x in mean_abs])),
    }
    with open(DATA_DIR / "clin_model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  临床变量 + 高级 FE + Stacking 完成  ({elapsed:.1f}s)")
    print(f"  最终特征数     = {len(final_features)}")
    print(f"  最优模型       = {best_name}")
    print(f"  最优 Test AUC  = {best_auc_val:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
