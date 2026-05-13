"""
建模 - 步骤 8：高级特征工程 + 特征筛选 + 7 模型对比 + Stacking + 可解释性
=========================================================================

在全量原始特征的基础上做高级特征工程（交互项、比值、复合指标、患者级聚合、
log 变换），然后通过 LightGBM importance 筛选 top-50 特征，再对比 7 个
单模型 + Stacking 集成，并输出最优模型的可解释性分析。

特征工程亮点：
  - 体积比值（la/nc/ff 与 whole、vessel 的比值）
  - 狭窄 × FFRct 交互项（ffrct_stenosis、stenosis_inv_lumen 等）
  - 高危斑块易损性复合评分及其与狭窄/FFRct 的交互
  - 实验室指标比值（NLR、TG/HDL 等）
  - 斑块负荷 × 血流动力学交互
  - 患者级聚合特征（n_lesions、max_stenosis、mean_ffrct、total_vol 等）
  - log 变换处理右偏分布
  - 分类变量 one-hot

输入：data/all_variable_step1.csv
输出（均在 data/ 目录下）：
  advfe_model_train.csv / advfe_model_test.csv
  advfe_selected_features.json
  advfe_model_comparison.csv / advfe_model_comparison_optimal.csv
  advfe_model_results.json
  advfe_roc_comparison.png
  advfe_best_feature_importance.png / advfe_best_permutation_importance.png
  advfe_best_shap_summary.png / advfe_best_shap_dependence_*.png
  advfe_best_calibration.png
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
TOP_N_FEATURES = 50

EXCLUDE_COLS = {TARGET_COL, ID_COL, "artery_vessel", "lesion_segment",
                "contact_segment"}
ONEHOT_COLS = ["lesion_stenosis_grade", "plaque_type"]
SCALE_MODELS = {"KNN", "LR", "SVM"}


# ===================================================================
# Helpers
# ===================================================================

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


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                       labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "AUC":         float(roc_auc_score(y_true, y_prob)),
        "Accuracy":    float(accuracy_score(y_true, y_pred)),
        "Precision":   float(precision_score(y_true, y_pred,
                                             zero_division=0)),
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

def engineer_features(d: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering in one function, operates in-place."""
    hr = ["positive_remodeling_flag", "low_attenuation_plaque_flag",
          "napkin_ring_sign_flag", "spotty_calcification_flag"]

    # Vulnerability composite
    d["vulnerability_score"] = d[hr].fillna(0).sum(axis=1)
    d["extreme_risk_ge2"] = (d["vulnerability_score"] >= 2).astype(int)
    d["extreme_risk_ge3"] = (d["vulnerability_score"] >= 3).astype(int)

    # FFRct categorical
    d["ffrct_class3"] = np.where(
        d["ffrct_value"] >= 0.8, 0,
        np.where(d["ffrct_value"] >= 0.7, 1, 2)).astype(float)

    # Stenosis-based interactions
    d["stenosis_sq"] = d["stenosis_percent"] ** 2
    d["stenosis_inv_lumen"] = d["stenosis_percent"] / (
        d["minimum_luminal_area_mm3"] + 1e-6)
    d["ffrct_stenosis"] = (1 - d["ffrct_value"]) * d["stenosis_percent"]
    d["ffrct_inv"] = 1 / (d["ffrct_value"] + 0.01)
    d["vuln_x_stenosis"] = d["vulnerability_score"] * d["stenosis_percent"]
    d["vuln_x_ffrct_inv"] = d["vulnerability_score"] * (
        1 - d["ffrct_value"])
    d["stenosis_severe"] = (d["stenosis_percent"] >= 70).astype(int)

    # Volume ratios
    for num, den, nm in [
        ("low_attenuation_volume_mm3", "whole_lesion_volume_mm3",
         "la_whole_ratio"),
        ("non_calcified_volume_mm3", "whole_lesion_volume_mm3",
         "nc_whole_ratio"),
        ("fibrous_fatty_volume_mm3", "whole_lesion_volume_mm3",
         "ff_whole_ratio"),
        ("fibrotic_volume_mm3", "whole_lesion_volume_mm3",
         "fib_whole_ratio"),
        ("non_calcified_volume_mm3", "calcified_volume_mm3",
         "nc_calc_ratio"),
        ("lumen_volume_mm3", "vessel_volume_mm3", "lumen_vessel_ratio"),
        ("whole_lesion_volume_mm3", "vessel_volume_mm3",
         "lesion_vessel_ratio"),
    ]:
        d[nm] = d[num] / (d[den] + 1e-6)

    # Plaque composition
    d["soft_plaque_pct"] = (
        d["low_attenuation_volume_mm3"] + d["fibrous_fatty_volume_mm3"]
    ) / (d["whole_lesion_volume_mm3"] + 1e-6)
    d["hard_plaque_pct"] = (
        d["calcified_volume_mm3"] + d["fibrotic_volume_mm3"]
    ) / (d["whole_lesion_volume_mm3"] + 1e-6)
    d["soft_hard_ratio"] = d["soft_plaque_pct"] / (
        d["hard_plaque_pct"] + 1e-6)

    # Burden × hemodynamics interactions
    d["nc_burden_x_stenosis"] = (
        d["non_calcified_plaque_burden"] * d["stenosis_percent"])
    d["la_burden_x_ffrct"] = (
        d["low_attenuation_plaque_burden"] * (1 - d["ffrct_value"]))
    d["total_burden_x_vuln"] = (
        d["whole_lesion_plaque_burden"] * d["vulnerability_score"])

    # Lab-value ratios
    d["nlr"] = d["baseline_neutrophil_pct"] / (
        d["baseline_lymphocyte_pct"] + 1e-6)
    d["tg_hdl_ratio"] = d["baseline_triglycerides"] / (
        d["baseline_hdl_cholesterol"] + 1e-6)
    d["tc_hdl_ratio"] = d["baseline_total_cholesterol"] / (
        d["baseline_hdl_cholesterol"] + 1e-6)
    d["crp_x_wbc"] = d["baseline_crp"].fillna(0) * \
        d["baseline_wbc"].fillna(0)
    d["hba1c_x_dm"] = d["baseline_hba1c"].fillna(0) * \
        d["history_diabetes_new"].fillna(0)

    # Mean HU threshold
    d["mean_hu_lt_minus70"] = (d["mean_hu（pcat_hu）"] < -70).astype(int)
    d["mean_hu_x_vuln"] = d["mean_hu（pcat_hu）"] * d["vulnerability_score"]

    # Composite risk score
    d["composite_risk"] = (
        (d["stenosis_percent"] / 100) * 0.4
        + (1 - d["ffrct_value"]) * 0.4
        + (d["vulnerability_score"] / 4) * 0.2)

    # Percentage-change features (baseline → followup)
    for c in ["creatinine", "uric_acid", "crp", "wbc",
              "neutrophil_pct", "lymphocyte_pct",
              "total_cholesterol", "hdl_cholesterol", "triglycerides"]:
        bc, dc = f"baseline_{c}", f"diff_{c}"
        if bc in d.columns and dc in d.columns:
            d[f"pct_change_{c}"] = d[dc] / (d[bc].abs() + 1e-6)

    # Log transforms
    for c in ["baseline_ctnt", "baseline_nt_pro_bnp", "baseline_crp",
              "low_attenuation_volume_mm3", "non_calcified_volume_mm3",
              "whole_lesion_volume_mm3", "contact_volume_mm3",
              "lumen_volume_mm3"]:
        if c in d.columns:
            d[f"log_{c}"] = np.log1p(d[c].fillna(0).clip(lower=0))

    # Patient-level aggregation
    d["n_lesions"] = d.groupby(ID_COL)[TARGET_COL].transform("count")
    d["pt_max_sten"] = d.groupby(ID_COL)["stenosis_percent"].transform(
        "max")
    d["is_worst"] = (
        d["stenosis_percent"] == d["pt_max_sten"]).astype(int)
    d["pt_total_vol"] = d.groupby(ID_COL)[
        "whole_lesion_volume_mm3"].transform("sum")
    d["pt_mean_ffrct"] = d.groupby(ID_COL)["ffrct_value"].transform(
        "mean")
    d["pt_any_hrp"] = d.groupby(ID_COL)[
        "High-Risk Plaque Flag"].transform("max")

    return d


# ===================================================================
# Plotting
# ===================================================================

def plot_roc_all(roc_data, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc_val) in sorted(
            roc_data.items(), key=lambda kv: -kv[1][2]):
        ax.plot(fpr, tpr, label=f"{name}  AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Test-set ROC — Advanced FE + Top-50 Features")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(values, names, xlabel, title, path, top_n=30):
    idx = np.argsort(values)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, max(5, len(idx) * 0.32)))
    ax.barh(range(len(idx)), np.array(values)[idx][::-1], color="#3b82f6")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx][::-1], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_perm_importance(result, features, path, top_n=30):
    si = result.importances_mean.argsort()[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, max(5, len(si) * 0.32)))
    ax.barh(range(len(si)),
            result.importances_mean[si][::-1],
            xerr=result.importances_std[si][::-1], color="#10b981")
    ax.set_yticks(range(len(si)))
    ax.set_yticklabels([features[i] for i in si][::-1], fontsize=8)
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
    plt.figure(figsize=(10, max(7, len(X.columns) * 0.28)))
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
# Models
# ===================================================================

def build_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=1),
        "LR": LogisticRegression(
            penalty="l2", solver="liblinear", max_iter=2000,
            class_weight="balanced", random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", n_jobs=1,
            random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=5.5, eval_metric="auc",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=1),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5,
            random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(
            n_estimators=500, max_depth=-1, learning_rate=0.05,
            num_leaves=31, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
        "SVM": SVC(C=1.0, kernel="rbf", probability=True,
                   class_weight="balanced",
                   random_state=RANDOM_STATE),
    }


def make_pipeline(name, estimator):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name in SCALE_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


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

    # ── 1. Split BEFORE engineering (prevent patient-level leakage) ──
    train, test = stratified_group_split(
        df, TARGET_COL, ID_COL, 0.2, RANDOM_STATE)

    # Keep raw train for one-hot reference
    train_raw = train.copy()

    # ── 2. Feature engineering ─────────────────────────────────────────
    train = engineer_features(train)
    test = engineer_features(test)

    # ── 3. One-hot ─────────────────────────────────────────────────────
    oh_new = []
    for c in ONEHOT_COLS:
        if c not in train.columns:
            continue
        cats = sorted(train_raw[c].dropna().astype(str).unique())
        for cat in cats:
            nm = f"{c}__{cat}"
            train[nm] = (train[c].astype(str) == cat).astype(int)
            test[nm] = ((test[c].astype(str) == cat).astype(int)
                        if c in test.columns else 0)
            oh_new.append(nm)
        train.drop(columns=[c], inplace=True)
        if c in test.columns:
            test.drop(columns=[c], inplace=True)
    if oh_new:
        print(f"[onehot] -> {oh_new}")

    print(f"训练集 {train.shape} (事件率 {train[TARGET_COL].mean():.2%})")
    print(f"测试集 {test.shape} (事件率 {test[TARGET_COL].mean():.2%})")

    # ── 4. Select numeric features ─────────────────────────────────────
    all_feats = [c for c in train.columns
                 if c not in EXCLUDE_COLS
                 and train[c].dtype in ("float64", "int64")
                 and c in test.columns]
    all_feats = list(dict.fromkeys(all_feats))
    print(f"工程后全量特征数：{len(all_feats)}")

    imputer = SimpleImputer(strategy="median")
    train[all_feats] = imputer.fit_transform(train[all_feats])
    test[all_feats] = imputer.transform(test[all_feats])
    y_train = train[TARGET_COL].astype(int).values
    y_test = test[TARGET_COL].astype(int).values

    # ── 5. Feature selection (LightGBM importance → top-N) ─────────────
    lgb_selector = LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=15,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=1, verbose=-1)
    lgb_selector.fit(train[all_feats].values, y_train)
    importances = lgb_selector.feature_importances_
    top_idx = np.argsort(importances)[::-1][:TOP_N_FEATURES]
    features = [all_feats[i] for i in top_idx]

    print(f"\n[特征筛选] LightGBM importance top-{TOP_N_FEATURES}：")
    for i, idx in enumerate(top_idx[:20]):
        print(f"  {i+1:2d}. {all_feats[idx]:<45s}  "
              f"importance={importances[idx]}")

    # ── 6. Save train / test matrices ─────────────────────────────────
    keep = ([ID_COL] if ID_COL in train.columns else []) \
        + [TARGET_COL] + features
    keep = [c for c in dict.fromkeys(keep) if c in train.columns]
    train[keep].to_csv(DATA_DIR / "advfe_model_train.csv",
                       index=False, encoding="utf-8-sig")
    test[keep].to_csv(DATA_DIR / "advfe_model_test.csv",
                      index=False, encoding="utf-8-sig")

    sel_json = {
        "all_features_count": len(all_feats),
        "selected_features_count": len(features),
        "selection_method": f"LightGBM_importance_top{TOP_N_FEATURES}",
        "selected_features": features,
    }
    with open(DATA_DIR / "advfe_selected_features.json", "w",
              encoding="utf-8") as f:
        json.dump(sel_json, f, ensure_ascii=False, indent=2)

    X_train = train[features].copy()
    X_test = test[features].copy()

    # ── 7. 7-model comparison ─────────────────────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    rows_default, rows_optimal = [], []
    roc_data = {}
    best_name, best_auc_val, best_pipe, best_y_prob = "", -1.0, None, None

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

        rows_default.append({
            "Model": name,
            "CV_AUC_mean": float(np.nanmean(cv_auc)),
            "CV_AUC_std": float(np.nanstd(cv_auc)),
            **m_def})
        rows_optimal.append({
            "Model": name,
            "CV_AUC_mean": float(np.nanmean(cv_auc)),
            "CV_AUC_std": float(np.nanstd(cv_auc)),
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

    # ── 8. Stacking ensemble ──────────────────────────────────────────
    print(f"\n[Stacking] 5 base learners -> LR meta ...", flush=True)
    t0 = time.time()
    stack_estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", n_jobs=1,
            random_state=RANDOM_STATE)),
        ("lgb", LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=15,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=1, verbose=-1)),
        ("xgb", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=5.5, eval_metric="auc",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=1)),
        ("et", ExtraTreesClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", n_jobs=1,
            random_state=RANDOM_STATE)),
        ("gbm", GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE)),
    ]
    stack = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(max_iter=2000,
                                           random_state=RANDOM_STATE),
        cv=cv, stack_method="predict_proba", passthrough=False,
        n_jobs=1)

    stack_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("stack", stack),
    ])

    cv_stack = cross_val_score(
        stack_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    stack_pipe.fit(X_train, y_train)
    y_stack = stack_pipe.predict_proba(X_test)[:, 1]

    m_stack_def = evaluate(y_test, y_stack, 0.5)
    thr_stack = youden_threshold(y_test, y_stack)
    m_stack_opt = evaluate(y_test, y_stack, thr_stack)

    rows_default.append({
        "Model": "Stacking",
        "CV_AUC_mean": float(np.mean(cv_stack)),
        "CV_AUC_std": float(np.std(cv_stack)),
        **m_stack_def})
    rows_optimal.append({
        "Model": "Stacking",
        "CV_AUC_mean": float(np.mean(cv_stack)),
        "CV_AUC_std": float(np.std(cv_stack)),
        **m_stack_opt})

    fpr_s, tpr_s, _ = roc_curve(y_test, y_stack)
    roc_data["Stacking"] = (fpr_s, tpr_s, m_stack_def["AUC"])

    print(f"  Stacking  CV_AUC={np.mean(cv_stack):.4f}±"
          f"{np.std(cv_stack):.4f}  Test_AUC={m_stack_def['AUC']:.4f}  "
          f"Sens={m_stack_def['Sensitivity']:.3f}  "
          f"Spec={m_stack_def['Specificity']:.3f}  "
          f"({time.time()-t0:.1f}s)")

    if m_stack_def["AUC"] > best_auc_val:
        best_auc_val = m_stack_def["AUC"]
        best_name = "Stacking"
        best_pipe = stack_pipe
        best_y_prob = y_stack

    # Save comparison tables
    cmp_def = pd.DataFrame(rows_default).sort_values("AUC",
                                                      ascending=False)
    cmp_opt = pd.DataFrame(rows_optimal).sort_values("AUC",
                                                      ascending=False)
    cmp_def.to_csv(DATA_DIR / "advfe_model_comparison.csv",
                   index=False, encoding="utf-8-sig")
    cmp_opt.to_csv(DATA_DIR / "advfe_model_comparison_optimal.csv",
                   index=False, encoding="utf-8-sig")

    print(f"\n=== 测试集对比（阈值=0.5，按 AUC 排序） ===")
    print(cmp_def.to_string(index=False,
                            float_format=lambda x: f"{x:.4f}"))
    print(f"\n=== 测试集对比（阈值=Youden 最优） ===")
    print(cmp_opt.to_string(index=False,
                            float_format=lambda x: f"{x:.4f}"))

    # ── 9. Plots ──────────────────────────────────────────────────────
    print(f"\n[plot] ROC 曲线 ...")
    plot_roc_all(roc_data, DATA_DIR / "advfe_roc_comparison.png")

    # ── 10. Best model interpretability ───────────────────────────────
    print(f"\n[最优模型] {best_name}  Test AUC={best_auc_val:.4f}")
    m_best_def = evaluate(y_test, best_y_prob, 0.5)
    m_best_opt = evaluate(y_test, best_y_prob,
                          youden_threshold(y_test, best_y_prob))

    # For interpretability, use LightGBM (interpretable tree model)
    lgb_final = LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=15,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=1, verbose=-1)
    lgb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", lgb_final),
    ])
    lgb_pipe.fit(X_train, y_train)
    lgb_model = lgb_pipe.named_steps["clf"]

    print(f"[plot] Feature Importance (LightGBM) ...")
    plot_importance(
        lgb_model.feature_importances_.tolist(), features,
        "Feature Importance (split count)",
        f"LightGBM Feature Importance (top 30)",
        DATA_DIR / "advfe_best_feature_importance.png")

    print(f"[plot] Permutation Importance ...")
    perm_result = permutation_importance(
        best_pipe, X_test, y_test,
        scoring="roc_auc", n_repeats=20,
        random_state=RANDOM_STATE, n_jobs=-1)
    plot_perm_importance(perm_result, features,
                         DATA_DIR / "advfe_best_permutation_importance.png")

    print(f"[plot] Calibration ...")
    plot_calibration(y_test, best_y_prob, best_name,
                     DATA_DIR / "advfe_best_calibration.png")

    # SHAP (use LightGBM for tree-based SHAP)
    print(f"[SHAP] TreeExplainer (LightGBM) ...")
    X_test_imp = pd.DataFrame(
        lgb_pipe.named_steps["imputer"].transform(X_test),
        columns=features, index=X_test.index)
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer(X_test_imp)
    shap_pos = shap_values

    print(f"[SHAP] summary plot ...")
    plot_shap_summary(shap_pos, X_test_imp,
                      DATA_DIR / "advfe_best_shap_summary.png")

    mean_abs_shap = np.abs(shap_pos.values).mean(axis=0)
    top5 = np.argsort(mean_abs_shap)[::-1][:5]
    for rank, idx in enumerate(top5, 1):
        fname = features[idx]
        out_path = DATA_DIR / \
            f"advfe_best_shap_dependence_{rank}_{fname}.png"
        print(f"[SHAP] 依赖图 #{rank}: {fname}")
        plot_shap_dep(shap_pos, X_test_imp, fname, out_path)

    # ── 11. Results JSON ──────────────────────────────────────────────
    perm_dict = dict(zip(features,
                         [float(x) for x in
                          perm_result.importances_mean]))
    shap_dict = dict(zip(features,
                         [float(x) for x in mean_abs_shap]))

    results = {
        "mode": "advanced_feature_engineering",
        "total_engineered_features": len(all_feats),
        "selected_features_count": len(features),
        "selection_method": f"LightGBM_importance_top{TOP_N_FEATURES}",
        "selected_features": features,
        "best_model": best_name,
        "best_cv_auc_mean": float(np.nanmean(
            [r["CV_AUC_mean"] for r in rows_default
             if r["Model"] == best_name])),
        "best_test_auc": best_auc_val,
        "test_threshold_0.5": m_best_def,
        "test_threshold_youden": m_best_opt,
        "all_model_comparison": rows_default,
        "feature_importance_permutation": perm_dict,
        "feature_importance_shap_mean_abs": shap_dict,
    }
    with open(DATA_DIR / "advfe_model_results.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  高级特征工程 + 7模型 + Stacking 完成  ({elapsed:.1f}s)")
    print(f"  工程后特征数   = {len(all_feats)}")
    print(f"  筛选后特征数   = {len(features)}")
    print(f"  最优模型       = {best_name}")
    print(f"  最优 Test AUC  = {best_auc_val:.4f}")
    print(f"{'='*60}")
    for f_name in [
        "advfe_model_train.csv", "advfe_model_test.csv",
        "advfe_selected_features.json",
        "advfe_model_comparison.csv",
        "advfe_model_comparison_optimal.csv",
        "advfe_model_results.json",
        "advfe_roc_comparison.png",
        "advfe_best_feature_importance.png",
        "advfe_best_permutation_importance.png",
        "advfe_best_shap_summary.png",
        "advfe_best_calibration.png",
    ]:
        print(f"[完成] {DATA_DIR / f_name}")


if __name__ == "__main__":
    main()
