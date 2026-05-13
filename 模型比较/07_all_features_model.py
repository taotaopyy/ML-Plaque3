"""
建模 - 步骤 7：全量特征输入 + 7 模型对比 + 可解释性分析
======================================================

将 ``all_variable_step1.csv`` 中所有可用数值特征（含 one-hot 后的分类变量）
全部输入模型，不做 RFE 特征筛选，直接对比 7 个模型并输出最优模型的可解释性分析。

流程：
1. 读取 ``data/all_variable_step1.csv``。
2. 将字符串分类列（lesion_stenosis_grade、plaque_type）做 one-hot；
   排除 ID 列（accession_id）和文本列（artery_vessel / lesion_segment /
   contact_segment）。
3. 按 accession_id 分组分层 8:2 抽样，防止同一受试者跨集泄漏。
4. 缺失值中位数插补。
5. 在全部特征上训练并对比 7 个模型。
6. 输出最优模型的可解释性分析。

输入：data/all_variable_step1.csv
输出（均在 data/ 目录下）：
  allf_model_train.csv / allf_model_test.csv
  allf_model_comparison.csv / allf_model_comparison_optimal.csv
  allf_model_results.json
  allf_roc_comparison.png
  allf_best_feature_importance.png / allf_best_permutation_importance.png
  allf_best_shap_summary.png / allf_best_shap_dependence_*.png
  allf_best_calibration.png
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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
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

EXCLUDE_COLS = {TARGET_COL, ID_COL, "artery_vessel", "lesion_segment",
                "contact_segment"}
ONEHOT_COLS = {"lesion_stenosis_grade", "plaque_type"}
SCALE_MODELS = {"KNN", "LR", "SVM"}


# ===================================================================
# Helpers
# ===================================================================

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
    for c in cols:
        if c not in train.columns:
            continue
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
# Models
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
    ax.set_title("Test-set ROC Comparison (all features)")
    ax.legend(loc="lower right", fontsize=9)
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
            xerr=result.importances_std[si][::-1],
            color="#10b981")
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
    plt.figure(figsize=(10, max(7, len(X.columns) * 0.25)))
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

    # ── 1. Train / Test split ──────────────────────────────────────────
    train, test = stratified_group_split(
        df, TARGET_COL, ID_COL, 0.2, RANDOM_STATE)
    print(f"训练集 {train.shape} (事件率 {train[TARGET_COL].mean():.2%})")
    print(f"测试集 {test.shape} (事件率 {test[TARGET_COL].mean():.2%})")

    # ── 2. One-hot for string categorical columns ──────────────────────
    oh_cols_to_process = [c for c in ONEHOT_COLS if c in train.columns]
    train, test, oh_new = onehot_fit_apply(train, test, oh_cols_to_process)
    if oh_new:
        print(f"[onehot] {oh_cols_to_process} -> {oh_new}")

    # ── 3. Select all numeric features ─────────────────────────────────
    features = [c for c in train.columns
                if c not in EXCLUDE_COLS
                and train[c].dtype in ("float64", "int64", "float32", "int32")]
    print(f"\n全量特征数：{len(features)}")

    # ── 4. Imputation ──────────────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    train[features] = imputer.fit_transform(train[features])
    test[features]  = imputer.transform(test[features])
    y_train = train[TARGET_COL].astype(int).values
    y_test  = test[TARGET_COL].astype(int).values

    # ── 5. Save train / test matrices ─────────────────────────────────
    keep = ([ID_COL] if ID_COL in train.columns else []) + [TARGET_COL] + features
    keep = [c for c in dict.fromkeys(keep) if c in train.columns]
    train[keep].to_csv(DATA_DIR / "allf_model_train.csv",
                       index=False, encoding="utf-8-sig")
    test[keep].to_csv(DATA_DIR / "allf_model_test.csv",
                      index=False, encoding="utf-8-sig")

    X_train = train[features].copy()
    X_test  = test[features].copy()

    # ── 6. 7-model comparison ─────────────────────────────────────────
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
    cmp_def.to_csv(DATA_DIR / "allf_model_comparison.csv",
                   index=False, encoding="utf-8-sig")
    cmp_opt.to_csv(DATA_DIR / "allf_model_comparison_optimal.csv",
                   index=False, encoding="utf-8-sig")

    print(f"\n=== 测试集对比（阈值=0.5，按 AUC 排序） ===")
    print(cmp_def.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n=== 测试集对比（阈值=Youden 最优） ===")
    print(cmp_opt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── 7. Plots ──────────────────────────────────────────────────────
    print(f"\n[plot] ROC 曲线（所有模型） ...")
    plot_roc_all(roc_data, DATA_DIR / "allf_roc_comparison.png")

    # ── 8. Best model interpretability ────────────────────────────────
    print(f"\n[最优模型] {best_name}  Test AUC={best_auc_val:.4f}")

    m_best_def = evaluate(y_test, best_y_prob, 0.5)
    m_best_opt = evaluate(y_test, best_y_prob,
                          youden_threshold(y_test, best_y_prob))

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
            imp_vals.tolist(), features, imp_label,
            f"{best_name} Feature Importance (top 30, all features)",
            DATA_DIR / "allf_best_feature_importance.png")

    print(f"[plot] Permutation Importance ...")
    perm_result = permutation_importance(
        best_pipe, X_test, y_test,
        scoring="roc_auc", n_repeats=20,
        random_state=RANDOM_STATE, n_jobs=-1)
    plot_perm_importance(perm_result, features,
                         DATA_DIR / "allf_best_permutation_importance.png")

    print(f"[plot] Calibration ...")
    plot_calibration(y_test, best_y_prob, best_name,
                     DATA_DIR / "allf_best_calibration.png")

    # ── SHAP ──────────────────────────────────────────────────────────
    print(f"[SHAP] 计算 {best_name} SHAP values ...")
    imp_step = best_pipe.named_steps["imputer"]
    X_test_imp = pd.DataFrame(
        imp_step.transform(X_test),
        columns=features, index=X_test.index)

    tree_models = (RandomForestClassifier, XGBClassifier, LGBMClassifier)
    if isinstance(clf_obj, tree_models):
        explainer = shap.TreeExplainer(clf_obj)
        shap_values = explainer(X_test_imp)
        shap_pos = shap_values[..., 1] if shap_values.values.ndim == 3 \
            else shap_values
    elif hasattr(clf_obj, "coef_"):
        if "scaler" in best_pipe.named_steps:
            X_scaled = pd.DataFrame(
                best_pipe.named_steps["scaler"].transform(X_test_imp),
                columns=features, index=X_test.index)
        else:
            X_scaled = X_test_imp
        explainer = shap.LinearExplainer(clf_obj, X_scaled)
        shap_pos = explainer(X_scaled)
    else:
        bg = shap.sample(X_test_imp, min(100, len(X_test_imp)))
        explainer = shap.KernelExplainer(best_pipe.predict_proba, bg)
        sv = explainer.shap_values(X_test_imp)
        sv1 = sv[1] if isinstance(sv, list) else sv
        ev = (explainer.expected_value[1]
              if isinstance(explainer.expected_value, (list, np.ndarray))
              else explainer.expected_value)
        shap_pos = shap.Explanation(
            values=sv1, base_values=np.full(len(sv1), ev),
            data=X_test_imp.values, feature_names=features)

    print(f"[SHAP] summary plot ...")
    plot_shap_summary(shap_pos, X_test_imp,
                      DATA_DIR / "allf_best_shap_summary.png")

    mean_abs_shap = np.abs(shap_pos.values).mean(axis=0)
    top5 = np.argsort(mean_abs_shap)[::-1][:5]
    for rank, idx in enumerate(top5, 1):
        fname = features[idx]
        out_path = DATA_DIR / f"allf_best_shap_dependence_{rank}_{fname}.png"
        print(f"[SHAP] 依赖图 #{rank}: {fname}")
        plot_shap_dep(shap_pos, X_test_imp, fname, out_path)

    # ── 9. Results JSON ──────────────────────────────────────────────
    perm_dict = dict(zip(features,
                         [float(x) for x in perm_result.importances_mean]))
    shap_dict = dict(zip(features,
                         [float(x) for x in mean_abs_shap]))

    results = {
        "mode": "all_features",
        "n_features": len(features),
        "features": features,
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
    with open(DATA_DIR / "allf_model_results.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  全量特征 7 模型对比 + 可解释性分析完成  ({elapsed:.1f}s)")
    print(f"  特征数         = {len(features)}")
    print(f"  最优模型       = {best_name}")
    print(f"  最优 Test AUC  = {best_auc_val:.4f}")
    print(f"{'='*60}")
    print(f"\n[完成] 训练矩阵        -> {DATA_DIR / 'allf_model_train.csv'}")
    print(f"[完成] 测试矩阵        -> {DATA_DIR / 'allf_model_test.csv'}")
    print(f"[完成] 模型对比（0.5） -> {DATA_DIR / 'allf_model_comparison.csv'}")
    print(f"[完成] 模型对比（最优）-> {DATA_DIR / 'allf_model_comparison_optimal.csv'}")
    print(f"[完成] 结果 JSON       -> {DATA_DIR / 'allf_model_results.json'}")
    print(f"[完成] ROC 曲线        -> {DATA_DIR / 'allf_roc_comparison.png'}")
    print(f"[完成] 最优模型重要性  -> {DATA_DIR / 'allf_best_feature_importance.png'}")
    print(f"[完成] Perm 重要性     -> {DATA_DIR / 'allf_best_permutation_importance.png'}")
    print(f"[完成] SHAP 汇总       -> {DATA_DIR / 'allf_best_shap_summary.png'}")
    print(f"[完成] 校准曲线        -> {DATA_DIR / 'allf_best_calibration.png'}")


if __name__ == "__main__":
    main()
