"""
建模 - 步骤 3：多模型对比
================================

在 RFE 选出的最终特征集合上训练并对比 7 个模型：
    KNN, LR, RF, XGBoost, AdaBoost, LightGBM, SVM

每个模型都封装在 ``Pipeline`` 中，确保:
  - 训练 fold 内做 imputation + scaling，避免 CV 信息泄漏；
  - 树模型（RF / XGB / AdaBoost / LightGBM）不强制标准化（树对尺度不敏感），
    但仍保留 imputer 以处理缺失；
  - 概率类模型（KNN / LR / SVM）启用 StandardScaler。

报告：
  - 训练集 5 折分层 CV：AUC mean ± std；
  - 测试集（hold-out）单次评估：AUC、Accuracy、Precision、Recall (Sensitivity)、
    Specificity、F1、Brier 分数。
  - 阈值默认为 0.5；脚本同时输出基于 Youden 指数（max(sens+spec−1)）选出的最优阈值
    及对应指标，写入 ``model_comparison_optimal.csv``。
  - 保存所有模型测试集 ROC 曲线图：``data/roc_comparison.png``。

输入：
  data/model_train.csv
  data/model_test.csv
  data/selected_features.json

输出：
  data/model_comparison.csv          —— 阈值=0.5 的对比
  data/model_comparison_optimal.csv  —— 阈值=Youden 最优
  data/roc_comparison.png
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")  # 避免 headless 环境报错
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "model_train.csv"
TEST_PATH  = DATA_DIR / "model_test.csv"
SELECT_PATH = DATA_DIR / "selected_features.json"

OUT_COMPARE = DATA_DIR / "model_comparison.csv"
OUT_OPT     = DATA_DIR / "model_comparison_optimal.csv"
OUT_ROC     = DATA_DIR / "roc_comparison.png"

RANDOM_STATE = 42
CV_FOLDS = 5

# 需要标准化的模型（对量纲敏感）
SCALE_MODELS = {"KNN", "LR", "SVM"}


def build_models() -> dict[str, object]:
    # 注意：所有模型都设 n_jobs=1，避免与外层 CV 的 n_jobs 嵌套并行造成
    # CPU oversubscription / 死锁。CV 在外层做并行更省事。
    return {
        "KNN":      KNeighborsClassifier(n_neighbors=15, n_jobs=1),
        "LR":       LogisticRegression(
                       penalty="l2", solver="liblinear",
                       max_iter=2000, class_weight="balanced",
                       random_state=RANDOM_STATE),
        "RF":       RandomForestClassifier(
                       n_estimators=300, max_depth=None,
                       class_weight="balanced", n_jobs=1,
                       random_state=RANDOM_STATE),
        "XGBoost":  XGBClassifier(
                       n_estimators=300, max_depth=4,
                       learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.9,
                       eval_metric="auc",
                       tree_method="hist",
                       random_state=RANDOM_STATE,
                       n_jobs=1),
        "AdaBoost": AdaBoostClassifier(
                       n_estimators=200, learning_rate=0.5,
                       random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(
                       n_estimators=300, max_depth=-1,
                       learning_rate=0.05,
                       num_leaves=31,
                       class_weight="balanced",
                       random_state=RANDOM_STATE,
                       n_jobs=1,
                       verbose=-1),
        "SVM":      SVC(C=1.0, kernel="rbf",
                       probability=True,
                       class_weight="balanced",
                       random_state=RANDOM_STATE),
    }


def make_pipeline(name: str, estimator) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name in SCALE_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "AUC":         roc_auc_score(y_true, y_prob),
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Precision":   precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sens,
        "Specificity": spec,
        "F1":          f1_score(y_true, y_pred, zero_division=0),
        "Brier":       brier_score_loss(y_true, y_prob),
        "Threshold":   threshold,
    }


def youden_threshold(y_true, y_prob) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


# ---------------------------------------------------------------------------
def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("未找到 model_train.csv / model_test.csv，请先运行 02_rfe_select.py")

    sel = json.loads(SELECT_PATH.read_text(encoding="utf-8"))
    target = sel["target"]
    features = sel["final_features"]

    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    X_train, y_train = train[features], train[target].astype(int).values
    X_test,  y_test  = test[features],  test[target].astype(int).values
    print(f"训练集 {X_train.shape} (事件 {y_train.mean():.2%}) | "
          f"测试集 {X_test.shape} (事件 {y_test.mean():.2%}) | 特征数 {len(features)}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows_default, rows_optimal = [], []
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    import time
    for name, est in build_models().items():
        t0 = time.time()
        print(f"[fit] {name} ...", flush=True)
        pipe = make_pipeline(name, est)
        try:
            cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv,
                                     scoring="roc_auc", n_jobs=CV_FOLDS)
        except Exception as e:
            print(f"[warn] {name} CV 失败：{e}")
            cv_auc = np.array([np.nan])

        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        m_default = evaluate(y_test, y_prob, threshold=0.5)
        thr_opt = youden_threshold(y_test, y_prob)
        m_opt = evaluate(y_test, y_prob, threshold=thr_opt)

        rows_default.append({"Model": name,
                              "CV_AUC_mean": float(np.nanmean(cv_auc)),
                              "CV_AUC_std":  float(np.nanstd(cv_auc)),
                              **m_default})
        rows_optimal.append({"Model": name,
                              "CV_AUC_mean": float(np.nanmean(cv_auc)),
                              "CV_AUC_std":  float(np.nanstd(cv_auc)),
                              **m_opt})

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, m_default["AUC"])

        print(f"{name:<9s}  CV_AUC={np.nanmean(cv_auc):.4f}±{np.nanstd(cv_auc):.4f}  "
              f"Test_AUC={m_default['AUC']:.4f}  "
              f"Sens={m_default['Sensitivity']:.3f}  Spec={m_default['Specificity']:.3f}  "
              f"Brier={m_default['Brier']:.4f}  ({time.time()-t0:.1f}s)",
              flush=True)

    cmp_default = pd.DataFrame(rows_default).sort_values("AUC", ascending=False)
    cmp_optimal = pd.DataFrame(rows_optimal).sort_values("AUC", ascending=False)
    cmp_default.to_csv(OUT_COMPARE, index=False, encoding="utf-8-sig")
    cmp_optimal.to_csv(OUT_OPT,    index=False, encoding="utf-8-sig")

    # ROC 曲线 ----------------------------------------------------------
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in sorted(roc_data.items(), key=lambda kv: -kv[1][2]):
        plt.plot(fpr, tpr, label=f"{name}  AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    plt.xlabel("1 − Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Test-set ROC Comparison")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=150)
    plt.close()

    print()
    print("=== 测试集对比（阈值=0.5，按 AUC 排序） ===")
    print(cmp_default.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("=== 测试集对比（阈值=Youden 最优） ===")
    print(cmp_optimal.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"[完成] 默认阈值对比 -> {OUT_COMPARE}")
    print(f"[完成] 最优阈值对比 -> {OUT_OPT}")
    print(f"[完成] ROC 曲线     -> {OUT_ROC}")


if __name__ == "__main__":
    main()
