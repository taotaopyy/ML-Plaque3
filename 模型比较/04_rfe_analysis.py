"""
建模 - 步骤 4：RFE 深度分析（多基学习器 + AUC 曲线 + 共识特征集）
================================

逻辑（与 02_rfe_select.py 保持一致）：
  • 固定输入变量始终保留；
  • 仅在 RFE 候选池上做递归特征消除。

为了让 RFE 结论更稳健，本脚本使用 **3 个不同的基学习器** 各自跑一次 RFECV，
然后取交集 / 投票得到共识特征集：
  1. Logistic Regression（L2, class_weight=balanced）
  2. Random Forest（class_weight=balanced）
  3. Gradient Boosting（sklearn 实现，避开 xgboost/lightgbm 对 RFE 的限制）

对每个基学习器输出：
  • RFECV 选出的特征数 & 最佳 CV AUC；
  • 完整的 ranking（小=优先保留）；
  • AUC vs 保留特征数 曲线（PNG）。

最终：
  • ``data/rfe_analysis_per_learner.csv`` —— 每个特征在每个基学习器下的
    rank / 是否选中；
  • ``data/rfe_consensus.json`` —— 出现在 ≥ ``--vote`` 个基学习器中的特征
    （默认 2，多数表决）；
  • ``data/rfe_curve_*.png`` —— 各基学习器的 AUC-vs-保留特征数曲线；
  • ``data/rfe_curves_combined.png`` —— 三条曲线同图叠加。

读入：
  data/features_engineered.csv
  data/feature_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEATURES_PATH = DATA_DIR / "features_engineered.csv"
MANIFEST_PATH = DATA_DIR / "feature_manifest.json"

OUT_PER = DATA_DIR / "rfe_analysis_per_learner.csv"
OUT_CONSENSUS = DATA_DIR / "rfe_consensus.json"
OUT_CURVE_COMBINED = DATA_DIR / "rfe_curves_combined.png"

MULTICLASS_COLUMNS = {"plaque_type"}
RANDOM_STATE = 42
CV_FOLDS = 5
MIN_FEATURES = 5


# ---------------------------------------------------------------------------
def stratified_group_split(df, target, group, test_size, seed):
    if group and group in df.columns:
        g = df.groupby(group)[target].max().reset_index()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        (tr_idx, _), = sss.split(np.zeros(len(g)), g[target].astype(int))
        tr_ids = set(g.iloc[tr_idx][group])
        return df[df[group].isin(tr_ids)].copy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (tr_idx, _), = sss.split(np.zeros(len(df)), df[target].astype(int))
    return df.iloc[tr_idx].copy()


def onehot_inplace(train: pd.DataFrame, cols: list[str]) -> list[str]:
    new_cols: list[str] = []
    for c in cols:
        if c not in train.columns:
            continue
        cats = sorted(train[c].dropna().astype(str).unique())
        for cat in cats:
            name = f"{c}__{cat}"
            train[name] = (train[c].astype(str) == cat).astype(int)
            new_cols.append(name)
        train.drop(columns=[c], inplace=True)
    return new_cols


# ---------------------------------------------------------------------------
def make_base_learners() -> dict[str, tuple[object, int]]:
    """返回 (estimator, step)。Tree-based learner 用更大 step 提速。"""
    return {
        "LR":  (LogisticRegression(
                    penalty="l2", solver="liblinear", max_iter=2000,
                    class_weight="balanced", random_state=RANDOM_STATE), 1),
        "RF":  (RandomForestClassifier(
                    n_estimators=120, max_depth=None,
                    class_weight="balanced", n_jobs=1,
                    random_state=RANDOM_STATE), 2),
        "GBM": (GradientBoostingClassifier(
                    n_estimators=100, max_depth=3,
                    learning_rate=0.1, random_state=RANDOM_STATE), 2),
    }


def run_rfecv(X: pd.DataFrame, y: np.ndarray, estimator, cv, step: int) -> RFECV:
    rfecv = RFECV(
        estimator=estimator, step=step, cv=cv,
        scoring="roc_auc",
        min_features_to_select=MIN_FEATURES,
        n_jobs=CV_FOLDS,
    )
    rfecv.fit(X, y)
    return rfecv


def plot_curve(rfecv: RFECV, name: str, step: int, n_total: int, ax=None):
    scores = rfecv.cv_results_["mean_test_score"]
    stds   = rfecv.cv_results_["std_test_score"]
    # RFECV 计算时 n_features 由 min 开始按 step 增加（递归消除从 n_total
    # 开始消减），cv_results_ 长度 = ceil((n_total - min)/step) + 1
    n_vals = np.linspace(MIN_FEATURES, n_total, len(scores)).astype(int)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n_vals, scores, marker="o", label=name)
    ax.fill_between(n_vals, scores - stds, scores + stds, alpha=0.15)
    ax.axvline(rfecv.n_features_, linestyle="--", lw=0.8, alpha=0.6)
    return ax


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="划分出来的测试集比例（仅为了确保 RFE 在训练集上做）")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--vote", type=int, default=2,
                        help="共识特征至少需要被多少个基学习器选中（默认 2/3）")
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"未找到 {FEATURES_PATH}，请先运行 模型比较/01_feature_engineering.py")

    df = pd.read_csv(FEATURES_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    target = manifest["target"]
    id_col = manifest.get("id_column")
    fixed = [c for c in manifest["fixed_features"] if c in df.columns]
    rfe_pool = [c for c in manifest["rfe_candidates"] if c in df.columns]

    df = df.dropna(subset=[target]).reset_index(drop=True)
    df[target] = df[target].astype(int)
    print(f"加载特征表：{df.shape}")
    print(f"  固定特征 (始终保留)：{len(fixed)} 个")
    print(f"  RFE 候选 (待筛选)  ：{len(rfe_pool)} 个")

    # 仅取训练集做 RFE，避免泄漏
    train = stratified_group_split(df, target, id_col, args.test_size, args.seed)
    print(f"训练集 {train.shape} (事件 {train[target].mean():.2%})")

    multi_cols = [c for c in rfe_pool if c in MULTICLASS_COLUMNS]
    new_cols = onehot_inplace(train, multi_cols)
    rfe_pool = [c for c in rfe_pool if c not in multi_cols] + new_cols
    if new_cols:
        print(f"[onehot] {multi_cols} -> {new_cols}")

    # 数值化 + 中位数插补 + 标准化（仅供 RFE 基学习器使用）
    X = train[rfe_pool].apply(pd.to_numeric, errors="coerce")
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                     columns=rfe_pool, index=train.index)
    X = pd.DataFrame(StandardScaler().fit_transform(X),
                     columns=rfe_pool, index=train.index)
    y = train[target].astype(int).values

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=args.seed)

    # 跑三个基学习器 ---------------------------------------------------
    ranking_table = pd.DataFrame({"feature": rfe_pool})
    fig, ax = plt.subplots(figsize=(8, 5))
    summary = {}

    import time
    n_total = len(rfe_pool)
    for name, (est, step) in make_base_learners().items():
        t0 = time.time()
        print(f"\n[RFE] base = {name}  step={step}  正在拟合 RFECV "
              f"({n_total} -> >={MIN_FEATURES}) ...", flush=True)
        rfecv = run_rfecv(X, y, est, cv, step)
        scores = rfecv.cv_results_["mean_test_score"]
        best_k = int(rfecv.n_features_)
        best_auc = float(scores.max())
        selected = [f for f, keep in zip(rfe_pool, rfecv.support_) if keep]

        print(f"      [{time.time()-t0:.1f}s] 最优保留特征数 = {best_k}  "
              f"最佳 CV AUC = {best_auc:.4f}")
        print(f"      选出 {len(selected)} 个特征：{selected}")

        ranking_table[f"rank_{name}"]     = rfecv.ranking_
        ranking_table[f"selected_{name}"] = rfecv.support_.astype(int)
        plot_curve(rfecv, name, step, n_total, ax)

        # 各自单独保存曲线
        single_path = DATA_DIR / f"rfe_curve_{name}.png"
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        plot_curve(rfecv, name, step, n_total, ax2)
        ax2.set_xlabel("Number of retained features")
        ax2.set_ylabel("CV ROC-AUC")
        ax2.set_title(f"RFECV — base learner: {name}")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(single_path, dpi=150)
        plt.close(fig2)

        summary[name] = {
            "best_n_features": best_k,
            "best_cv_auc":     best_auc,
            "selected":        selected,
        }

    ax.set_xlabel("Number of retained features")
    ax.set_ylabel("CV ROC-AUC")
    ax.set_title("RFECV — AUC vs. retained features (per base learner)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_CURVE_COMBINED, dpi=150)
    plt.close(fig)

    # 共识特征集 -------------------------------------------------------
    vote_cols = [c for c in ranking_table.columns if c.startswith("selected_")]
    ranking_table["votes"] = ranking_table[vote_cols].sum(axis=1)
    ranking_table = ranking_table.sort_values(["votes", "feature"], ascending=[False, True])
    ranking_table.to_csv(OUT_PER, index=False, encoding="utf-8-sig")

    consensus = ranking_table.loc[
        ranking_table["votes"] >= args.vote, "feature"
    ].tolist()
    print(f"\n[共识] 至少被 {args.vote}/{len(make_base_learners())} 个基学习器选中的特征 "
          f"({len(consensus)} 个)：")
    for f in consensus:
        votes = int(ranking_table.set_index("feature").loc[f, "votes"])
        print(f"    {f:<40s}  votes={votes}")

    final_features = fixed + consensus
    out_summary = {
        "target": target,
        "id_column": id_col,
        "fixed_features": fixed,
        "rfe_pool": rfe_pool,
        "per_learner": summary,
        "consensus_features": consensus,
        "vote_threshold": args.vote,
        "final_features (fixed + consensus)": final_features,
    }
    with open(OUT_CONSENSUS, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 每特征逐学习器排名 -> {OUT_PER}")
    print(f"[完成] 共识特征集 JSON     -> {OUT_CONSENSUS}")
    print(f"[完成] AUC 曲线（合并）    -> {OUT_CURVE_COMBINED}")
    for name in make_base_learners().keys():
        print(f"[完成] AUC 曲线 ({name})       -> {DATA_DIR / f'rfe_curve_{name}.png'}")


if __name__ == "__main__":
    main()
