"""
建模 - 步骤 2：分层划分 + RFE 特征筛选
================================

流程：
1. 读取 ``data/features_engineered.csv`` 与 ``data/feature_manifest.json``。
2. 按 ``target_lesion_positive`` 分层抽样（默认 8:2），若存在
   ``accession_id`` 则按受试者分组，防止跨集泄漏。
3. **仅在训练集上**：
   - 多分类列（如 plaque_type）做 one-hot；
   - 缺失值用中位数 / 众数插补；
   - 数值列用 StandardScaler 标准化（仅用于 RFE 的基学习器，
     便于 LR 系数尺度可比，最终训练时由各模型 pipeline 自行处理）。
4. 在 **RFE 候选** 集合上跑 ``RFECV``：基学习器 Logistic Regression（L2），
   5 折分层 CV，评分 = ROC-AUC，自动决定保留特征数。
5. 最终特征 = 固定特征 ∪ RFE 选出的特征；保存训练 / 测试矩阵。

输入：
  data/features_engineered.csv
  data/feature_manifest.json
输出：
  data/model_train.csv      （只含最终特征 + 结局 + ID）
  data/model_test.csv
  data/selected_features.json
  data/rfe_ranking.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEATURES_PATH = DATA_DIR / "features_engineered.csv"
MANIFEST_PATH = DATA_DIR / "feature_manifest.json"

OUT_TRAIN = DATA_DIR / "model_train.csv"
OUT_TEST  = DATA_DIR / "model_test.csv"
OUT_SELECT = DATA_DIR / "selected_features.json"
OUT_RANK   = DATA_DIR / "rfe_ranking.csv"

MULTICLASS_COLUMNS = {"plaque_type"}   # 需要 one-hot 的多分类候选


# ---------------------------------------------------------------------------
def stratified_group_split(df, target, group, test_size, seed):
    if group and group in df.columns:
        g = df.groupby(group)[target].max().reset_index()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        (tr_idx, te_idx), = sss.split(np.zeros(len(g)), g[target].astype(int))
        tr_ids = set(g.iloc[tr_idx][group])
        te_ids = set(g.iloc[te_idx][group])
        return df[df[group].isin(tr_ids)].copy(), df[df[group].isin(te_ids)].copy()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (tr_idx, te_idx), = sss.split(np.zeros(len(df)), df[target].astype(int))
    return df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()


def onehot_fit_apply(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """对训练集 fit one-hot，应用到训练 / 测试集，返回新增列名。"""
    new_cols: list[str] = []
    cols = [c for c in cols if c in train.columns]
    for c in cols:
        cats = sorted(train[c].dropna().astype(str).unique())
        for cat in cats:
            new_name = f"{c}__{cat}"
            train[new_name] = (train[c].astype(str) == cat).astype(int)
            test[new_name]  = (test[c].astype(str)  == cat).astype(int) if c in test.columns else 0
            new_cols.append(new_name)
        train.drop(columns=[c], inplace=True)
        if c in test.columns:
            test.drop(columns=[c], inplace=True)
    return train, test, new_cols


def median_impute(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    imp = SimpleImputer(strategy="median")
    train[cols] = imp.fit_transform(train[cols])
    test[cols]  = imp.transform(test[cols])
    return train, test, imp


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-features", type=int, default=5,
                        help="RFECV 至少保留的特征数")
    parser.add_argument("--use-consensus", action="store_true",
                        help="改用 04_rfe_analysis.py 输出的共识特征集（"
                             "data/rfe_consensus.json），跳过本脚本 RFECV")
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"未找到 {FEATURES_PATH}，请先运行 01_feature_engineering.py")
    df = pd.read_csv(FEATURES_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    target = manifest["target"]
    id_col = manifest.get("id_column")
    fixed = [c for c in manifest["fixed_features"] if c in df.columns]
    rfe_pool = [c for c in manifest["rfe_candidates"] if c in df.columns]

    df = df.dropna(subset=[target]).reset_index(drop=True)
    df[target] = df[target].astype(int)
    print(f"加载特征表：{df.shape}  | 固定特征 {len(fixed)} | RFE 候选 {len(rfe_pool)}")

    # 1. 划分 ----------------------------------------------------------
    train, test = stratified_group_split(df, target, id_col, args.test_size, args.seed)
    print(f"训练集 {train.shape} (事件 {train[target].mean():.2%}) | "
          f"测试集 {test.shape} (事件 {test[target].mean():.2%})")

    # 2. 多分类 one-hot（仅在 RFE 候选中可能出现）-----------------------
    multi_cols = [c for c in rfe_pool if c in MULTICLASS_COLUMNS]
    train, test, onehot_cols = onehot_fit_apply(train, test, multi_cols)
    rfe_pool = [c for c in rfe_pool if c not in multi_cols] + onehot_cols
    if onehot_cols:
        print(f"[onehot] {multi_cols} -> {onehot_cols}")

    # 3. 数值列插补 + 标准化（用于 RFE 基学习器）------------------------
    all_feat = fixed + rfe_pool
    all_feat = [c for c in dict.fromkeys(all_feat) if c in train.columns]
    train, test, _ = median_impute(train, test, all_feat)

    X_rfe_train = train[rfe_pool].copy()
    scaler = StandardScaler()
    X_rfe_train_std = pd.DataFrame(scaler.fit_transform(X_rfe_train),
                                   columns=rfe_pool, index=train.index)

    y_train = train[target].astype(int).values

    # 4. 选特征：默认 RFECV(LR)，也可读取 04_rfe_analysis.py 的共识集合 -
    consensus_path = DATA_DIR / "rfe_consensus.json"
    if args.use_consensus and consensus_path.exists():
        cons = json.loads(consensus_path.read_text(encoding="utf-8"))
        selected_rfe = [c for c in cons["consensus_features"] if c in rfe_pool]
        ranking = pd.DataFrame({
            "feature": rfe_pool,
            "rank":    [1 if c in selected_rfe else 2 for c in rfe_pool],
            "selected": [c in selected_rfe for c in rfe_pool],
        }).sort_values(["rank", "feature"])
        ranking.to_csv(OUT_RANK, index=False, encoding="utf-8-sig")
        print(f"\n[RFE] 复用共识特征集 ({len(selected_rfe)} 个) "
              f"来自 {consensus_path}")
        cv_best_auc = float(max(v["best_cv_auc"] for v in cons["per_learner"].values()))
    else:
        estimator = LogisticRegression(
            penalty="l2", solver="liblinear", max_iter=2000, class_weight="balanced"
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring="roc_auc",
            min_features_to_select=max(1, args.min_features),
            n_jobs=5,
        )
        print(f"\n[RFE] 在 {len(rfe_pool)} 个 RFE 候选上跑 RFECV（LR, 5-fold, AUC）...")
        rfecv.fit(X_rfe_train_std, y_train)

        ranking = pd.DataFrame({
            "feature": rfe_pool,
            "rank":    rfecv.ranking_,
            "selected": rfecv.support_,
        }).sort_values(["rank", "feature"])
        ranking.to_csv(OUT_RANK, index=False, encoding="utf-8-sig")

        selected_rfe = [c for c, keep in zip(rfe_pool, rfecv.support_) if keep]
        cv_best_auc = float(rfecv.cv_results_["mean_test_score"].max())
        print(f"[RFE] 选出特征 {len(selected_rfe)} / {len(rfe_pool)}")
        print(f"      {selected_rfe}")
        print(f"[RFE] 最佳 CV AUC = {cv_best_auc:.4f}（@ k={rfecv.n_features_}）")

    final_features = fixed + selected_rfe

    # 5. 输出最终训练 / 测试集 ----------------------------------------
    keep_cols = ([id_col] if id_col and id_col in train.columns else []) + [target] + final_features
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in train.columns]
    train[keep_cols].to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
    test [keep_cols].to_csv(OUT_TEST,  index=False, encoding="utf-8-sig")

    summary = {
        "target": target,
        "id_column": id_col,
        "fixed_features": fixed,
        "rfe_pool": rfe_pool,
        "rfe_selected": selected_rfe,
        "final_features": final_features,
        "rfe_best_n_features": int(len(selected_rfe)),
        "rfe_best_cv_auc": cv_best_auc,
        "source": "consensus" if (args.use_consensus and consensus_path.exists()) else "rfecv_lr",
    }
    with open(OUT_SELECT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 训练矩阵 -> {OUT_TRAIN}  shape={train[keep_cols].shape}")
    print(f"[完成] 测试矩阵 -> {OUT_TEST}   shape={test[keep_cols].shape}")
    print(f"[完成] 选出特征 -> {OUT_SELECT}")
    print(f"[完成] RFE 排序 -> {OUT_RANK}")


if __name__ == "__main__":
    main()
