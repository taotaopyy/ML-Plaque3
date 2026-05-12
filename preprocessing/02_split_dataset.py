"""
步骤 2：数据集划分
================================

采用分层随机抽样（Stratified Random Split），按结局事件
``target_lesion_positive`` 的比例划分为训练集 (D_train) 与测试集 (D_test)，
默认 8:2，可通过命令行参数调整为 7:3。

由于同一受试者（accession_id）可能在数据集中出现多次（多支血管 / 多个病
变），为防止数据泄漏，本脚本默认开启 ``--group-by-patient`` 选项，按 ID
分组后再做分层抽样；若关闭，则退化为常规分层抽样。

输入：data/all_variable_step1.csv
输出：
    data/train.csv  —— 训练集
    data/test.csv   —— 测试集
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "all_variable_step1.csv"
OUTPUT_DIR = ROOT / "data"
TARGET_COL = "target_lesion_positive"
GROUP_COL = "accession_id"


def stratified_split(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """常规分层抽样。"""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    y = df[target].astype(int)
    (train_idx, test_idx), = sss.split(np.zeros(len(df)), y)
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def group_stratified_split(
    df: pd.DataFrame,
    target: str,
    group: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按受试者分组的分层抽样。

    思路：先把每个受试者归并为"是否发生过结局事件"的二值标签，
    然后在受试者级别做分层抽样，最后再把各自的全部行展开到 train/test。
    """
    # 每个 ID 的事件标签：组内任一行=1 即视为事件组
    grouped = df.groupby(group)[target].max().reset_index()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    y = grouped[target].astype(int)
    (train_idx, test_idx), = sss.split(np.zeros(len(grouped)), y)
    train_ids = set(grouped.iloc[train_idx][group])
    test_ids  = set(grouped.iloc[test_idx][group])
    train_df = df[df[group].isin(train_ids)].copy()
    test_df  = df[df[group].isin(test_ids)].copy()
    return train_df, test_df


def report(name: str, df: pd.DataFrame, target: str) -> None:
    pos = int(df[target].sum())
    n = len(df)
    print(f"{name}: n={n}  事件={pos} ({pos/n:.2%})  阴性={n-pos}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="测试集比例，常用 0.2 (8:2) 或 0.3 (7:3)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-group", action="store_true",
                        help="关闭按 accession_id 分组（默认会按受试者分组防泄漏）")
    args = parser.parse_args()

    df = pd.read_csv(INPUT_PATH)
    if TARGET_COL not in df.columns:
        raise KeyError(f"找不到结局列 {TARGET_COL}")

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    if args.no_group or GROUP_COL not in df.columns:
        print("[split] 按行做分层随机抽样")
        train_df, test_df = stratified_split(df, TARGET_COL, args.test_size, args.random_state)
    else:
        print(f"[split] 按 {GROUP_COL} 分组做分层随机抽样（防止同一受试者跨集泄漏）")
        train_df, test_df = group_stratified_split(
            df, TARGET_COL, GROUP_COL, args.test_size, args.random_state
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.csv"
    test_path  = OUTPUT_DIR / "test.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"\n划分比例 train:test = {1-args.test_size:.0%}:{args.test_size:.0%}")
    report("全样本", df, TARGET_COL)
    report("训练集", train_df, TARGET_COL)
    report("测试集", test_df, TARGET_COL)
    print(f"\n[完成] {train_path}")
    print(f"[完成] {test_path}")


if __name__ == "__main__":
    main()
