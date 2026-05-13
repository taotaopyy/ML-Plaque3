# 数据预处理流水线

针对 `all_variable.csv` 的三步预处理脚本，需按顺序执行：

```bash
# 步骤 1：单位 / 日期 / 基础编码 统一化
python 数据预处理/01_format_unify.py
#   -> data/all_variable_step1.csv

# 步骤 2：分层随机抽样（默认 8:2，按 accession_id 分组防泄漏）
python 数据预处理/02_split_dataset.py            # 8:2
python 数据预处理/02_split_dataset.py --test-size 0.3   # 7:3
#   -> data/train.csv, data/test.csv

# 步骤 3：基于训练集的局部预处理（异常值 / 缺失 / 偏态 / one-hot）
python 数据预处理/03_train_local_preprocess.py
#   -> data/train_processed.csv, data/test_processed.csv
#   -> data/preprocess_params.json

# 步骤 4：离散变量与连续变量检查（描述统计 + 正态性 + 与结局的单变量检验）
python 数据预处理/04_variable_check.py
#   -> data/var_check_continuous.csv
#   -> data/var_check_discrete.csv
#   -> data/var_check_summary.txt

# 步骤 5：共线性检查（Pearson/Spearman 相关、VIF、删除候选）
python 数据预处理/05_collinearity_check.py
#   -> data/collinearity_pearson.csv
#   -> data/collinearity_spearman.csv
#   -> data/collinearity_high_pairs.csv
#   -> data/collinearity_vif.csv
#   -> data/collinearity_drop_candidates.csv
```

## 依赖

```
pandas
numpy
scikit-learn
scipy
statsmodels
```

## 设计要点

* **训练集驱动**：所有需要"学习"的参数（IQR 上下限、缺失插补值、log
  转换列表、one-hot 类别集合）只在训练集上拟合，再原样应用到测试集，
  避免测试集信息泄漏。
* **缺失率剔除**：阈值 30%，可在 `03_train_local_preprocess.py`
  中调整 `MISSING_THRESHOLD`。
* **偏态判定**：`|skew| > 1` 且全部非负的连续变量做 `log1p`。
* **分组抽样**：同一 `accession_id` 在原表中出现多次（不同病变 /
  血管），默认按受试者分组后再分层抽样，可加 `--no-group` 关闭。
* **参数审计**：步骤 3 的所有学习到的参数写入
  `data/preprocess_params.json`，便于事后复盘与上线复现。
* **变量类型识别（步骤 4）**：根据 dtype 与唯一值数自动区分连续 / 二元 /
  多分类，避免对 one-hot 后的哑变量当作连续变量做正态性检验。
* **正态性 → 检验选择（步骤 4）**：通过 Shapiro（n≤5000）或 KS（n>5000）
  判断分布，再决定连续变量与结局比较时使用 t 检验或 Mann-Whitney U；
  离散变量使用卡方，期望频数<5 的 2×2 表退化为 Fisher 精确检验。
* **共线性判定（步骤 5）**：Spearman `|r| ≥ 0.8` 视为高共线，VIF > 10
  视为严重共线。脚本会输出"删除候选清单"，但**不会**自动删除任何
  特征——最终保留 / 删除应结合临床意义复核。
