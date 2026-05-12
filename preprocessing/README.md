# 数据预处理流水线

针对 `all_variable.csv` 的三步预处理脚本，需按顺序执行：

```bash
# 步骤 1：单位 / 日期 / 基础编码 统一化
python preprocessing/01_format_unify.py
#   -> data/all_variable_step1.csv

# 步骤 2：分层随机抽样（默认 8:2，按 accession_id 分组防泄漏）
python preprocessing/02_split_dataset.py            # 8:2
python preprocessing/02_split_dataset.py --test-size 0.3   # 7:3
#   -> data/train.csv, data/test.csv

# 步骤 3：基于训练集的局部预处理（异常值 / 缺失 / 偏态 / one-hot）
python preprocessing/03_train_local_preprocess.py
#   -> data/train_processed.csv, data/test_processed.csv
#   -> data/preprocess_params.json
```

## 依赖

```
pandas
numpy
scikit-learn
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
