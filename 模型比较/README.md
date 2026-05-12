# 多模型对比建模流水线

在已完成单位 / 编码统一（`数据预处理/01_format_unify.py`）的基础上，对
`data/all_variable_step1.csv` 进行：

1. **特征工程**（固定输入 + RFE 候选 + 衍生特征）
2. **数据划分 + RFECV 特征选择**
3. **多模型对比**：KNN / LR / RF / XGBoost / AdaBoost / LightGBM / SVM

## 运行

```bash
pip install -r requirements.txt

# 前置：先跑数据预处理步骤 1（产出 data/all_variable_step1.csv）
python 数据预处理/01_format_unify.py

# 步骤 1：特征工程（含 4 个衍生特征）
python 模型比较/01_feature_engineering.py
#   -> data/features_engineered.csv
#   -> data/feature_manifest.json

# 步骤 2：分层划分（默认 8:2，按 accession_id 分组）+ RFECV（LR, AUC, 5-fold）
python 模型比较/02_rfe_select.py
#   或：python 模型比较/02_rfe_select.py --use-consensus   # 复用步骤 4 的共识集
#   -> data/model_train.csv, data/model_test.csv
#   -> data/selected_features.json, data/rfe_ranking.csv

# 步骤 3：7 个模型对比，5 折 CV + 测试集评估
python 模型比较/03_model_comparison.py
#   -> data/model_comparison.csv           （阈值=0.5）
#   -> data/model_comparison_optimal.csv   （阈值=Youden 最优）
#   -> data/roc_comparison.png

# 步骤 4：RFE 深度分析（LR / RF / GBM 三基学习器 + AUC 曲线 + 共识特征集）
python 模型比较/04_rfe_analysis.py
#   -> data/rfe_analysis_per_learner.csv   每特征在每个基学习器下的 rank / 是否选中
#   -> data/rfe_consensus.json             多数表决得到的共识特征集
#   -> data/rfe_curve_{LR,RF,GBM}.png      各自的 AUC vs 保留特征数曲线
#   -> data/rfe_curves_combined.png        三条曲线叠加
```

## 关键设计

### 列名映射
原始数据列名与表格不完全一致，脚本自动做以下映射：

| 表格名 | 实际列 |
| --- | --- |
| `history_diabetes` | `history_diabetes_new` |
| `mean_hu` | `mean_hu（pcat_hu）` |
| `maximum_luminal_area_mm3` | `minimum_luminal_area_mm3` |
| `followup_triglycerides` | `diff_triglycerides`（表格备注：仅作为基线差值） |
| `followup_lipoprotein_a` | `diff_lipoprotein_a` |

数据中**完全没有**的 `cda_admission_number` / `stent_present` 直接跳过并提示。

### 衍生特征（写入 `data/feature_manifest.json` 中的 `derived_features`）

| 名称 | 来源 | 定义 |
| --- | --- | --- |
| `extreme_risk_ge2` | positive_remodeling + low_attenuation + napkin_ring + spotty_calcification 四个 flag 求和 ≥ 2 |  |
| `extreme_risk_ge3` |  | ≥ 3 |
| `extreme_risk_ge4` |  | = 4 |
| `ffrct_class3` | `ffrct_value` | ≥0.8→0（低危）, [0.7,0.8)→1（中）, <0.7→2（高） |
| `mean_hu_lt_minus70` | PCAT `mean_hu` | < −70 → 1 |

### 数据划分
默认 8:2，分层 + 按 `accession_id` 分组防止同一受试者跨集泄漏。

### RFECV
- 在 **RFE 候选** 集合上独立运行（固定特征始终保留）。
- 步骤 2 默认 `LogisticRegression(L2, class_weight='balanced')`，5 折分层 CV，
  step=1，最少保留 5 个特征。
- 步骤 4 同样的 RFECV 框架下换 **3 个基学习器** 各跑一遍（LR / RF / GBM），
  取多数表决得到 **共识特征集**，并绘制 AUC vs 保留特征数 曲线，输出
  `data/rfe_consensus.json`。可通过 `python 02_rfe_select.py --use-consensus`
  让步骤 3 用共识集而非单一基学习器的结果。

### 模型 Pipeline
对 KNN / LR / SVM 启用 `StandardScaler`；树模型仅做缺失值中位数插补。
所有模型都封装在 `Pipeline` 中，5 折 CV 时不会泄漏。

### 评估指标
- 训练集：5 折 CV AUC（mean ± std）
- 测试集：AUC、Accuracy、Precision、Sensitivity、Specificity、F1、Brier
- 阈值同时给出 **0.5** 和 **Youden 最优**（max(sens+spec−1)）两套结果
- 所有模型测试集 ROC 曲线绘制在同一张图

## 依赖

参见根目录 `requirements.txt`。建模新增依赖：`xgboost`, `lightgbm`, `matplotlib`。
