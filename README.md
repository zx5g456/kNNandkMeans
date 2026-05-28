# kNN and kMeans AFCD Food Classification

本项目基于 AFCD Release 3 营养数据集，对食物营养成分数据进行探索、清洗、特征选择、标准化和训练/测试集划分，为后续使用 kNN 和 kMeans 进行食物分类或聚类分析做准备。

## 项目内容

- 读取 AFCD Release 3 的 Sheet 2 和 Sheet 3 数据。
- 统计数据集行数、列数和营养成分字段。
- 分析 Sheet 2 的 Classification 分布、特征范围和缺失值。
- 选择 20 个更适合作为用户输入的营养特征。
- 删除样本数较少的分类。
- 将以 g 为单位的主要营养成分转换为 mg。
- 对 20 个特征进行标准化。
- 按 70% / 30% 生成训练集和测试集。

## 目录结构

```text
.
├── data/
│   ├── AFCD Release 3 - Nutrient profiles.xlsx
│   ├── AFCD Release 3 - sheet1.csv
│   ├── AFCD Release 3 - sheet2.csv
│   ├── AFCD Release 3 - sheet3.csv
│   ├── sheet2_processed_20_features.csv
│   ├── sheet2_standardized_20_features.csv
│   ├── train/sheet2_train_70.csv
│   └── test/sheet2_test_30.csv
└── src/
    ├── datasetOverview.py
    ├── datasetExploration.py
    ├── datasetProcessing.py
    └── AFCD_kNN.py
```

## 环境要求

建议使用 Python 3.10 或以上版本。

需要安装的主要依赖：

```bash
pip install pandas matplotlib openpyxl
```

## 使用方法

在项目根目录运行以下命令。

### 1. 查看数据集概览

```bash
python src/datasetOverview.py
```

该脚本会统计 Sheet 2 和 Sheet 3 的营养成分字段，并生成：

- `data/sheet2_nutrient_components.txt`
- `data/sheet3_nutrient_components.txt`

### 2. 探索 Sheet 2 数据

```bash
python src/datasetExploration.py
```

该脚本会输出分类分布、特征统计、类别内特征范围和 20 个选中特征说明，主要生成：

- `data/sheet2_feature_summary.csv`
- `data/sheet2_class_distribution.csv`
- `data/sheet2_feature_ranges_by_class.csv`
- `data/sheet2_dataset_exploration_report.txt`
- `data/sheet2_selected_20_features.txt`

### 3. 数据清洗、标准化和切分

```bash
python src/datasetProcessing.py
```

该脚本会生成建模用数据：

- `data/sheet2_processed_20_features.csv`
- `data/sheet2_standardized_20_features.csv`
- `data/train/sheet2_train_70.csv`
- `data/test/sheet2_test_30.csv`

## 选中的 20 个特征

项目保留了常见、可解释、适合作为用户输入的营养成分，包括水分、蛋白质、脂肪、膳食纤维、糖、淀粉、钠、钾、钙、磷、镁、铁、锌、胆固醇和部分维生素等。具体列表定义在 `src/datasetProcessing.py` 和 `src/datasetExploration.py` 的 `SELECTED_20_FEATURES` 中。

## 说明

`src/AFCD_kNN.py` 当前保留为 kNN / kMeans 建模入口文件，可在完成数据预处理后继续实现分类、聚类、评估和可视化逻辑。
