# kNN and kMeans AFCD Food Classification

This project uses the AFCD Release 3 nutrient dataset to explore, clean, select features, standardize, and split food nutrition data for later kNN classification and kMeans clustering work.

## Project Features

- Read AFCD Release 3 Sheet 2 and Sheet 3 data.
- Summarize dataset rows, columns, and nutrient component fields.
- Analyze Sheet 2 classification distribution, feature ranges, and missing values.
- Select 20 practical nutrient features for user input.
- Remove classifications with very small sample counts.
- Convert major gram-based nutrient features to milligrams.
- Standardize the selected 20 features.
- Create a 70% training set and 30% testing set.

## Project Structure

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

## Requirements

Python 3.10 or later is recommended.

Install the main dependencies:

```bash
pip install pandas matplotlib openpyxl
```

## Usage

Run the following commands from the project root directory.

### 1. Generate Dataset Overview

```bash
python src/datasetOverview.py
```

This script summarizes the nutrient component fields in Sheet 2 and Sheet 3, and generates:

- `data/sheet2_nutrient_components.txt`
- `data/sheet3_nutrient_components.txt`

### 2. Explore Sheet 2 Data

```bash
python src/datasetExploration.py
```

This script exports classification distributions, feature statistics, feature ranges by class, and details about the 20 selected features. Main outputs include:

- `data/sheet2_feature_summary.csv`
- `data/sheet2_class_distribution.csv`
- `data/sheet2_feature_ranges_by_class.csv`
- `data/sheet2_dataset_exploration_report.txt`
- `data/sheet2_selected_20_features.txt`

### 3. Clean, Standardize, and Split the Data

```bash
python src/datasetProcessing.py
```

This script generates the modeling-ready datasets:

- `data/sheet2_processed_20_features.csv`
- `data/sheet2_standardized_20_features.csv`
- `data/train/sheet2_train_70.csv`
- `data/test/sheet2_test_30.csv`

## Selected 20 Features

The project keeps common, interpretable nutrients that are suitable as required user inputs. These include moisture, protein, fat, dietary fibre, sugars, starch, sodium, potassium, calcium, phosphorus, magnesium, iron, zinc, cholesterol, and several vitamins.

The full feature list is defined as `SELECTED_20_FEATURES` in:

- `src/datasetProcessing.py`
- `src/datasetExploration.py`

## Notes

`src/AFCD_kNN.py` is currently reserved as the main entry point for kNN and kMeans modeling. It can be extended with classification, clustering, evaluation, and visualization logic after preprocessing is complete.
