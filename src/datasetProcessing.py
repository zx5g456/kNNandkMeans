from pathlib import Path

import pandas as pd


# These 20 features are selected as practical required user inputs.
# Selection logic:
# - keep common, interpretable nutrients that a user can understand and enter;
# - keep Moisture (water) because sheet2 contains both solids and liquids, so water
#   content is useful for separating food forms;
# - prefer features with low missing values in sheet2;
# - cover macronutrients, minerals, vitamins, and cholesterol instead of taking
#   many narrow components from only one nutrient family.
SELECTED_20_FEATURES = [
    "Moisture (water) (g)",
    "Protein (g)",
    "Fat, total (g)",
    "Ash (g)",
    "Total dietary fibre (g)",
    "Available carbohydrate, without sugar alcohols (g)",
    "Total sugars (g)",
    "Starch (g)",
    "Sodium (Na) (mg)",
    "Potassium (K) (mg)",
    "Calcium (Ca) (mg)",
    "Phosphorus (P) (mg)",
    "Magnesium (Mg) (mg)",
    "Iron (Fe) (mg)",
    "Zinc (Zn) (mg)",
    "Cholesterol (mg)",
    "Thiamin (B1) (mg)",
    "Riboflavin (B2) (mg)",
    "Niacin (B3) (mg)",
    "Vitamin C (mg)",
]

GRAM_TO_MILLIGRAM_FEATURES = [
    "Moisture (water) (g)",
    "Protein (g)",
    "Fat, total (g)",
    "Ash (g)",
    "Total dietary fibre (g)",
    "Available carbohydrate, without sugar alcohols (g)",
    "Total sugars (g)",
    "Starch (g)",
]

SAMPLE_COUNT_BELOW_6 = [
    13203, 13105, 12302, 12506, 13301, 25201, 12403, 13508, 20603, 21201,
    18601, 18606, 15502, 28301, 26101, 28101, 27101, 31501, 11201, 11501,
    11401, 12511, 12601, 13305, 19403, 19106, 19204, 20502, 20106, 14502,
    16403, 16501, 16302, 16103, 16401, 18906, 33101, 22201, 27201, 24703,
    31301, 29101, 29203, 11904, 11503, 11601, 11101, 11802, 12203, 12202,
    12301, 13304, 13603, 13601, 13306, 22102, 12104, 13509, 19101, 19105,
    20105, 20104, 20102, 14101, 14303, 16801, 16804, 16304, 16301, 16402,
    16102, 16104, 16702, 18401, 15102, 27301, 24801, 31303, 31401, 29102,
    29301, 11903, 30105, 11205, 11603, 11604, 11602, 11703, 13104, 12204,
    12205, 12306, 12304, 12510, 12504, 12505, 12508, 13303, 13506, 13403,
    23301, 23106, 18701, 23305, 22202, 31102, 19405, 19501, 27303, 19104,
    19102, 19103, 19601, 19201, 20103, 20601, 14201, 14304, 14301, 25102,
    18706, 18602, 28302, 28303, 26301, 26201, 23201, 23202, 24404, 24503,
    24601, 31304, 31502, 31101, 29401, 29505, 11902, 11901, 30104, 11209,
    11208, 11202, 11504, 11502, 11302, 11103, 11605, 11801, 13204, 13202,
    13205, 13103, 13102, 12303, 13604, 11905, 12501, 12502, 12509, 12512,
    12503, 13307, 13511, 13504, 13406, 13402, 23303, 23302, 23107, 23101,
    15505, 23103, 19402, 19406, 19407, 20301, 19301, 19303, 19304, 19305,
    19503, 19802, 32103, 19203, 19208, 19205, 19701, 20101, 21301, 14102,
    14501, 17201, 23503, 18905, 18502, 18605, 15302, 26401, 26202, 26102,
    27204, 24001, 24803
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SHEET2_CSV = DATA_DIR / "AFCD Release 3 - sheet2.csv"
PROCESSED_SHEET2_CSV = DATA_DIR / "sheet2_processed_20_features.csv"
STANDARDIZED_SHEET2_CSV = DATA_DIR / "sheet2_standardized_20_features.csv"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_CSV = TRAIN_DIR / "sheet2_train_70.csv"
TEST_CSV = TEST_DIR / "sheet2_test_30.csv"


def normalise_column_name(column: object) -> str:
    """Return a readable single-line column name."""
    return " ".join(str(column).split())


def read_sheet2(csv_file: Path = SHEET2_CSV) -> pd.DataFrame:
    """Read AFCD Release 3 sheet2 with the real header row."""
    if not csv_file.exists():
        raise FileNotFoundError(f"Could not find sheet2 CSV file: {csv_file}")

    df = pd.read_csv(csv_file, header=2).dropna(how="all")
    df.columns = [normalise_column_name(column) for column in df.columns]
    return df


def remove_below_6_classes(
    df: pd.DataFrame,
    class_column: str = "Classification",
) -> pd.DataFrame:
    """Remove rows whose Classification is in SAMPLE_COUNT_BELOW_6."""
    if class_column not in df.columns:
        return df

    classes_to_remove = {str(classification) for classification in SAMPLE_COUNT_BELOW_6}
    keep_rows = ~df[class_column].astype(str).isin(classes_to_remove)
    return df.loc[keep_rows].copy()


def clean_sheet2_selected_features(
    input_csv: Path = SHEET2_CSV,
    output_csv: Path = PROCESSED_SHEET2_CSV,
    keep_metadata_columns: tuple[str, ...] = ("Classification", "Food Name"),
) -> pd.DataFrame:
    """Clean sheet2 and export only useful metadata plus the selected 20 features."""
    df = read_sheet2(input_csv)
    df = remove_below_6_classes(df)

    missing_features = [
        feature for feature in SELECTED_20_FEATURES if feature not in df.columns
    ]
    if missing_features:
        raise ValueError(
            "The following selected features are missing from the dataset: "
            + ", ".join(missing_features)
        )

    metadata_columns = [
        column for column in keep_metadata_columns if column in df.columns
    ]
    processed_df = df[[*metadata_columns, *SELECTED_20_FEATURES]].copy()

    processed_df[SELECTED_20_FEATURES] = (
        processed_df[SELECTED_20_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    processed_df[GRAM_TO_MILLIGRAM_FEATURES] = (
        processed_df[GRAM_TO_MILLIGRAM_FEATURES] * 1000
    )
    processed_df = processed_df.rename(
        columns={
            feature: feature.replace("(g)", "(mg)")
            for feature in GRAM_TO_MILLIGRAM_FEATURES
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_csv, index=False)
    return processed_df


def remove_unit_from_feature_name(column: str) -> str:
    """Remove the trailing unit from a feature column name."""
    return column.rsplit(" (", 1)[0] if column.endswith(")") else column


def standardize_processed_features(
    input_csv: Path = PROCESSED_SHEET2_CSV,
    output_csv: Path = STANDARDIZED_SHEET2_CSV,
    metadata_columns: tuple[str, ...] = ("Classification", "Food Name"),
) -> pd.DataFrame:
    """Standardize nutrient features and export a unit-free feature CSV."""
    df = pd.read_csv(input_csv)
    df = remove_below_6_classes(df)
    existing_metadata_columns = [
        column for column in metadata_columns if column in df.columns
    ]
    feature_columns = [
        column for column in df.columns if column not in existing_metadata_columns
    ]

    standardized_df = df[existing_metadata_columns].copy()
    numeric_features = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    feature_std = numeric_features.std()
    feature_std = feature_std.where(feature_std != 0, 1)
    standardized_features = (numeric_features - numeric_features.mean()) / feature_std
    standardized_features = standardized_features.rename(
        columns={
            column: remove_unit_from_feature_name(column)
            for column in feature_columns
        }
    )

    standardized_df = pd.concat(
        [standardized_df, standardized_features],
        axis=1,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    standardized_df.to_csv(output_csv, index=False)
    return standardized_df


def split_standardized_train_test(
    input_csv: Path = STANDARDIZED_SHEET2_CSV,
    train_csv: Path = TRAIN_CSV,
    test_csv: Path = TEST_CSV,
    train_fraction: float = 0.7,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly split the standardized sheet2 data into train and test CSV files."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    df = pd.read_csv(input_csv)
    df = remove_below_6_classes(df)
    train_df = df.sample(frac=train_fraction, random_state=random_state)
    test_df = df.drop(train_df.index)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_csv.parent.mkdir(parents=True, exist_ok=True)
    test_csv.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    return train_df, test_df


def main() -> None:
    """Create processed, standardized, train, and test sheet2 CSV files."""
    processed_df = clean_sheet2_selected_features()
    print(
        f"Wrote {PROCESSED_SHEET2_CSV} with "
        f"{len(processed_df)} rows and {len(processed_df.columns)} columns."
    )
    standardized_df = standardize_processed_features()
    print(
        f"Wrote {STANDARDIZED_SHEET2_CSV} with "
        f"{len(standardized_df)} rows and {len(standardized_df.columns)} columns."
    )
    train_df, test_df = split_standardized_train_test()
    print(
        f"Wrote {TRAIN_CSV} with {len(train_df)} rows and "
        f"{TEST_CSV} with {len(test_df)} rows."
    )


if __name__ == "__main__":
    main()
