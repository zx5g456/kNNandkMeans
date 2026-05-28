import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SHEET2_CSV = DATA_DIR / "AFCD Release 3 - sheet2.csv"
SHEET2_CLASS_DISTRIBUTION_CSV = DATA_DIR / "sheet2_class_distribution.csv"

NON_FEATURE_COLUMNS = {
    "Public Food Key",
    "Classification",
    "Derivation",
    "Food Name",
    "Energy with dietary fibre, equated (kJ)",
    "Energy, without dietary fibre, equated (kJ)",
}

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

# These feature groups are not used together in the required input list because
# they contain overlapping or derived information. One representative feature is
# kept from each group to reduce user input burden and avoid redundant signals.
REMOVED_CORRELATED_FEATURE_GROUPS = [
    {
        "kept": "Protein (g)",
        "removed": ["Nitrogen (g)"],
        "reason": "Nitrogen is closely related to protein and is less intuitive for users.",
    },
    {
        "kept": "Available carbohydrate, without sugar alcohols (g)",
        "removed": ["Available carbohydrate, with sugar alcohols (g)"],
        "reason": "Both carbohydrate measures are very similar; one representative is enough.",
    },
    {
        "kept": "Total sugars (g)",
        "removed": [
            "Added sugars (g)",
            "Free sugars (g)",
            "Fructose (g)",
            "Glucose (g)",
            "Sucrose (g)",
            "Lactose (g)",
        ],
        "reason": "Specific sugar components overlap with total sugars and increase input complexity.",
    },
    {
        "kept": "Fat, total (g)",
        "removed": [
            "Total saturated fatty acids, equated (g)",
            "Total monounsaturated fatty acids, equated (g)",
            "Total polyunsaturated fatty acids, equated (g)",
        ],
        "reason": "Detailed fatty acid groups are useful for advanced modelling but too specific for required input.",
    },
    {
        "kept": "Niacin (B3) (mg)",
        "removed": [
            "Niacin derived from tryptophan (mg)",
            "Niacin derived equivalents (mg)",
            "Tryptophan (mg)",
        ],
        "reason": "Derived niacin and tryptophan measures overlap with the direct vitamin B3 feature.",
    },
    {
        "kept": "Vitamin C (mg)",
        "removed": [
            "Dehydroascorbic acid (mg)",
            "L-ascorbic acid (mg)",
        ],
        "reason": "Vitamin C is easier to enter than its component forms.",
    },
]


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


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return nutrient component columns used as features."""
    return [
        column
        for column in df.columns
        if column
        and not column.startswith("Unnamed:")
        and column not in NON_FEATURE_COLUMNS
    ]


def build_feature_summary(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Summarise min, max, mean, std, and missing values for every feature."""
    numeric_features = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    summary = pd.DataFrame(
        {
            "feature": feature_columns,
            "min": numeric_features.min().values,
            "max": numeric_features.max().values,
            "mean": numeric_features.mean().values,
            "std": numeric_features.std().values,
            "missing_values": numeric_features.isna().sum().values,
            "missing_percent": (numeric_features.isna().mean().values * 100).round(2),
        }
    )

    return summary


def build_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count samples in each Classification label."""
    distribution = (
        df["Classification"]
        .value_counts(dropna=False)
        .rename_axis("classification")
        .reset_index(name="sample_count")
    )
    distribution["sample_percent"] = (
        distribution["sample_count"] / len(df) * 100
    ).round(2)
    return distribution


def build_feature_ranges_by_class(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Summarise feature ranges and missing values inside each Classification."""
    numeric_df = df[["Classification", *feature_columns]].copy()
    numeric_df[feature_columns] = numeric_df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    grouped = numeric_df.groupby("Classification", dropna=False)[feature_columns]
    min_values = grouped.min()
    max_values = grouped.max()
    missing_values = grouped.apply(lambda group: group[feature_columns].isna().sum())

    rows = []
    for classification in min_values.index:
        for feature in feature_columns:
            rows.append(
                {
                    "classification": classification,
                    "feature": feature,
                    "min": min_values.loc[classification, feature],
                    "max": max_values.loc[classification, feature],
                    "range": max_values.loc[classification, feature]
                    - min_values.loc[classification, feature],
                    "missing_values": missing_values.loc[classification, feature],
                }
            )

    return pd.DataFrame(rows)


def describe_class_balance(class_distribution: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Return a concise balance judgement and rare-class table."""
    max_count = class_distribution["sample_count"].max()
    min_count = class_distribution["sample_count"].min()
    imbalance_ratio = max_count / min_count if min_count else float("inf")
    rare_threshold = 5
    rare_classes = class_distribution[
        class_distribution["sample_count"] <= rare_threshold
    ].copy()

    if imbalance_ratio <= 2:
        judgement = "The classes are reasonably balanced."
    else:
        judgement = (
            "The classes are not balanced. "
            f"The largest class has {imbalance_ratio:.1f} times as many samples "
            "as the smallest class."
        )

    return judgement, rare_classes


def calculate_classes_below_sample_count(
    class_distribution_csv: Path = SHEET2_CLASS_DISTRIBUTION_CSV,
    threshold: int = 4,
) -> dict[str, float | int]:
    """Calculate how many classes have fewer samples than the given threshold."""
    class_distribution = pd.read_csv(class_distribution_csv)
    rare_classes = class_distribution[
        class_distribution["sample_count"] < threshold
    ]

    total_classes = len(class_distribution)
    rare_class_count = len(rare_classes)
    total_samples = class_distribution["sample_count"].sum()
    rare_sample_count = rare_classes["sample_count"].sum()

    return {
        "threshold": threshold,
        "total_classes": total_classes,
        "classes_below_threshold": rare_class_count,
        "class_percent": float(round(rare_class_count / total_classes * 100, 2)),
        "total_samples": int(total_samples),
        "samples_in_classes_below_threshold": int(rare_sample_count),
        "sample_percent": float(round(rare_sample_count / total_samples * 100, 2)),
    }


def build_selected_features_payload(feature_summary: pd.DataFrame) -> dict[str, object]:
    """Return JSON-serialisable metadata for the selected input feature list."""
    selected_summary = (
        feature_summary.set_index("feature")
        .loc[SELECTED_20_FEATURES]
        .reset_index()
    )

    return {
        "dataset": "AFCD Release 3 - sheet2.csv",
        "selected_feature_count": len(SELECTED_20_FEATURES),
        "selection_reason": (
            "The selected features are intended as required user inputs. They keep "
            "interpretable nutrients with low missing values, retain Moisture "
            "(water) because sheet2 includes both solids and liquids, cover several "
            "nutrient families, and avoid highly overlapping derived features."
        ),
        "selected_features": selected_summary[
            [
                "feature",
                "min",
                "max",
                "mean",
                "std",
                "missing_values",
                "missing_percent",
            ]
        ].to_dict(orient="records"),
        "removed_correlated_feature_groups": REMOVED_CORRELATED_FEATURE_GROUPS,
    }


def write_selected_features_json(
    feature_columns: list[str],
    feature_summary: pd.DataFrame,
    output_file: Path,
) -> None:
    """Write selected required input features as JSON text."""
    missing_selected_features = [
        feature for feature in SELECTED_20_FEATURES if feature not in feature_columns
    ]
    if missing_selected_features:
        raise ValueError(
            "Selected features not found in sheet2: "
            + ", ".join(missing_selected_features)
        )

    payload = build_selected_features_payload(feature_summary)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def write_text_report(
    df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    class_distribution: pd.DataFrame,
    rare_classes: pd.DataFrame,
    balance_judgement: str,
    output_file: Path,
) -> None:
    """Write a short report that directly answers the exploration questions."""
    number_of_classes = df["Classification"].nunique(dropna=True)
    missing_class_labels = df["Classification"].isna().sum()

    with output_file.open("w", encoding="utf-8") as file:
        file.write("Dataset Exploration: AFCD Release 3 - sheet2.csv\n")
        file.write("=" * 55 + "\n\n")
        file.write(f"Rows / samples: {len(df)}\n")
        file.write(f"Feature count: {len(feature_summary)}\n")
        file.write(f"Number of classifications: {number_of_classes}\n")
        file.write(f"Missing Classification labels: {missing_class_labels}\n\n")

        file.write("Feature list summary\n")
        file.write("-" * 20 + "\n")
        file.write(
            feature_summary[
                ["feature", "min", "max", "mean", "std", "missing_values"]
            ].to_string(index=False)
        )
        file.write("\n\n")

        file.write("Class distribution\n")
        file.write("-" * 18 + "\n")
        file.write(class_distribution.to_string(index=False))
        file.write("\n\n")

        file.write("Balance answer\n")
        file.write("-" * 14 + "\n")
        file.write(balance_judgement + "\n\n")

        file.write("Rare classifications\n")
        file.write("-" * 20 + "\n")
        if rare_classes.empty:
            file.write("No classifications have 5 or fewer samples.\n")
        else:
            file.write(rare_classes.to_string(index=False))
            file.write("\n")


def explore_sheet2() -> None:
    """Run nutrient feature and Classification exploration for sheet2 only."""
    df = read_sheet2()
    feature_columns = get_feature_columns(df)

    feature_summary = build_feature_summary(df, feature_columns)
    class_distribution = build_class_distribution(df)
    feature_ranges_by_class = build_feature_ranges_by_class(df, feature_columns)
    balance_judgement, rare_classes = describe_class_balance(class_distribution)

    feature_summary_file = DATA_DIR / "sheet2_feature_summary.csv"
    class_distribution_file = DATA_DIR / "sheet2_class_distribution.csv"
    class_feature_ranges_file = DATA_DIR / "sheet2_feature_ranges_by_class.csv"
    report_file = DATA_DIR / "sheet2_dataset_exploration_report.txt"
    selected_features_file = DATA_DIR / "sheet2_selected_20_features.txt"

    feature_summary.to_csv(feature_summary_file, index=False, encoding="utf-8-sig")
    class_distribution.to_csv(
        class_distribution_file,
        index=False,
        encoding="utf-8-sig",
    )
    feature_ranges_by_class.to_csv(
        class_feature_ranges_file,
        index=False,
        encoding="utf-8-sig",
    )
    write_text_report(
        df,
        feature_summary,
        class_distribution,
        rare_classes,
        balance_judgement,
        report_file,
    )
    write_selected_features_json(
        feature_columns,
        feature_summary,
        selected_features_file,
    )

    print("Dataset Exploration: AFCD Release 3 - sheet2.csv")
    print(f"Rows / samples: {len(df)}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Number of classifications: {df['Classification'].nunique(dropna=True)}")
    print("\nTop 10 classifications by sample count:")
    print(class_distribution.head(10).to_string(index=False))
    print(f"\nBalance: {balance_judgement}")
    print(f"Rare classifications with <= 5 samples: {len(rare_classes)}")
    print("\nOutput files:")
    print(f"- {feature_summary_file}")
    print(f"- {class_distribution_file}")
    print(f"- {class_feature_ranges_file}")
    print(f"- {report_file}")
    print(f"- {selected_features_file}")


if __name__ == "__main__":
    explore_sheet2()
