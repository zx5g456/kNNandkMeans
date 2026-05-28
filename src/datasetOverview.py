from pathlib import Path
import os

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SHEETS_TO_EXPORT = [
    "All solids & liquids per 100 g",
    "Liquids only per 100 mL",
]

CSV_SHEETS_TO_ANALYSE = {
    "sheet2": "All solids & liquids per 100 g",
    "sheet3": "Liquids only per 100 mL",
}

DESCRIPTION_KEYWORDS = ["food", "name", "description", "survey", "id"]
LABEL_KEYWORDS = ["group", "classification", "category", "class", "code"]
NON_NUTRIENT_COLUMNS = {
    "Public Food Key",
    "Classification",
    "Derivation",
    "Energy with dietary fibre, equated (kJ)",
    "Energy, without dietary fibre, equated (kJ)",
    "Food Name",
}


def normalise_column_name(column: object) -> str:
    """Return a readable single-line column name."""
    return " ".join(str(column).split())


def read_nutrient_sheet(csv_file: Path) -> pd.DataFrame:
    """Read a nutrient profile CSV using the real header row."""
    return pd.read_csv(csv_file, header=2).dropna(how="all")


def get_nutrient_components(columns: pd.Index) -> list[str]:
    """Return nutrient component columns, excluding identifiers and energy totals."""
    return [
        normalise_column_name(column)
        for column in columns
        if normalise_column_name(column)
        and normalise_column_name(column) not in NON_NUTRIENT_COLUMNS
        and not str(column).startswith("Unnamed:")
    ]


def classify_columns(columns: pd.Index) -> dict[str, list[str]]:
    """Classify dataset columns into description, label, and nutrient feature groups."""
    column_groups = {
        "food name / description": [],
        "classification label": [],
        "nutrient features": [],
    }

    for column in columns:
        column_name = normalise_column_name(column)

        if not column_name or str(column).startswith("Unnamed:"):
            continue

        if column_name in {"Public Food Key", "Food Name"}:
            column_groups["food name / description"].append(column_name)
        elif column_name in {"Classification", "Derivation"}:
            column_groups["classification label"].append(column_name)
        elif column_name not in NON_NUTRIENT_COLUMNS:
            column_groups["nutrient features"].append(column_name)

    return column_groups


# Helper to find CSV files for sheet2 and sheet3
def find_sheet_csv_files(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Find the CSV files corresponding to sheet2 and sheet3 in the data folder."""
    csv_files: dict[str, Path] = {}

    for sheet_key in CSV_SHEETS_TO_ANALYSE:
        matched_files = list(data_dir.glob(f"*{sheet_key}.csv"))

        if not matched_files:
            raise FileNotFoundError(f"Could not find a CSV file ending with {sheet_key}.csv in {data_dir}")

        csv_files[sheet_key] = matched_files[0]

    return csv_files


def describe_sheet(df: pd.DataFrame, sheet_name: str) -> dict[str, object]:
    """Return a simple structural overview for one worksheet."""
    column_groups = classify_columns(df.columns)
    nutrient_components = get_nutrient_components(df.columns)

    return {
        "sheet_name": sheet_name,
        "rows": len(df),
        "columns": len(df.columns),
        "nutrient_components": nutrient_components,
        "row_meaning": "Each row represents one food item or food record in the nutrient dataset.",
        "column_groups": column_groups,
    }


def visualise_sheet_overview(sheet_overviews: list[dict[str, object]], output_path: Path) -> None:
    """Create a bar chart comparing rows, columns, and nutrient counts across sheets."""
    sheet_names = [overview["sheet_name"] for overview in sheet_overviews]
    rows = [overview["rows"] for overview in sheet_overviews]
    columns = [overview["columns"] for overview in sheet_overviews]
    nutrient_features = [
        len(overview["nutrient_components"])
        for overview in sheet_overviews
    ]

    x_positions = range(len(sheet_names))
    bar_width = 0.25

    plt.figure(figsize=(11, 6))
    plt.bar([x - bar_width for x in x_positions], rows, width=bar_width, label="Rows")
    plt.bar(x_positions, columns, width=bar_width, label="Columns")
    plt.bar(
        [x + bar_width for x in x_positions],
        nutrient_features,
        width=bar_width,
        label="Nutrient components",
    )

    plt.xticks(list(x_positions), sheet_names, rotation=15, ha="right")
    plt.ylabel("Count")
    plt.title("Dataset Structure Overview by Sheet")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



def overview_xlsx_sheets(data_dir: Path = DATA_DIR) -> None:
    """Print and visualise the structure of sheet2.csv and sheet3.csv."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    csv_files = find_sheet_csv_files(data_dir)
    sheet_overviews = []

    print("\nDataset overview for sheet2.csv and sheet3.csv")

    for sheet_key, csv_file in csv_files.items():
        sheet_name = CSV_SHEETS_TO_ANALYSE[sheet_key]
        df = read_nutrient_sheet(csv_file)
        overview = describe_sheet(df, sheet_name)
        sheet_overviews.append(overview)

        print(f"\nCSV file: {csv_file.name}")
        print(f"Sheet meaning: {overview['sheet_name']}")
        print(f"Rows: {overview['rows']}")
        print(f"Columns: {overview['columns']}")
        print(f"Nutrient components: {len(overview['nutrient_components'])}")
        print(f"Row meaning: {overview['row_meaning']}")
        print("Column groups:")

        for group_name, group_columns in overview["column_groups"].items():
            print(f"  - {group_name}: {len(group_columns)} columns")
            print(f"    {group_columns[:8]}")

    output_path = data_dir / "sheet2_sheet3_overview.png"
    visualise_sheet_overview(sheet_overviews, output_path)
    print(f"\nVisualisation saved to: {output_path}")


def convert_xlsx_to_csv(data_dir: Path = DATA_DIR) -> None:
    """Convert selected sheets from all .xlsx files in the data folder to CSV files."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    xlsx_files = list(data_dir.glob("*.xlsx"))

    if not xlsx_files:
        print("No .xlsx files found in the data folder.")
        return

    for xlsx_file in xlsx_files:
        for sheet_name in SHEETS_TO_EXPORT:
            safe_sheet_name = (
                sheet_name.lower()
                .replace(" ", "_")
                .replace("&", "and")
                .replace("/", "_")
            )
            csv_file = xlsx_file.with_name(f"{xlsx_file.stem}_{safe_sheet_name}.csv")

            df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
            df.to_csv(csv_file, index=False, encoding="utf-8-sig")

            print(f"Converted: {xlsx_file.name} [{sheet_name}] -> {csv_file.name}")

def count_food_components_by_sheet(data_dir: Path = DATA_DIR) -> None:
    """Count, print, and save nutrient components in sheet2.csv and sheet3.csv."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    csv_files = find_sheet_csv_files(data_dir)

    print("\nNutrient component count for sheet2.csv and sheet3.csv")

    for sheet_key, csv_file in csv_files.items():
        sheet_name = CSV_SHEETS_TO_ANALYSE[sheet_key]
        df = read_nutrient_sheet(csv_file)
        nutrient_components = get_nutrient_components(df.columns)

        output_file = data_dir / f"{sheet_key}_nutrient_components.txt"

        with output_file.open("w", encoding="utf-8") as file:
            file.write(f"{csv_file.name} ({sheet_name})\n")
            file.write(f"Rows: {len(df)}\n")
            file.write(f"Columns: {len(df.columns)}\n")
            file.write(f"Total nutrient components: {len(nutrient_components)}\n\n")

            print(f"\n{csv_file.name} ({sheet_name})")
            print(f"Rows: {len(df)}")
            print(f"Columns: {len(df.columns)}")
            print(f"Total nutrient components: {len(nutrient_components)}")

            for index, nutrient_component in enumerate(nutrient_components, start=1):
                line = f"{index}. {nutrient_component}"
                print(line)
                file.write(line + "\n")

        print(f"\nSaved all nutrient components to: {output_file}")

if __name__ == "__main__":
    # convert_xlsx_to_csv()
    # overview_xlsx_sheets()
    count_food_components_by_sheet()
