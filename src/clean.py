"""Load, validate and clean the BCSC Risk Factors (summarized, 09/2020) dataset.

Code labels follow https://www.bcsc-research.org/datasets/rf/documentation.
Run from the project root:  python src/clean.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

EXPECTED_ROWS = 1_522_340
EXPECTED_COUNT = 6_788_436

COLUMNS = [
    "year", "age_group_5_years", "race_eth", "first_degree_hx", "age_menarche",
    "age_first_birth", "BIRADS_breast_density", "current_hrt", "menopaus",
    "bmi_group", "biophx", "breast_cancer_history", "count",
]

YES_NO = {0: "No", 1: "Yes"}

LABELS = {
    "age_group_5_years": {
        1: "18-29", 2: "30-34", 3: "35-39", 4: "40-44", 5: "45-49", 6: "50-54",
        7: "55-59", 8: "60-64", 9: "65-69", 10: "70-74", 11: "75-79",
        12: "80-84", 13: "85+",
    },
    "race_eth": {
        1: "Non-Hispanic white", 2: "Non-Hispanic black", 3: "Asian/Pacific Islander",
        4: "Native American", 5: "Hispanic", 6: "Other/mixed",
    },
    "first_degree_hx": YES_NO,
    "age_menarche": {0: ">=14", 1: "12-13", 2: "<12"},
    "age_first_birth": {0: "<20", 1: "20-24", 2: "25-29", 3: ">=30", 4: "Nulliparous"},
    "BIRADS_breast_density": {
        1: "Almost entirely fat", 2: "Scattered fibroglandular densities",
        3: "Heterogeneously dense", 4: "Extremely dense",
    },
    "current_hrt": YES_NO,
    "menopaus": {1: "Pre-/peri-menopausal", 2: "Post-menopausal", 3: "Surgical menopause"},
    "bmi_group": {1: "10-24.99", 2: "25-29.99", 3: "30-34.99", 4: "35+"},
    "biophx": YES_NO,
    "breast_cancer_history": YES_NO,
}

# Code 9 means "Unknown" for every categorical EXCEPT age_group_5_years,
# where 9 is a real value (age 65-69) and must not be nulled.
MISSING_CODE = 9
MISSING_CODE_COLS = [c for c in LABELS if c != "age_group_5_years"]
CATEGORICAL_COLS = list(LABELS)


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("bcsc_risk_factors_summarized*_092020.csv"))
    if len(files) != 3:
        raise FileNotFoundError(f"Expected 3 raw CSVs in {raw_dir}, found {len(files)}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    missing_cols = set(COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    return df[COLUMNS]


def verify_totals(df: pd.DataFrame) -> bool:
    rows, total = len(df), int(df["count"].sum())
    ok = rows == EXPECTED_ROWS and total == EXPECTED_COUNT
    print(f"Rows:       {rows:>12,}  (expected {EXPECTED_ROWS:,})")
    print(f"Sum count:  {total:>12,}  (expected {EXPECTED_COUNT:,})")
    print("Totals OK" if ok else "MISMATCH in totals!")
    return ok


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_COLS:
        codes = out[col].astype("Int64")
        if col in MISSING_CODE_COLS:
            codes = codes.mask(codes == MISSING_CODE)
        unexpected = set(codes.dropna().unique()) - set(LABELS[col])
        if unexpected:
            raise ValueError(f"{col}: undocumented codes {sorted(unexpected)}")
        out[col] = codes
        order = list(LABELS[col].values())
        out[f"{col}_label"] = pd.Categorical(codes.map(LABELS[col]), categories=order, ordered=True)

    out["high_density"] = out["BIRADS_breast_density"].map({1: 0, 2: 0, 3: 1, 4: 1}).astype("Int64")
    return out


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Share of observations (weighted by count) and of rows that are missing per column."""
    w = df["count"]
    report = pd.DataFrame({
        "missing_weighted_pct": {c: 100 * w[df[c].isna()].sum() / w.sum() for c in MISSING_CODE_COLS + ["high_density"]},
        "missing_rows_pct": {c: 100 * df[c].isna().mean() for c in MISSING_CODE_COLS + ["high_density"]},
    }).round(2).sort_values("missing_weighted_pct", ascending=False)
    return report


def main() -> None:
    raw = load_raw()
    verify_totals(raw)
    df = clean(raw)

    print("\nMissingness (weighted by 'count'):")
    print(missingness_report(df).to_string())

    complete = df[MISSING_CODE_COLS].notna().all(axis=1)
    print(f"\nObservations with no missing field: {100 * df.loc[complete, 'count'].sum() / df['count'].sum():.1f}%")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "bcsc_clean.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
