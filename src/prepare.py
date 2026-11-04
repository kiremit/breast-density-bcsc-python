"""Build train/test modeling tables for predicting high_density.

Outputs (data/processed/), one row per covariate pattern, columns = features + target + sample_weight:
  model_missingcat_{train,test}.parquet  missing values are their own one-hot category ("<col>_Unknown")
  model_nan_{train,test}.parquet         missing values stay NaN across that variable's dummies
                                         (for XGBoost's native missing handling or later imputation)
  model_menohrt_{train,test}.parquet     no 'year'; menopaus + current_hrt combined into meno_hrt;
                                         missing-as-category; identical covariate patterns are
                                         aggregated (weights summed), which leaves weighted fits unchanged
All versions use the same split. Run from the project root after clean.py:  python src/prepare.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "high_density"
WEIGHT = "sample_weight"
# BIRADS_breast_density defines the target; breast_cancer_history is excluded by design.
CATEGORICAL_FEATURES = [
    "age_group_5_years", "race_eth", "first_degree_hx", "age_menarche", "age_first_birth",
    "current_hrt", "menopaus", "bmi_group", "biophx",
]
NUMERIC_FEATURES = ["year"]
MENO_HRT_FEATURES = [c for c in CATEGORICAL_FEATURES if c not in ("menopaus", "current_hrt")] + ["meno_hrt"]
MENO_HRT_LEVELS = ["Premenopausal", "Postmenopausal, no HRT", "Postmenopausal, HRT"]


def add_meno_hrt(df: pd.DataFrame) -> pd.DataFrame:
    """Premenopausal / postmenopausal (natural or surgical) by current HRT use; anything else is Unknown."""
    post = df["menopaus"].isin([2, 3])
    meno_hrt = pd.Series(pd.NA, index=df.index, dtype="object")
    meno_hrt[df["menopaus"] == 1] = MENO_HRT_LEVELS[0]
    meno_hrt[post & (df["current_hrt"] == 0)] = MENO_HRT_LEVELS[1]
    meno_hrt[post & (df["current_hrt"] == 1)] = MENO_HRT_LEVELS[2]
    return df.assign(meno_hrt_label=pd.Categorical(meno_hrt, categories=MENO_HRT_LEVELS, ordered=True))


def one_hot(df: pd.DataFrame, missing_as_category: bool, categorical=CATEGORICAL_FEATURES,
            numeric=NUMERIC_FEATURES) -> pd.DataFrame:
    parts = [df[numeric]]
    for col in categorical:
        labels = df[f"{col}_label"]
        if missing_as_category and labels.isna().any():
            labels = labels.cat.add_categories("Unknown").fillna("Unknown")
        dummies = pd.get_dummies(labels, prefix=col, prefix_sep="_", dtype="float32")
        if not missing_as_category:
            dummies.loc[labels.isna()] = float("nan")
        parts.append(dummies)
    return pd.concat(parts, axis=1)


def safe_name(col: str) -> str:
    """Column names XGBoost accepts (no '[', ']' or '<'): '<12' -> 'lt12', '>=30' -> 'ge30'."""
    return col.replace(" ", "_").replace(",", "").replace(">=", "ge").replace("<", "lt")


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "bcsc_clean.parquet")
    df = df[df[TARGET].notna()].reset_index(drop=True)
    y = df[TARGET].astype("int8")
    w = df["count"].astype("int32")

    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=y, random_state=42)

    for name, missing_as_category in [("missingcat", True), ("nan", False)]:
        X = one_hot(df, missing_as_category)
        X.columns = [safe_name(c) for c in X.columns]
        table = X.assign(**{TARGET: y, WEIGHT: w})
        for split, idx in [("train", train_idx), ("test", test_idx)]:
            part = table.loc[idx].reset_index(drop=True)
            path = PROCESSED_DIR / f"model_{name}_{split}.parquet"
            part.to_parquet(path, index=False)
            dense = (part[TARGET] * part[WEIGHT]).sum() / part[WEIGHT].sum()
            print(f"{path.name:<32} rows={len(part):>9,}  weight={part[WEIGHT].sum():>10,}  "
                  f"features={X.shape[1]}  weighted dense share={dense:.3f}")

    df = add_meno_hrt(df)
    X = one_hot(df, missing_as_category=True, categorical=MENO_HRT_FEATURES, numeric=[])
    X.columns = [safe_name(c) for c in X.columns]
    table = X.assign(**{TARGET: y, WEIGHT: w})
    for split, idx in [("train", train_idx), ("test", test_idx)]:
        part = (table.loc[idx].groupby(list(X.columns) + [TARGET], as_index=False)[WEIGHT].sum()
                .astype({TARGET: "int8", WEIGHT: "int32"}))
        path = PROCESSED_DIR / f"model_menohrt_{split}.parquet"
        part.to_parquet(path, index=False)
        dense = (part[TARGET] * part[WEIGHT]).sum() / part[WEIGHT].sum()
        print(f"{path.name:<32} rows={len(part):>9,}  weight={part[WEIGHT].sum():>10,}  "
              f"features={X.shape[1]}  weighted dense share={dense:.3f}")
    print("meno_hrt (weighted %):")
    print((df.groupby(df["meno_hrt_label"].cat.add_categories("Unknown").fillna("Unknown"), observed=True)["count"]
           .sum() / df["count"].sum() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
