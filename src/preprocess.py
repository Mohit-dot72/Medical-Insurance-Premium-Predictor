# =============================================================================
# src/preprocess.py
# Data loading, cleaning, and feature engineering for insurance premium model.
# =============================================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "insurance.csv")


# ---------------------------------------------------------------------------
# 1. Load & Validate
# ---------------------------------------------------------------------------
def load_data(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    """Load the insurance CSV and perform basic validation.

    Args:
        csv_path: Path to insurance.csv

    Returns:
        Raw DataFrame (unchanged except column-name stripping).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    # --- sanity checks ---
    expected_cols = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    print(f"[preprocess] Loaded {len(df)} rows × {len(df.columns)} columns.")
    return df


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and obvious data-quality issues.

    Strategy:
        • Numeric nulls → median imputation
        • Categorical nulls → mode imputation
        • Negative charges → abs()
        • Negative age/bmi/children → drop rows

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame (copy).
    """
    df = df.copy()

    # Drop fully-duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[preprocess] Dropped {before - len(df)} duplicate rows.")

    # Numeric columns
    num_cols = ["age", "bmi", "children", "charges"]
    for col in num_cols:
        null_count = df[col].isnull().sum()
        if null_count:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[preprocess] Imputed {null_count} nulls in '{col}' with median={median_val:.2f}")

    # Categorical columns
    cat_cols = ["sex", "smoker", "region"]
    for col in cat_cols:
        null_count = df[col].isnull().sum()
        if null_count:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"[preprocess] Imputed {null_count} nulls in '{col}' with mode='{mode_val}'")

    # Fix negative/invalid numeric values
    df["charges"] = df["charges"].abs()
    df = df[df["age"] > 0]
    df = df[df["bmi"] > 0]
    df = df[df["children"] >= 0]

    print(f"[preprocess] After cleaning: {len(df)} rows remain.")
    return df


# ---------------------------------------------------------------------------
# 3. Encode
# ---------------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables to numeric form.

    Encoding scheme (mirrors what app.py must replicate exactly):
        • sex    : label-encode  → female=0, male=1
        • smoker : label-encode  → no=0,     yes=1
        • region : one-hot encode with drop_first=True
                   (drops 'northeast'; creates northwest, southeast, southwest dummies)

    Args:
        df: Cleaned DataFrame (contains original string columns)

    Returns:
        Encoded DataFrame (copy) – ready for model training.
    """
    df = df.copy()

    # Binary label encoding (consistent alphabetical order)
    df["sex"]    = df["sex"].str.strip().str.lower().map({"female": 0, "male": 1})
    df["smoker"] = df["smoker"].str.strip().str.lower().map({"no": 0, "yes": 1})

    # One-hot encode region (drop_first drops 'northeast')
    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    # Cast boolean dummies to int for sklearn compatibility
    region_cols = [c for c in df.columns if c.startswith("region_")]
    df[region_cols] = df[region_cols].astype(int)

    return df


# ---------------------------------------------------------------------------
# 4. Feature / Target Split
# ---------------------------------------------------------------------------
def get_feature_matrix(df_encoded: pd.DataFrame):
    """Split encoded DataFrame into feature matrix X and target vector y.

    Args:
        df_encoded: Output of encode_features()

    Returns:
        Tuple (X: DataFrame, y: Series, feature_names: List[str])
    """
    target_col = "charges"
    feature_names = [c for c in df_encoded.columns if c != target_col]
    X = df_encoded[feature_names]
    y = df_encoded[target_col]
    return X, y, feature_names


# ---------------------------------------------------------------------------
# 5. Full pipeline (convenience)
# ---------------------------------------------------------------------------
def build_dataset(csv_path: str = DEFAULT_CSV):
    """End-to-end preprocessing pipeline.

    Returns:
        (X, y, feature_names, raw_df)
    """
    raw_df  = load_data(csv_path)
    clean   = clean_data(raw_df)
    encoded = encode_features(clean)
    X, y, feature_names = get_feature_matrix(encoded)
    return X, y, feature_names, raw_df


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X, y, features, raw = build_dataset()
    print("\nFeature columns:", features)
    print("X shape:", X.shape)
    print("y stats:\n", y.describe())
