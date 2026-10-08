# =============================================================================
# src/predict.py
# Inference helper — mirrors the training encoding exactly.
# Used by app.py to preprocess user input and run predictions.
# =============================================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Ensure sibling modules are importable
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.pkl")

# ---------------------------------------------------------------------------
# Region dummy columns produced by encode_features(drop_first=True)
# drop_first removes 'region_northeast' → these three columns remain.
# ---------------------------------------------------------------------------
REGION_DUMMIES = ["region_northwest", "region_southeast", "region_southwest"]


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
def load_model(model_path: str = MODEL_PATH) -> dict:
    """Load the serialized model payload.

    Returns:
        dict with keys: 'model', 'feature_names', 'model_name'

    Raises:
        FileNotFoundError if best_model.pkl is missing.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            "Please run `python src/train.py` first to generate the model."
        )
    payload = joblib.load(model_path)
    return payload


# ---------------------------------------------------------------------------
# 2. Preprocess a single user input
# ---------------------------------------------------------------------------
def preprocess_input(
    age: float,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
    feature_names: list,
) -> pd.DataFrame:
    """Convert raw user-facing inputs into the exact feature vector used during training.

    Encoding mirrors preprocess.py → encode_features():
        sex    : female=0, male=1
        smoker : no=0,     yes=1
        region : one-hot with drop_first=True (drops 'northeast')
                 Valid values: northeast | northwest | southeast | southwest

    Args:
        age          : Integer age in years (18–64 typical).
        sex          : 'male' or 'female'.
        bmi          : Body-mass index (float).
        children     : Number of dependents (int ≥ 0).
        smoker       : 'yes' or 'no'.
        region       : One of northeast / northwest / southeast / southwest.
        feature_names: Ordered list of feature column names from training.

    Returns:
        Single-row DataFrame aligned to training feature columns.
    """
    # Normalise string inputs
    sex    = sex.strip().lower()
    smoker = smoker.strip().lower()
    region = region.strip().lower()

    # Binary encodings
    sex_enc    = 1 if sex    == "male" else 0
    smoker_enc = 1 if smoker == "yes"  else 0

    # Region one-hot (drop_first=northeast)
    region_northwest  = 1 if region == "northwest"  else 0
    region_southeast  = 1 if region == "southeast"  else 0
    region_southwest  = 1 if region == "southwest"  else 0

    row = {
        "age":               age,
        "sex":               sex_enc,
        "bmi":               bmi,
        "children":          children,
        "smoker":            smoker_enc,
        "region_northwest":  region_northwest,
        "region_southeast":  region_southeast,
        "region_southwest":  region_southwest,
    }

    # Build DataFrame aligned exactly to training columns
    input_df = pd.DataFrame([row])[feature_names]
    return input_df


# ---------------------------------------------------------------------------
# 3. Predict
# ---------------------------------------------------------------------------
def predict_premium(
    age: float,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
    model_path: str = MODEL_PATH,
) -> dict:
    """End-to-end inference: load model, encode input, predict premium.

    Args:
        age, sex, bmi, children, smoker, region: User-supplied values.
        model_path: Path to the saved .pkl file.

    Returns:
        dict with:
            'predicted_charge' : float (raw prediction)
            'formatted'        : str  (e.g. "$12,345.67")
            'model_name'       : str  (name of the model used)
    """
    payload = load_model(model_path)
    model         = payload["model"]
    feature_names = payload["feature_names"]
    model_name    = payload["model_name"]

    input_df  = preprocess_input(age, sex, bmi, children, smoker, region, feature_names)
    raw_pred  = float(model.predict(input_df)[0])

    # Clamp to realistic range (charges can't be negative)
    charge = max(0.0, raw_pred)

    return {
        "predicted_charge": charge,
        "formatted":        f"Rs. {charge:,.2f}",
        "model_name":       model_name,
    }


# ---------------------------------------------------------------------------
# Quick test (run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        # (age, sex,    bmi,  children, smoker, region)
        (25, "male",   22.0, 0, "no",  "northeast"),   # Low risk
        (55, "female", 38.0, 2, "yes", "southeast"),   # High risk (smoker, high BMI)
        (35, "male",   28.5, 1, "no",  "northwest"),   # Medium risk
    ]

    print("\nPrediction Test Cases")
    print("=" * 70)
    for age, sex, bmi, children, smoker, region in test_cases:
        try:
            result = predict_premium(age, sex, bmi, children, smoker, region)
            print(f"  Age={age}, Sex={sex}, BMI={bmi}, Children={children}, "
                  f"Smoker={smoker}, Region={region}")
            print(f"  -> Predicted Premium: {result['formatted']}  [{result['model_name']}]")
            print()
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            break
