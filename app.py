# =============================================================================
# app.py — Medical Insurance Premium Predictor
# Streamlit single-page web application
#
# Run:  python -m streamlit run app.py
# =============================================================================

import os
import sys
import time
import streamlit as st
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

# pyrefly: ignore [missing-import]
from predict import load_model, preprocess_input

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🏥",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏥 InsureIQ")
    st.caption("AI-powered medical insurance premium estimator")
    st.divider()

    st.subheader("How it works")
    st.markdown("1. Fill in your details\n2. Click **Predict Premium**\n3. See your estimate instantly")
    st.divider()

    st.subheader("Model")
    try:
        payload    = load_model()
        model_name = payload["model_name"]
        feature_ct = len(payload["feature_names"])
        st.success(f"**{model_name}**")
        st.caption(f"{feature_ct} features | 1,338 training records")
    except Exception:
        st.warning("Model not found. Run `python src/train.py` first.")

    st.divider()
    st.caption("For educational purposes only. Not for actual underwriting.")

# ---------------------------------------------------------------------------
# Main heading
# ---------------------------------------------------------------------------
st.title("Medical Insurance Premium Predictor")
st.write("Enter your details below to get an estimated annual insurance premium in Indian Rupees.")
st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")
    children = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)

with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    smoker = st.selectbox("Smoker?", ["No", "Yes"])
    region = st.selectbox("US Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# BMI hint
bmi_cat = (
    "Underweight (< 18.5)" if bmi < 18.5
    else "Normal (18.5 – 24.9)" if bmi < 25
    else "Overweight (25 – 29.9)" if bmi < 30
    else "Obese (>= 30)"
)
st.caption(f"BMI category: **{bmi_cat}**")

st.write("")

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
predict_clicked = st.button("Predict Premium", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_clicked:

    # Validation
    errors = []
    if not (18 <= age <= 100):
        errors.append("Age must be between 18 and 100.")
    if not (10.0 <= bmi <= 60.0):
        errors.append("BMI must be between 10 and 60.")
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    with st.spinner("Calculating your premium..."):
        time.sleep(0.5)
        try:
            payload       = load_model()
            model         = payload["model"]
            feature_names = payload["feature_names"]

            input_df = preprocess_input(
                age=age,
                sex=gender.lower(),
                bmi=bmi,
                children=int(children),
                smoker=smoker.lower(),
                region=region.lower(),
                feature_names=feature_names,
            )

            USD_TO_INR = 84.0
            raw_pred   = float(model.predict(input_df)[0])
            charge     = max(0.0, raw_pred) * USD_TO_INR
            monthly    = charge / 12.0

        except FileNotFoundError:
            st.error("Model file not found. Please run `python src/train.py` first.")
            st.stop()
        except Exception as ex:
            st.error(f"Prediction error: {ex}")
            st.stop()

    st.divider()
    st.subheader("Estimated Premium")

    # Main result metrics
    m1, m2 = st.columns(2)
    m1.metric("Annual Premium", f"Rs. {charge:,.0f}")
    m2.metric("Monthly Premium", f"Rs. {monthly:,.0f}")

    # Risk tier
    if charge < 840_000:
        st.success("Risk Tier: LOW")
    elif charge < 2_520_000:
        st.warning("Risk Tier: MEDIUM")
    else:
        st.error("Risk Tier: HIGH")

    # Key factors
    factors = []
    if smoker == "Yes":
        factors.append("Smoking (strongest predictor)")
    if bmi >= 30:
        factors.append(f"Obese BMI ({bmi:.1f})")
    elif bmi >= 25:
        factors.append(f"Overweight BMI ({bmi:.1f})")
    if age >= 50:
        factors.append(f"Senior age ({age} yrs)")
    if children >= 3:
        factors.append(f"Large family ({children} dependents)")

    if factors:
        st.write("**Factors driving your premium:**")
        for f in factors:
            st.write(f"- {f}")
    else:
        st.write("**Profile:** Generally low-risk — healthy BMI, non-smoker, younger age.")

    # Input summary
    with st.expander("Show input summary"):
        st.dataframe(
            pd.DataFrame({
                "Field": ["Age", "Gender", "BMI", "Dependents", "Smoker", "Region"],
                "Value": [age, gender, f"{bmi:.1f}", children, smoker, region],
            }).set_index("Field"),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Info sections
# ---------------------------------------------------------------------------
st.divider()

with st.expander("About the model"):
    st.markdown("""
**Dataset:** 1,338 US insurance records with features: age, sex, BMI, children, smoker, region.

**Models trained and compared:**

| Model | R2 | RMSE (Rs.) | MAE (Rs.) |
|---|---|---|---|
| Random Forest | 0.89 | ~3,78,000 | ~2,12,000 |
| Decision Tree | 0.82 | ~4,77,000 | ~2,22,000 |
| Linear Regression | 0.81 | ~5,00,000 | ~3,51,000 |

**Selected model:** Random Forest (highest R2 score)

**Currency:** Predictions are converted from USD to INR at 1 USD = Rs. 84.

**Key insight:** Smoking status is the single most important predictor — smokers pay 3–4x more.
    """)

with st.expander("Model performance metrics"):
    metrics_path = os.path.join(os.path.dirname(__file__), "outputs", "model_metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        USD_TO_INR = 84.0
        if "RMSE" in df.columns:
            df["RMSE (Rs.)"] = (df["RMSE"] * USD_TO_INR).round(0).astype(int)
            df["MAE (Rs.)"]  = (df["MAE"]  * USD_TO_INR).round(0).astype(int)
            df = df.drop(columns=["RMSE", "MAE"])
        st.dataframe(df, use_container_width=True)
        st.caption("RMSE and MAE converted from USD to INR (1 USD = Rs. 84)")
    else:
        st.info("Run `python src/train.py` to generate metrics.")

st.divider()
st.caption("InsureIQ — For educational purposes only. Not actual insurance advice.")
