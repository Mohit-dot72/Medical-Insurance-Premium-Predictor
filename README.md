# Medical Insurance Premium Predictor 🏥

An end-to-end machine learning project that predicts a person's annual medical insurance premium from six personal attributes. Includes data preprocessing, EDA visualizations, multi-model training, evaluation, and a Streamlit web app.

---

## 📁 Project Structure

```
insurance_predictor/
├── data/
│   └── insurance.csv            # 1,338-row insurance dataset
├── src/
│   ├── preprocess.py            # Data loading, cleaning, encoding
│   ├── train.py                 # Model training, evaluation, persistence
│   ├── predict.py               # Inference helper (used by app.py)
│   └── generate_dataset.py      # Downloads clean CSV from GitHub
├── notebooks/
│   └── eda_and_training.py      # Standalone EDA + full training walkthrough
├── models/
│   └── best_model.pkl           # Serialized best model (auto-generated)
├── outputs/
│   ├── model_comparison.png     # Bar chart of R²/RMSE/MAE across models
│   ├── predictions_scatter.png  # Actual vs predicted scatter
│   ├── feature_importance.png   # Random Forest feature importances
│   └── eda/                     # All EDA plots (8 figures)
├── app.py                       # Streamlit web application
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / navigate to the project directory

```bash
cd insurance_predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

This will:
- Load and preprocess `data/insurance.csv`
- Train Linear Regression, Decision Tree, and Random Forest models
- Print a comparison table (R², RMSE, MAE)
- Save plots to `outputs/`
- Serialize the best model to `models/best_model.pkl`

### 4. (Optional) Run the full EDA notebook

```bash
python notebooks/eda_and_training.py
```

Generates 8 detailed EDA plots in `outputs/eda/`.

### 5. Launch the web app

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## 📊 Dataset

| Column   | Type        | Description                              |
|----------|-------------|------------------------------------------|
| age      | Numeric     | Age of the primary beneficiary           |
| sex      | Categorical | Insurance contractor gender (male/female)|
| bmi      | Numeric     | Body Mass Index                          |
| children | Numeric     | Number of dependents covered             |
| smoker   | Categorical | Smoking status (yes/no)                  |
| region   | Categorical | US region (NE / NW / SE / SW)            |
| charges  | Numeric     | Individual medical costs (target)        |

Source: Kaggle / R ML datasets repository (1,338 rows, 7 columns)

---

## 🤖 Models Trained

| Model               | Typical R² | Typical RMSE |
|---------------------|-----------|--------------|
| Linear Regression   | ~0.78     | ~$5,800      |
| Decision Tree       | ~0.75     | ~$6,100      |
| **Random Forest**   | **~0.87** | **~$4,200**  |

> Random Forest is selected as the best model and serialized to `models/best_model.pkl`.

---

## 🌐 Web App Features

- **Dark glassmorphism UI** with gradient typography
- **Six input fields**: Age, Gender, BMI, Children, Smoker, Region
- **Real-time prediction** with animated result card
- **Risk tier badge**: Low / Medium / High based on predicted premium
- **Factor pills**: Human-readable contributing factors
- **Monthly breakdown**: Annual and per-month estimates
- **Input validation** with descriptive error messages
- **Model info panel**: Performance metrics, feature descriptions, EDA insights

---

## 🔧 Encoding Details

Training encoding (replicated exactly at inference time in `predict.py`):

| Feature | Encoding                                               |
|---------|--------------------------------------------------------|
| sex     | Label: female=0, male=1                                |
| smoker  | Label: no=0, yes=1                                     |
| region  | One-hot with drop_first=True (drops 'northeast')       |
|         | → Dummies: region_northwest, region_southeast, region_southwest |

---

## 🔮 Extending the Project

### Add SHAP Explainability

```python
import shap
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"CV R² = {scores.mean():.4f} ± {scores.std():.4f}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
params = {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]}
gs = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=5, scoring="r2")
gs.fit(X_train, y_train)
```

### Deploy to Streamlit Cloud

1. Push the project to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo, branch, and `app.py`
4. Click Deploy — done!

> **Note**: Include `models/best_model.pkl` in your repo, or add a `setup.py` / `Makefile` that runs `python src/train.py` as a build step.

---

## 📦 Tech Stack

| Library      | Purpose                          |
|--------------|----------------------------------|
| pandas       | Data loading & manipulation      |
| numpy        | Numerical computing              |
| scikit-learn | Model training & evaluation      |
| matplotlib   | Plotting                         |
| seaborn      | Statistical visualizations       |
| joblib       | Model serialization              |
| streamlit    | Web application framework        |

---

## 📝 License

MIT — feel free to use, adapt, and extend this project.
