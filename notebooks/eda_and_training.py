# =============================================================================
# notebooks/eda_and_training.py
#
# Full Exploratory Data Analysis + Model Training walkthrough.
# Run this script directly:  python notebooks/eda_and_training.py
# All plots are saved to ../outputs/eda/
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Resolve sibling src imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import build_dataset, load_data, clean_data, encode_features, get_feature_matrix
from train       import (get_candidate_models, train_and_evaluate,
                         plot_model_comparison, plot_predictions,
                         plot_feature_importance, save_model)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Output directory for EDA plots
# ---------------------------------------------------------------------------
ROOT       = os.path.join(os.path.dirname(__file__), "..")
EDA_DIR    = os.path.join(ROOT, "outputs", "eda")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
BLUE = "#4C9BE8"
RED  = "#E84C4C"
GRN  = "#4CE87A"

# ===========================================================================
# SECTION 0: Load Raw Data
# ===========================================================================
print("\n" + "=" * 65)
print("  SECTION 0 — Load & Inspect Raw Data")
print("=" * 65)

raw = load_data()
print("\nFirst 5 rows:")
print(raw.head().to_string())
print("\nData types:\n", raw.dtypes)
print("\nNull counts:\n",  raw.isnull().sum())
print("\nBasic statistics (numeric):")
print(raw.describe().to_string())

# ===========================================================================
# SECTION 1: Distribution of Charges (Target Variable)
# ===========================================================================
print("\n" + "=" * 65)
print("  SECTION 1 — Distribution of Insurance Charges")
print("=" * 65)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribution of Insurance Charges", fontsize=15, fontweight="bold")

# Histogram
axes[0].hist(raw["charges"], bins=50, color=BLUE, edgecolor="white", linewidth=0.5)
axes[0].set_xlabel("Annual Charges ($)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Histogram")

# Box plot
axes[1].boxplot(raw["charges"], vert=True, patch_artist=True,
                boxprops=dict(facecolor=BLUE, color="navy"),
                medianprops=dict(color="red", linewidth=2))
axes[1].set_ylabel("Annual Charges ($)")
axes[1].set_title("Box Plot")
axes[1].set_xticklabels(["All Patients"])

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "01_charges_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_charges_distribution.png")

# Skewness
print(f"\nCharges — Mean: ${raw['charges'].mean():,.0f}  "
      f"Median: ${raw['charges'].median():,.0f}  "
      f"Skew: {raw['charges'].skew():.2f}")

# ===========================================================================
# SECTION 2: Charges vs Categorical Variables
# ===========================================================================
print("\n" + "=" * 65)
print("  SECTION 2 — Charges by Categorical Features")
print("=" * 65)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Insurance Charges by Categorical Features", fontsize=15, fontweight="bold")

# Smoker
sns.boxplot(data=raw, x="smoker", y="charges", palette={"no": BLUE, "yes": RED},
            ax=axes[0], width=0.5)
axes[0].set_title("Smoker vs Charges")
axes[0].set_xlabel("Smoker")
axes[0].set_ylabel("Charges ($)")

# Sex
sns.violinplot(data=raw, x="sex", y="charges", palette={"male": "#A0C4FF", "female": "#FFB3BA"},
               ax=axes[1], inner="quartile")
axes[1].set_title("Sex vs Charges")
axes[1].set_xlabel("Sex")
axes[1].set_ylabel("")

# Region
region_order = raw.groupby("region")["charges"].median().sort_values(ascending=False).index
sns.boxplot(data=raw, x="region", y="charges", order=region_order,
            palette="Set2", ax=axes[2], width=0.5)
axes[2].set_title("Region vs Charges")
axes[2].set_xlabel("Region")
axes[2].set_ylabel("")
axes[2].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "02_charges_by_category.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_charges_by_category.png")

# Print medians
print("\nMedian charges by smoker:\n",
      raw.groupby("smoker")["charges"].median().apply(lambda x: f"${x:,.0f}"))
print("\nMedian charges by region:\n",
      raw.groupby("region")["charges"].median().sort_values(ascending=False).apply(lambda x: f"${x:,.0f}"))

# ===========================================================================
# SECTION 3: Charges vs Numeric Variables
# ===========================================================================
print("\n" + "=" * 65)
print("  SECTION 3 — Charges vs Numeric Features")
print("=" * 65)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Charges vs Numeric Features (coloured by Smoker)", fontsize=15, fontweight="bold")

colors_smoker = raw["smoker"].map({"no": BLUE, "yes": RED})

for ax, feat, label in zip(axes,
                            ["age", "bmi", "children"],
                            ["Age (years)", "BMI", "Number of Children"]):
    ax.scatter(raw[feat], raw["charges"], c=colors_smoker, alpha=0.4, s=20, edgecolors="none")
    ax.set_xlabel(label)
    ax.set_ylabel("Charges ($)")
    ax.set_title(f"Charges vs {label}")

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,  label="Smoker",     markersize=10),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, label="Non-Smoker", markersize=10),
]
axes[0].legend(handles=legend_elements, loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "03_charges_vs_numeric.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_charges_vs_numeric.png")

# ===========================================================================
# SECTION 4: Correlation Heatmap
# ===========================================================================
print("\n" + "=" * 65)
print("  SECTION 4 — Correlation Heatmap (Encoded Features)")
print("=" * 65)

clean   = clean_data(raw)
encoded = encode_features(clean)

fig, ax = plt.subplots(figsize=(10, 7))
corr = encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 10})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "04_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_correlation_heatmap.png")

# Print top correlations with charges
print("\nTop correlations with 'charges':")
print(corr["charges"].drop("charges").sort_values(key=abs, ascending=False).to_string())

# ===========================================================================
# SECTION 5: BMI & Age Group Analysis
# ===========================================================================
print("\n" + "=" * 65)
print("  SECTION 5 — BMI & Age Group Segmentation")
print("=" * 65)

df = clean.copy()

# Age buckets
df["age_group"] = pd.cut(df["age"], bins=[17, 25, 35, 45, 55, 65],
                          labels=["18-25", "26-35", "36-45", "46-55", "56-65"])
# BMI category
df["bmi_cat"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 100],
                        labels=["Underweight", "Normal", "Overweight", "Obese"])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Charges by Age Group & BMI Category (split by Smoker)", fontsize=14, fontweight="bold")

sns.barplot(data=df, x="age_group", y="charges", hue="smoker",
            palette={"no": BLUE, "yes": RED}, ax=axes[0],
            estimator=np.median, errorbar=None)
axes[0].set_title("Median Charges by Age Group")
axes[0].set_xlabel("Age Group")
axes[0].set_ylabel("Median Charges ($)")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

sns.barplot(data=df, x="bmi_cat", y="charges", hue="smoker",
            palette={"no": BLUE, "yes": RED}, ax=axes[1],
            estimator=np.median, errorbar=None)
axes[1].set_title("Median Charges by BMI Category")
axes[1].set_xlabel("BMI Category")
axes[1].set_ylabel("")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

for ax in axes:
    ax.legend(title="Smoker")

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "05_age_bmi_segmentation.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_age_bmi_segmentation.png")

# ===========================================================================
# SECTION 6: Model Training & Evaluation
# ===========================================================================
print("\n" + "=" * 65)
print("  SECTION 6 — Model Training & Evaluation")
print("=" * 65)

X, y, feature_names = get_feature_matrix(encoded)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")

models     = get_candidate_models()
results_df, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test, models)

print("\n" + "=" * 55)
print("  Final Model Comparison Table")
print("=" * 55)
print(results_df.to_string(index=False))
print("=" * 55)

# Save comparison plot to EDA dir as well
plot_model_comparison(results_df,
                      save_path=os.path.join(EDA_DIR, "06_model_comparison.png"))

# ===========================================================================
# SECTION 7: Best Model Details
# ===========================================================================
print("\n" + "=" * 65)
print("  SECTION 7 — Best Model Details")
print("=" * 65)

best_name  = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
print(f"\nBest model: {best_name}")

plot_predictions(best_model, X_test, y_test, best_name,
                 save_path=os.path.join(EDA_DIR, "07_predictions_scatter.png"))
plot_feature_importance(best_model, feature_names,
                        save_path=os.path.join(EDA_DIR, "08_feature_importance.png"))

# ===========================================================================
# SECTION 8: Save the Best Model
# ===========================================================================
print("\n" + "=" * 65)
print("  SECTION 8 — Saving Best Model")
print("=" * 65)

model_path = os.path.join(MODELS_DIR, "best_model.pkl")
save_model(best_model, feature_names, path=model_path)

print("\n✓  EDA & Training script complete!")
print(f"   Plots  → {EDA_DIR}")
print(f"   Model  → {model_path}")
print("=" * 65)
