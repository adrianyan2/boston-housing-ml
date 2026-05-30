# Boston Housing — End-to-End Machine Learning Project

Predicting median home values in Boston suburbs using the classic 1970 SMSA census dataset. This project covers the full data science workflow: EDA, feature engineering, model comparison, hyperparameter tuning, and explainability.

---

## Dataset

506 records, 13 features, 1 continuous target (`MEDV`).

| Feature | Description |
|---------|-------------|
| `CRIM` | Per capita crime rate by town |
| `ZN` | Proportion of residential land zoned for lots > 25,000 sq.ft. |
| `INDUS` | Proportion of non-retail business acres per town |
| `CHAS` | Charles River dummy variable (1 if tract bounds river) |
| `NOX` | Nitric oxides concentration (parts per 10 million) |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of owner-occupied units built prior to 1940 |
| `DIS` | Weighted distances to five Boston employment centres |
| `RAD` | Index of accessibility to radial highways |
| `TAX` | Full-value property-tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | 1000(Bk − 0.63)², Bk = proportion of Black residents |
| `LSTAT` | % lower status of the population |
| `MEDV` | **Target** — Median value of owner-occupied homes ($1,000s) |

---

## Project Structure

```
boston_housing_ml/
├── housing.csv                     # Raw dataset
├── boston_housing_analysis.ipynb   # Full interactive notebook
├── run_pipeline.py                 # Headless script (saves plots to outputs/)
├── build_notebook.py               # Regenerates notebook via nbformat API
├── requirements.txt
└── src/
    ├── data_loader.py              # Load CSV, column names, descriptions
    ├── preprocessing.py            # Feature engineering, train/test split
    ├── models.py                   # 9 model definitions + tuning grids
    └── evaluate.py                 # RMSE, MAE, R², MAPE, cross-validation
```

---

## Pipeline

1. **EDA** — distributions, correlation heatmap, scatter plots, outlier detection
2. **Feature Engineering** — 6 new features: `RM²`, `LOG_CRIM`, `LOG_DIS`, `LOG_LSTAT`, `TAX_PTRATIO`, `AGE_DIS`
3. **Baseline Comparison** — 9 algorithms benchmarked with 5-fold cross-validation
4. **Hyperparameter Tuning** — `RandomizedSearchCV` (20 iterations) on top 3 models
5. **Evaluation** — RMSE, MAE, R², MAPE on held-out test set (80/20 split)
6. **Explainability** — Permutation importance, built-in feature importance, SHAP
7. **Persistence** — Best model saved with `joblib`

---

## Results

| Model | CV RMSE | CV R² | Test RMSE | Test R² |
|-------|--------:|------:|----------:|--------:|
| **Gradient Boosting** | **3.43** | **0.859** | **2.36** | **0.924** |
| Extra Trees | 3.65 | 0.840 | 2.83 | 0.891 |
| Random Forest | 3.71 | 0.836 | 2.82 | 0.891 |
| SVR | 3.27 | 0.874 | 3.30 | 0.852 |
| Linear Regression | 4.05 | 0.800 | 3.71 | 0.812 |
| Ridge | 4.07 | 0.799 | 3.78 | 0.806 |
| KNN | 4.17 | 0.797 | 4.53 | 0.720 |
| Lasso | 4.28 | 0.781 | 4.26 | 0.753 |
| ElasticNet | 4.29 | 0.780 | 4.27 | 0.751 |

**Best model: Gradient Boosting** — Test RMSE of $2,360, R² = 0.924

**Top predictors:** `LSTAT` (% lower status) and `RM` (rooms per dwelling) dominate across all importance metrics.

---

## Quickstart

```bash
# Clone
git clone https://github.com/adrianyan2/boston-housing-ml.git
cd boston-housing-ml

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (saves plots to outputs/)
python run_pipeline.py

# Or open the interactive notebook
jupyter notebook boston_housing_analysis.ipynb
```
