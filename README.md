# Project 1: Ames Housing Data Preprocessing & Feature Engineering Pipeline

## Overview & Problem Statement
In real-world predictive modeling, tabular datasets typically suffer from complex data quality issues: pervasive missing values across heterogeneous data types, irregular cardinalities, multi-collinear variables, and severe outliers. 

The objective of this project is to construct a rigorous, production-grade **Data Preprocessing and Feature Engineering Pipeline** on the comprehensive Ames Housing dataset. The goal is to clean, transform, and normalize 80 raw predictor features into an ML-ready numerical feature matrix optimized for downstream regression models predicting residential property sale prices (`SalePrice`).

---

## Dataset Description
- **Source**: Ames Housing Dataset compiled by Dean De Cock (Truman State University), describing individual residential property sales in Ames, Iowa.
- **File**: `Ames_Housing_Dat.csv`
- **Volume**: 2,930 instances, 81 attributes (80 features + 1 target).
- **Target Variable**: `SalePrice` — Continuous numerical property sale price in USD.
- **Key Feature Domains**:
  - **Property Dimensions**: `Lot Frontage`, `Lot Area`, `Gr Liv Area`, `Total Bsmt SF`, `1st Flr SF`, `2nd Flr SF`.
  - **Quality & Condition**: `Overall Qual` (1-10 scale), `Overall Cond`, `Exter Qual`, `Kitchen Qual`, `Bsmt Qual`.
  - **Age & Remodeling**: `Year Built`, `Year Remod/Add`, `Garage Yr Blt`.
  - **Zoning & Location**: `MS Zoning`, `Neighborhood`, `Condition 1`, `Bldg Type`, `House Style`.
  - **Amenities & Infrastructure**: `Alley`, `Central Air`, `Fireplaces`, `Garage Cars`, `Pool Area`, `Fence`.

---

## Methodology & Pipeline Architecture

### 1. Data Cleaning & Outlier Mitigation
- Evaluated extreme anomalies in continuous physical dimensions. Specifically filtered abnormal properties with `Gr Liv Area > 4000` sq ft with uncharacteristically low sale prices, which distort linear relationships.
- Separated unique identifiers (`PID`) to prevent data leakage.

### 2. Missing Value Imputation Strategy
- Assessed column-wise missing percentages systematically.
- Removed ultra-sparse features with >80% missing records where information was deemed non-recoverable.
- Applied domain-informed imputation:
  - **Numerical features** (`Lot Frontage`, `Mas Vnr Area`, `Garage Yr Blt`): Imputed with median or zero depending on property attributes.
  - **Categorical features** (`Alley`, `Bsmt Qual`, `Garage Type`, `Fireplace Qu`): Imputed missing values with explicit `'None'` category to represent absence of the amenity.

### 3. Feature Transformation & Categorical Encoding
- Converted nominal and ordinal categorical variables using dummy variable encoding (`pd.get_dummies`) with `drop_first=True` to eliminate multi-collinearity and the dummy variable trap.
- Separated predictors ($X$) and target ($y = \text{SalePrice}$).

### 4. Splitting & Feature Standardization
- Partitioned the dataset using an 80/20 train-test split (`test_size=0.2, random_state=2`).
- Fitted `StandardScaler` strictly on the training partition (`fit_transform`) and applied the learned parameters to transform the test partition (`transform`), strictly preventing data leakage across train and holdout sets.

---

## Key Pipeline Deliverables & Results
- **Output Space**: Transformed raw raw 81 mixed-type columns into a normalized, zero-null feature representation.
- **Leakage Prevention**: Verified strict train-only parameter estimation across imputation and scaling steps.
- **Downstream Readiness**: Feature matrix is directly consumable by linear models (OLS, Ridge, Lasso, ElasticNet) and non-linear ensembles (Random Forest, Gradient Boosting, XGBoost).

---

## Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and navigate to the project directory:
```bash
cd 01-ames-housing-feature-engineering
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Pipeline
Launch the Jupyter Notebook:
```bash
jupyter notebook Data_preprocessing_online.ipynb
```
Execute cells sequentially from top to bottom to inspect data profiles, visualizations, missing value distributions, and the final scaled feature matrices.
