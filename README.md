# SECOM-ML-project Example

This repository contains a classroom practice notebook from a machine learning course.

## 📘 Overview

The notebook demonstrates a modeling workflow starting **after data imputation**, including:

- 🔍 **Feature reduction** using Boruta
- ⚖️ **Balancing** with SMOTE, ADASYN, RandomOverSampler
- 📏 **Scaling** via StandardScaler and MinMaxScaler
- 🤖 **Modeling** using multiple algorithms such as:
  - Random Forest
  - Logistic Regression
  - KNN
  - LDA
  - SVM (with optional normality test and transformation)
  - 
## 📂 Dataset

This project is based on the **SECOM manufacturing dataset** from the UCI Machine Learning Repository:

- 📥 [Download SECOM dataset](https://archive.ics.uci.edu/ml/datasets/SECOM)
- The dataset contains 1567 observations and 590 features collected from semiconductor manufacturing processes, with a binary classification target (`Pass/Fail`).

⚠️ The modeling starts **after data imputation**, so raw data loading and imputation steps are not included in this notebook.

## 🛠 Function-Based Design

This notebook adopts a modular design using Python functions for flexible experimentation:

1. 🔧 Various preprocessing methods (imputers, scalers, balancers, models) are stored in **dictionaries**.
2. ⚙️ The core **modeling process** is encapsulated in a reusable function.
3. 🔁 Another function uses **for-loops and recursion** to iterate through all combinations of techniques.
4. 📊 Each result is recorded and optionally written to CSV, avoiding duplication by checking identifiers.

## 📄 Output

- A summary table of evaluation metrics for each model configuration.
- Exported results saved as `model_comparison_results.csv`.

## 🧪 Key Libraries Used

- `scikit-learn`
- `imbalanced-learn`
- `boruta_py`
- `pandas`, `numpy`

---

This example serves as a reference for building standardized, reproducible modeling pipelines in real-world machine learning workflows.
