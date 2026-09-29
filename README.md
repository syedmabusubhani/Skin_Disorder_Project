# Skin Disorder Prediction

**Differential Diagnosis of Erythemato-Squamous Diseases (PRCP-1027)**

A machine learning project that predicts the type of erythemato-squamous skin disease from clinical and histopathological findings, and turns the model's results into practical suggestions for doctors.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-green)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models and Results](#models-and-results)
- [Key Findings](#key-findings)
- [Recommendations for Doctors](#recommendations-for-doctors)
- [Challenges and Solutions](#challenges-and-solutions)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)

---

## Overview

Six skin diseases in the erythemato-squamous family share very similar clinical signs (erythema and scaling), which makes early differential diagnosis difficult and often requires a biopsy. This project analyses a dermatology dataset, compares seven classification models, and identifies the features that best separate the six diseases.

**Domain:** Healthcare

## Problem Statement

1. Prepare a complete data analysis report on the given data.
2. Create a predictive model using ML techniques to predict the class of skin disease.
3. Provide suggestions to doctors for identifying skin diseases at the earliest.

## Dataset

- **Records:** 366 patients
- **Attributes:** 34 (12 clinical + 22 histopathological + age)
- **Feature scale:** clinical and histopathological features are rated 0–3; family history is binary (0/1)
- **Target:** 6 disease classes

| Class | Disease |
|---|---|
| 1 | Psoriasis |
| 2 | Seboreic Dermatitis |
| 3 | Lichen Planus |
| 4 | Pityriasis Rosea |
| 5 | Chronic Dermatitis |
| 6 | Pityriasis Rubra Pilaris |

**Class balance:** moderately imbalanced. Psoriasis has 112 cases, while Pityriasis Rubra Pilaris has only 20.

> The notebook expects the data file `skin_disorder.csv` in the same folder as the notebook.

## Project Workflow

1. **Import libraries** – NumPy, pandas, Matplotlib, Seaborn, scikit-learn
2. **Load data** – shape, head/tail, info, summary statistics, missing values
3. **Data cleaning** – the `Age` column contained `?` for 8 patients; converted to `NaN` and imputed with the median
4. **Exploratory data analysis** – class distribution, age distribution, correlation heatmap, top features correlated with the class
5. **Preprocessing** – stratified 80/20 train/test split and `StandardScaler` for scale-sensitive models
6. **Modeling** – seven classifiers evaluated with a held-out test set and stratified 5-fold cross-validation
7. **Model comparison and diagnostics** – confusion matrix, classification report, feature importance
8. **Challenges report** – problems faced and the techniques used
9. **Suggestions to doctors** – recommendations based on feature importance
10. **Conclusion**

## Models and Results

Models were trained on scaled features (Logistic Regression, SVM, KNN) or raw features (tree-based models and Naive Bayes).

| Rank | Model | Test Accuracy | Weighted F1 | 5-fold CV Accuracy |
|---|---|---|---|---|
| 1 | **Random Forest** | 0.960 | 0.960 | 0.975 ± 0.016 |
| 2 | SVM (RBF) | 0.973 | 0.973 | 0.973 ± 0.017 |
| 3 | Gradient Boosting | 0.932 | 0.931 | 0.970 ± 0.005 |
| 4 | Logistic Regression | 0.960 | 0.961 | 0.970 ± 0.016 |
| 5 | KNN | 0.905 | 0.908 | 0.962 ± 0.022 |
| 6 | Decision Tree | 0.932 | 0.929 | 0.940 ± 0.022 |
| 7 | Naive Bayes | 0.865 | 0.845 | 0.874 ± 0.027 |

**Best model: Random Forest**, with 97.5% mean CV accuracy, 95.9% held-out test accuracy and a weighted F1 of 0.96.

Why Random Forest:
- Works with ordinal and mixed-scale features without needing scaling, so deployment is simpler
- Robust to class imbalance and to correlated histopathological features
- Provides feature importances, which feed the doctor-facing recommendations
- Lowest CV standard deviation among the top performers, so performance on unseen patients is more reliable

SVM (RBF) is a close second and a good fallback.

## Key Findings

- Only `Age` had missing values (8 of 366), and there were no duplicate rows.
- Histopathological features carry the strongest signal for separating the diseases, ahead of purely clinical signs.
- Top features by Random Forest importance:

| Feature | Importance |
|---|---|
| Thinning of the suprapapillary epidermis | 0.090 |
| Clubbing of the rete ridges | 0.087 |
| Fibrosis of the papillary dermis | 0.084 |
| Elongation of the rete ridges | 0.066 |
| Koebner phenomenon | 0.061 |
| Vacuolisation and damage of basal layer | 0.043 |
| Spongiosis | 0.043 |
| Band-like infiltrate | 0.040 |

## Recommendations for Doctors

1. **Prioritise the histopathological work-up early.** Clinical signs are shared across all six diseases, so requesting a biopsy sooner shortens the diagnostic path.
2. **Use the model as a second-opinion triage tool, not a final diagnosis.** It can flag the most likely candidate diseases for a dermatopathologist to confirm, which helps especially for rarer classes like Pityriasis Rubra Pilaris.
3. **Flag low-confidence predictions for manual review.** Most residual errors sit between Seboreic Dermatitis and Pityriasis Rosea.
4. **Record family history consistently.** It is a simple field that contributes real signal.
5. **Track age alongside histopathology** and avoid missing values at intake.

> **Disclaimer:** This project is for educational and research purposes only. It is not a certified medical device and must not replace professional clinical diagnosis.

## Challenges and Solutions

| Challenge | Technique Used |
|---|---|
| Missing `Age` values (encoded as `?`) | Converted to `NaN` and imputed with the median, which is robust to age skew |
| Class imbalance (20 to 112 patients per class) | Stratified split and stratified 5-fold CV, with weighted precision/recall/F1 |
| High feature overlap between diseases | Relied on histopathological features and feature-importance analysis |
| Multicollinearity among histopathological features | Used tree-based ensembles as primary candidates |
| Small dataset (366 rows) | 5-fold cross-validation as the primary comparison metric |
| Different scaling needs across models | Kept scaled and raw feature sets and routed each model to the right one |

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Run the notebook

1. Place `skin_disorder.csv` in the project folder.
2. Start Jupyter:
   ```bash
   jupyter notebook Skin_Disorder_Project.ipynb
   ```
3. Run all cells from top to bottom.

## Repository Structure

```
.
├── Skin_Disorder_Project.ipynb   # Full analysis, modeling and report
├── skin_disorder.csv             # Dataset (add this file)
└── README.md
```

## Tech Stack

- **Language:** Python
- **Data handling:** NumPy, pandas
- **Visualisation:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes)
- **Environment:** Jupyter Notebook
