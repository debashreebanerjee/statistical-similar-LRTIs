# Differential Diagnosis of SARS-CoV-2 vs. Other Lower Respiratory Tract Infections Using Routine Laboratory Parameters

This repository contains the analysis code accompanying the publication.
All statistical analyses and machine learning models were implemented in Python.

---

## Overview

The pipeline discriminates between **COVID-19** from **other lower respiratory tract infections (LRTIs)**
and **non-infected controls** using routine haematological and biochemical blood parameters.

---

## Repository Structure

```
.
├── main.py                  # End-to-end pipeline entrypoint
├── requirements.txt
├── src/
│   ├── preprocessing.py     # Data loading, encoding, cleaning, label assignment
│   ├── analysis.py          # Normality screening, t-tests, ANOVA, univariate AUC
│   └── modeling.py          # Multivariate logistic regression, ROC curves, coefficients
├── outputs/                 # Generated CSV results, figures
```

---

## Methods Summary

### Statistical Testing
- **Normality**: Shapiro-Wilk test and skewness were assessed for each continuous feature prior to transformation.
- **Transformation**: Box-Cox power transformation (with standardisation) was applied to all continuous features to reduce skewness before parametric testing.
- **Pairwise comparisons**: Independent-samples t-tests (two-tailed, equal variance) were used for group-wise comparisons.
- **Effect size**: One-way ANOVA with eta-squared (η²) was computed for each feature across the three outcome groups.

### Univariate Analysis
Each haematological/biochemical feature was individually evaluated as a predictor using logistic regression. Macro-averaged one-vs-rest (OvR) AUC was reported with 95 % bootstrap confidence intervals (1,000 resamples, seed = 42).

### Multivariate Model
A multivariate logistic regression model was trained on a selected feature subset with standardised inputs and a stratified 70/30 train–test split (seed = 42). Model performance was assessed via classification report, confusion matrix, and per-class ROC curves.

---

## Data Availability

The raw dataset used in this study is not included in this repository.  
Please refer to the data availability statement in the published article for access instructions.

