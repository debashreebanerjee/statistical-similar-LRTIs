"""
main.py
-------
End-to-end analysis pipeline for the statistical differentiation of the similar lower respiratory tract infections (LRTIs) using routine
haematological and biochemical parameters.

Execution order
---------------
1.  Load and encode raw data                    (preprocessing)
2.  Build cleaned datasets                      (preprocessing)
3.  Assign three-class outcome labels           (preprocessing)
4.  Normality screening (Shapiro-Wilk + skew)   (analysis)
5.  Box-Cox transformation of continuous feats  (analysis)
6.  Independent-samples t-tests                 (analysis)
7.  One-way ANOVA + eta-squared                 (analysis)
8.  Univariate logistic regression / AUC        (analysis)
9.  Multivariate logistic regression            (modeling)
10. Feature coefficient ranking                 (modeling)
"""

import os
import pandas as pd

from src.preprocessing import (
    load_raw,
    build_full_dataset,
    build_subset_dataset,
    assign_outcome_labels,
    VIRAL_COLS_SUBSET,
)
from src.analysis import (
    normality_report,
    boxcox_transform,
    ttest_1v1,
    compute_anova_pvalues,
    run_univariate_analysis,
)
from src.modeling import run_logreg, print_feature_coefficients

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DATA_PATH = "data/raw.csv"
OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

CONTINUOUS_FEATURES = [
    "Hematocrit", "Hemoglobin", "Platelets", "MPV", "RBCs",
    "Lymphocytes", "MCHC", "Leukocytes", "Basophils", "MCH",
    "Eosinophils", "MCV", "Monocytes", "RDW", "Neutrophils",
    "Urea", "CRP", "Creatinine",
]

LOGREG_FEATURES = [
    "Leukocytes", "Platelets", "Eosinophils", "Monocytes",
    "RBCs", "Creatinine", "Hemoglobin", "Hematocrit",
]

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Load raw data
    print("Loading raw data...")
    raw = load_raw(RAW_DATA_PATH)

    # 2. Build datasets
    print("Building datasets...")
    proc_full = build_full_dataset(raw)
    proc_subset = build_subset_dataset(
        raw,
        output_path=os.path.join(OUTPUT_DIR, "processed-2.csv"),
    )

    # 3. Assign outcome labels (working from the full processed dataset)
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "processed.csv"))
    df = assign_outcome_labels(df)

    # 4. Normality screening
    print("\nNormality Screening (Shapiro-Wilk)\n" + "=" * 65)
    normality_report(df, CONTINUOUS_FEATURES)

    # 5. Box-Cox transformation
    print("\nApplying Box-Cox transformation...")
    df[CONTINUOUS_FEATURES] = boxcox_transform(df[CONTINUOUS_FEATURES])

    # 6. Pairwise t-tests
    print("\nRunning independent-samples t-tests...")
    Y_dummies = pd.get_dummies(df["Y"])
    ttest_results = ttest_1v1(df, CONTINUOUS_FEATURES, "Y", baseline="None")
    ttest_results.to_csv(os.path.join(OUTPUT_DIR, "T_Tests.csv"))

    # 7. ANOVA + eta-squared
    print("Running one-way ANOVA...")
    anova_results = compute_anova_pvalues(df, CONTINUOUS_FEATURES, "Y")
    anova_results.to_csv(os.path.join(OUTPUT_DIR, "ANOVA.csv"), index=False)

    # 8. Univariate AUC analysis
    print("\nRunning univariate logistic regression / AUC analysis...")
    univariate_results = run_univariate_analysis(
        df,
        CONTINUOUS_FEATURES,
        "Y",
        figure_path=os.path.join(FIGURE_DIR, "Univariate_ROC.png"),
    )
    univariate_results.to_csv(
        os.path.join(OUTPUT_DIR, "Univariate.csv"), index=False
    )

    # 9. Multivariate logistic regression
    print("\nRunning multivariate logistic regression...")
    model, scaler = run_logreg(
        df,
        LOGREG_FEATURES,
        "Y",
        figure_path=os.path.join(FIGURE_DIR, "LogReg_ROC.png"),
    )

    # 10. Feature coefficients
    print_feature_coefficients(model, LOGREG_FEATURES)
