"""
analysis.py
-----------
Statistical analysis utilities:
  - Box-Cox power transformation
  - Shapiro-Wilk normality screening
  - Independent-samples t-tests (one group vs. all others)
  - One-way ANOVA with eta-squared effect size
  - Univariate logistic regression with bootstrap AUC confidence intervals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, f_oneway, shapiro, skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import PowerTransformer, label_binarize


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------

def boxcox_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a Box-Cox power transformation (with standardisation) to every
    column in *df*.  A per-column shift of ``min + 1`` is applied first to
    ensure all values are strictly positive, as required by Box-Cox.
    """
    df_shifted = df - df.min() + 1
    pt = PowerTransformer(method="box-cox", standardize=True)
    return pd.DataFrame(pt.fit_transform(df_shifted), columns=df.columns)


# ---------------------------------------------------------------------------
# Normality screening
# ---------------------------------------------------------------------------

def normality_report(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Run Shapiro-Wilk tests and compute skewness for each feature.
    Returns a DataFrame with columns: Feature, Shapiro_p, Skewness, Verdict.
    Also prints a formatted summary table to stdout.
    """
    header = f"{'Column':<15} | {'p-value':<10} | {'Skewness':<10} | Verdict"
    print(header)
    print("-" * 65)

    rows = []
    for col in features:
        data = df[col].dropna()
        _, p_val = shapiro(data)
        skew_val = skew(data)

        if p_val > 0.05:
            verdict = "Normal"
        elif abs(skew_val) < 0.5:
            verdict = "Near Normal (Safe)"
        else:
            verdict = "Non-Normal"

        print(f"{col:<15} | {p_val:.4f}     | {skew_val:.4f}     | {verdict}")
        rows.append(
            {"Feature": col, "Shapiro_p": p_val,
             "Skewness": round(skew_val, 4), "Verdict": verdict}
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairwise t-tests
# ---------------------------------------------------------------------------

def ttest_1v1(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    baseline: str,
) -> pd.DataFrame:
    """
    Independent-samples t-test (two-tailed, equal variance assumed) comparing
    *baseline* against every other group in *target_col* for each feature.

    Returns a DataFrame of p-values indexed by feature, with one column per
    comparison group.
    """
    results = {}
    baseline_data = df.loc[df[target_col] == baseline]
    comparison_groups = [
        g for g in df[target_col].unique()
        if g != baseline and pd.notna(g)
    ]

    for feature in features:
        results[feature] = {}
        for group in comparison_groups:
            group_data = df.loc[df[target_col] == group]
            if len(baseline_data) > 0 and len(group_data) > 0:
                _, p_val = ttest_ind(
                    baseline_data[feature].dropna(),
                    group_data[feature].dropna(),
                    equal_var=True,
                )
                results[feature][group] = p_val
            else:
                results[feature][group] = np.nan

    return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# ANOVA + eta-squared
# ---------------------------------------------------------------------------

def _eta_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Compute eta-squared (η²) effect size for a one-way ANOVA."""
    mean_grand = np.mean(x)
    sst = np.sum((x - mean_grand) ** 2)
    if sst == 0:
        return 0.0
    ssb = sum(
        len(x[y == c]) * (np.mean(x[y == c]) - mean_grand) ** 2
        for c in np.unique(y)
    )
    return ssb / sst


def compute_anova_pvalues(
    df: pd.DataFrame,
    features: list,
    target: str,
) -> pd.DataFrame:
    """
    One-way ANOVA F-test and eta-squared for each feature across all groups
    defined by *target*.

    Returns a DataFrame sorted by ascending p-value with columns:
    Feature, ANOVA_p_value, Eta_Squared.
    """
    y = df[target].values
    groups = df[target].unique()
    rows = []

    for feature in features:
        x = df[feature].values
        data_by_class = [df.loc[df[target] == g, feature] for g in groups]
        _, p_val = f_oneway(*data_by_class)
        rows.append(
            {
                "Feature": feature,
                "ANOVA_p_value": p_val,
                "Eta_Squared": round(_eta_squared(x, y), 3),
            }
        )

    return pd.DataFrame(rows).sort_values(by="ANOVA_p_value")


# ---------------------------------------------------------------------------
# Univariate AUC analysis
# ---------------------------------------------------------------------------

def _bootstrap_auc_ci(
    X: np.ndarray,
    y: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
) -> tuple:
    """
    Bootstrap 95 % confidence interval for the macro-averaged OvR AUC.

    Parameters
    ----------
    X, y       : feature matrix and true labels (arrays).
    y_prob     : predicted class probabilities from a fitted classifier.
    n_bootstraps : number of bootstrap resamples.
    confidence_level : desired coverage, default 0.95.

    Returns
    -------
    (ci_lower, ci_upper) : float tuple.
    """
    rng = np.random.RandomState(42)
    auc_scores = []

    for _ in range(n_bootstraps):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            auc_scores.append(
                roc_auc_score(y[idx], y_prob[idx], multi_class="ovr")
            )
        except Exception:
            continue

    alpha = 1 - confidence_level
    ci_lower = np.percentile(auc_scores, alpha / 2 * 100)
    ci_upper = np.percentile(auc_scores, (1 - alpha / 2) * 100)
    return ci_lower, ci_upper


def run_univariate_analysis(
    df: pd.DataFrame,
    features: list,
    target: str,
    figure_path: str = None,
) -> pd.DataFrame:
    """
    Fit a separate logistic regression model for each feature individually and
    report the macro-averaged OvR AUC with bootstrap 95 % CIs.

    Parameters
    ----------
    df           : DataFrame containing features and *target* column.
    features     : list of continuous feature names.
    target       : name of the outcome column.
    figure_path  : if provided, save the ROC curve figure to this path.

    Returns
    -------
    DataFrame with columns: Parameter, Macro_AUC, AUC_CI_Lower, AUC_CI_Upper,
    sorted by descending AUC.
    """
    y = df[target].values
    n_classes = len(np.unique(y))
    y_bin = label_binarize(y, classes=np.unique(y))

    plt.figure(figsize=(10, 8))
    rows = []

    for feature in features:
        X_feat = df[[feature]].values
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(X_feat, y)
        y_prob = clf.predict_proba(X_feat)

        macro_auc = roc_auc_score(y, y_prob, multi_class="ovr")
        ci_lower, ci_upper = _bootstrap_auc_ci(X_feat, y, y_prob)

        rows.append(
            {
                "Parameter": feature,
                "Macro_AUC": round(macro_auc, 3),
                "AUC_CI_Lower": round(ci_lower, 3),
                "AUC_CI_Upper": round(ci_upper, 3),
            }
        )

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
        else:
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())

        plt.plot(fpr, tpr, label=f"{feature} (AUC = {macro_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Univariate Multiclass ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if figure_path:
        plt.savefig(figure_path, dpi=256, bbox_inches="tight", transparent=True)
    plt.show()

    return pd.DataFrame(rows).sort_values(by="Macro_AUC", ascending=False)
