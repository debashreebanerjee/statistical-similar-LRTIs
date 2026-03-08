"""
modeling.py
-----------
Multivariate logistic regression pipeline:
  - Standardisation and stratified train/test split
  - Model fitting and evaluation (classification report, confusion matrix)
  - Per-class ROC curves with macro-averaged AUC
  - Ranked feature coefficients per class
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize


def run_logreg(
    df: pd.DataFrame,
    features: list,
    target: str,
    test_size: float = 0.3,
    random_state: int = 42,
    figure_path: str = None,
) -> tuple:
    """
    Train a multivariate logistic regression classifier and evaluate it on a
    held-out test set.

    Parameters
    ----------
    df           : DataFrame containing feature columns and *target*.
    features     : list of feature column names to use as predictors.
    target       : name of the outcome column.
    test_size    : proportion of data reserved for testing (default 0.3).
    random_state : random seed for reproducibility (default 42).
    figure_path  : if provided, save the ROC curve figure to this path.

    Returns
    -------
    (model, scaler) : fitted LogisticRegression and StandardScaler instances.
    """
    X = df[features].values
    y = df[target].values
    classes = np.unique(y)
    n_classes = len(classes)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # --- Performance summary ---
    print("=" * 50)
    print("Model Performance Report")
    print("=" * 50)
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print("-" * 50)
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    print(df_cm, "\n")

    try:
        macro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    except ValueError:
        macro_auc = 0.5

    # --- ROC curves ---
    y_test_bin = label_binarize(y_test, classes=classes)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

    plt.figure(figsize=(10, 8))
    for i, (cls, color) in enumerate(zip(classes, colors)):
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f"Model (AUC = {macro_auc:.2f})")
            break
        else:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            class_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f"Class {cls} (AUC = {class_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.title(
        f"Multivariate Logistic Regression ROC\n"
        f"Macro-Averaged AUC = {macro_auc:.2f}"
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if figure_path:
        plt.savefig(figure_path, dpi=256, bbox_inches="tight", transparent=True)
    plt.show()

    return model, scaler


def print_feature_coefficients(model: LogisticRegression, features: list) -> None:
    """
    Print a ranked table of logistic regression coefficients for each class,
    sorted by absolute magnitude (most influential predictors first).

    Parameters
    ----------
    model    : fitted LogisticRegression instance.
    features : list of feature names corresponding to model.coef_ columns.
    """
    for i, class_label in enumerate(model.classes_):
        coef_df = (
            pd.DataFrame({"Feature": features, "Coefficient": model.coef_[i]})
            .sort_values(by="Coefficient", key=abs, ascending=False)
        )
        print(f"\nCoefficients for class '{class_label}':")
        print(coef_df.to_string(index=False))
