"""
preprocessing.py
----------------
Handles raw data ingestion, encoding of categorical labels, column selection,
renaming, and threshold-based row filtering. Produces cleaned DataFrames
ready for downstream statistical analysis and modelling.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------

RENAME_MAP_FULL = {
    "Respiratory Syncytial Virus": "RSV",
    "Influenza A": "Inf-A",
    "Influenza B": "Inf-B",
    "Parainfluenza 1": "Parainfl-1",
    "CoronavirusNL63": "Covi-NL63",
    "Rhinovirus/Enterovirus": "Rhino-Entero",
    "Coronavirus HKU1": "Covi-HKU1",
    "Parainfluenza 3": "Parainfl-3",
    "Chlamydophila pneumoniae": "C.pneumoniae",
    "Parainfluenza 4": "Parainfl-4",
    "Coronavirus229E": "Covi-229E",
    "CoronavirusOC43": "Covi-OC43",
    "Inf A H1N1 2009": "Inf-A(H1N1-09)",
    "Bordetella pertussis": "B.pertussis",
    "Parainfluenza 2": "Parainfl-2",
    "SARS-Cov-2 exam result": "SARS-CoV-2",
    "Mean platelet volume_(MPV)": "MPV",
    "Red blood Cells": "RBCs",
    "Mean corpuscular hemoglobin concentration_(MCHC)": "MCHC",
    "Mean corpuscular hemoglobin (MCH)": "MCH",
    "Mean corpuscular volume (MCV)": "MCV",
    "Red blood cell distribution width (RDW)": "RDW",
    "Proteina C reativa mg/dL": "CRP",
}

RENAME_MAP_SUBSET = {
    "Respiratory Syncytial Virus": "RSV",
    "Influenza B": "Inf-B",
    "CoronavirusNL63": "Covi-NL63",
    "Rhinovirus/Enterovirus": "Rhino-Entero",
    "Inf A H1N1 2009": "Inf-A(H1N1-09)",
    "SARS-Cov-2 exam result": "SARS-CoV-2",
    "Mean platelet volume_(MPV)": "MPV",
    "Red blood Cells": "RBCs",
    "Mean corpuscular hemoglobin concentration_(MCHC)": "MCHC",
    "Mean corpuscular hemoglobin (MCH)": "MCH",
    "Mean corpuscular volume (MCV)": "MCV",
    "Red blood cell distribution width (RDW)": "RDW",
    "Proteina C reativa mg/dL": "CRP",
}

COLS_FULL = [
    "Respiratory Syncytial Virus", "Influenza A", "Influenza B",
    "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus",
    "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae",
    "Adenovirus", "Parainfluenza 4", "Coronavirus229E", "CoronavirusOC43",
    "Inf A H1N1 2009", "Bordetella pertussis", "Metapneumovirus",
    "Parainfluenza 2", "SARS-Cov-2 exam result",
    "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume_(MPV)",
    "Red blood Cells", "Lymphocytes",
    "Mean corpuscular hemoglobin concentration_(MCHC)", "Leukocytes",
    "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils",
    "Mean corpuscular volume (MCV)", "Monocytes",
    "Red blood cell distribution width (RDW)", "Neutrophils",
    "Urea", "Proteina C reativa mg/dL", "Creatinine",
]

COLS_SUBSET = [
    "Respiratory Syncytial Virus", "Influenza B", "CoronavirusNL63",
    "Rhinovirus/Enterovirus", "Inf A H1N1 2009", "SARS-Cov-2 exam result",
    "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume_(MPV)",
    "Red blood Cells", "Lymphocytes",
    "Mean corpuscular hemoglobin concentration_(MCHC)", "Leukocytes",
    "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils",
    "Mean corpuscular volume (MCV)", "Monocytes",
    "Red blood cell distribution width (RDW)", "Neutrophils",
    "Urea", "Proteina C reativa mg/dL", "Creatinine",
]

VIRAL_COLS_SUBSET = [
    "RSV", "Inf-B", "Covi-NL63", "Rhino-Entero", "Inf-A(H1N1-09)",
]

LABEL_ENCODING = {
    "positive": 1,
    "negative": 0,
    "not_done": None,
    "Não Realizado": None,
    "not_detected": 0,
    "detected": 1,
    "absent": 0,
    "present": 1,
}


def load_raw(path: str) -> pd.DataFrame:
    """Read raw CSV and encode categorical string labels to numeric values."""
    raw = pd.read_csv(path)
    for string_val, numeric_val in LABEL_ENCODING.items():
        raw = raw.replace([string_val], numeric_val)
    return raw


def build_full_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Select the full column set, rename columns, and drop rows where fewer
    than 25 % of columns contain valid data.
    """
    proc = raw[COLS_FULL].copy()
    proc.rename(columns=RENAME_MAP_FULL, inplace=True)
    threshold = len(proc.columns) * 0.25
    proc.dropna(thresh=threshold, inplace=True)
    proc.reset_index(drop=True, inplace=True)
    return proc


def build_subset_dataset(raw: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    Select the reduced column set, rename columns, and drop rows that are
    missing any of the six key viral result columns.
    Optionally write the result to *output_path*.
    """
    proc2 = raw[COLS_SUBSET].copy()
    proc2.rename(columns=RENAME_MAP_SUBSET, inplace=True)
    proc2.dropna(subset=["SARS-CoV-2"] + VIRAL_COLS_SUBSET, inplace=True)
    proc2.reset_index(drop=True, inplace=True)
    if output_path:
        proc2.to_csv(output_path, index=False)
    return proc2


def assign_outcome_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a three-class outcome column 'Y':
      - 'SARS-CoV-2'   : SARS-CoV-2 positive
      - 'Other-LRTIs'  : SARS-CoV-2 negative but positive for another respiratory pathogen
      - 'None'         : SARS-CoV-2 negative and negative for all other pathogens tested
    """
    other_viral_positive = df[VIRAL_COLS_SUBSET].sum(axis=1) > 0
    conditions = [
        df["SARS-CoV-2"] == 1,
        (df["SARS-CoV-2"] == 0) & other_viral_positive,
        (df["SARS-CoV-2"] == 0) & ~other_viral_positive,
    ]
    labels = ["SARS-CoV-2", "Other-LRTIs", "None"]
    df = df.copy()
    df["Y"] = pd.Series(
        np.select(conditions, labels, default="Unknown"), index=df.index
    )
    return df
