"""
data_utils.py
=============
Data loading, preprocessing, and splitting for local .tsv datasets.
Reads from the folder structure under Implementation/ directly.
No internet access or pmlb download needed.

PMLB .tsv format: tab-separated, header row, last column = "target".
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Ensure Implementation/ is on sys.path so utils package resolves correctly
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils.config import DATASET_PATHS


# ─── LOAD ─────────────────────────────────────────────────────────────────────

def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a dataset from its local .tsv file.
    The name must be one of the keys in DATASET_PATHS (config.py).
    PMLB files are tab-separated with a 'target' column.
    """
    if name not in DATASET_PATHS:
        raise ValueError(
            f"Dataset '{name}' not in DATASET_PATHS. "
            f"Check utils/config.py DATASET_PATHS dict.\n"
            f"Available: {list(DATASET_PATHS.keys())}"
        )
    path = DATASET_PATHS[name]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Make sure the .tsv file is in the correct folder."
        )
    df = pd.read_csv(path, sep="\t")

    # Verify target column exists
    if "target" not in df.columns:
        # Some PMLB files use the last column as target without naming it
        df = df.rename(columns={df.columns[-1]: "target"})

    assert "target" in df.columns, \
        f"'target' column not found in {name}. Columns: {df.columns.tolist()}"

    # Drop unnamed index columns if any
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    return df.reset_index(drop=True)


# ─── INSPECT ──────────────────────────────────────────────────────────────────

def inspect_dataset(df: pd.DataFrame, name: str) -> dict:
    """
    Print and return basic dataset statistics.
    Detects binary, categorical, and continuous columns automatically.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    binary_cols      = [c for c in X.columns if X[c].nunique() == 2]
    categorical_cols = [c for c in X.columns
                        if c not in binary_cols and
                        (X[c].dtype in ["object", "category"] or
                         (X[c].dtype in [int, "int64", "int32"] and
                          X[c].nunique() <= 20))]
    continuous_cols  = [c for c in X.columns
                        if c not in binary_cols and c not in categorical_cols]

    info = {
        "name":        name,
        "n_samples":   len(df),
        "n_features":  X.shape[1],
        "binary_cols": binary_cols,
        "cat_cols":    categorical_cols,
        "cont_cols":   continuous_cols,
        "n_classes":   int(y.nunique()),
        "class_counts":y.value_counts().to_dict(),
        "missing":     int(df.isnull().sum().sum()),
    }

    print(f"\n{'='*60}")
    print(f"  Dataset : {name}")
    print(f"  Samples : {info['n_samples']}  |  Features: {info['n_features']}")
    print(f"  binary={len(binary_cols)}  cat={len(categorical_cols)}  cont={len(continuous_cols)}")
    print(f"  Classes : {info['n_classes']}  |  Missing: {info['missing']}")
    if info['missing'] > 0:
        print(f"  WARNING: dataset has {info['missing']} missing values — will be dropped")
    print(f"{'='*60}\n")
    return info


# ─── CLEAN ────────────────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values. Log if any dropped."""
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with missing values "
              f"({n_dropped/n_before*100:.1f}%)")
    return df


# ─── ENCODE ───────────────────────────────────────────────────────────────────

def encode_features(X_train: pd.DataFrame,
                    X_test:  pd.DataFrame,
                    cat_cols: list) -> tuple:
    """
    Ordinal-encode categorical+binary columns.
    Encoder is FITTED on X_train ONLY — then applied to X_test.
    Returns (X_train_enc, X_test_enc, encoder).
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()

    if cat_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=float,
        )
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols])
        X_te[cat_cols] = enc.transform(X_te[cat_cols])
    else:
        enc = None

    X_tr = X_tr.astype(float)
    X_te = X_te.astype(float)
    return X_tr, X_te, enc


# ─── SPLIT ────────────────────────────────────────────────────────────────────

def split_dataset(df: pd.DataFrame,
                  test_size: float,
                  task: str,
                  seed: int) -> tuple:
    """
    Stratified split for classification; random for regression.
    Includes data-leakage assertion (professor point 5).
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )

    # ── Professor point 5 assertions ──────────────────────────────────────────
    assert len(X_test)  == len(y_test),  "Test X/y length mismatch"
    assert len(X_train) == len(y_train), "Train X/y length mismatch"
    assert set(X_train.index).isdisjoint(set(X_test.index)), \
        "DATA LEAKAGE: train and test indices overlap!"

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


# ─── SUBSAMPLE (large datasets for GReaT / TabDDPM) ──────────────────────────

def impute_missing(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   cat_cols: list) -> tuple:
    """
    Leakage-safe missing-value imputation (fit on TRAIN only).

    Strategy: most-frequent value for categorical/binary columns, median for
    continuous columns — both computed on X_train and applied to X_train AND
    X_test. Datasets without missing values pass through untouched.

    Added for the medium regression phase: titanic carries 129 NaN cells
    (age / fare / sibsp / parch / country). Every augmentation method assumes
    complete data — SDV samples the missingness back, DiCE's internal
    GradientBoosting rejects NaN, SMOGN silently drops NaN-bearing columns,
    and _combine's no-NaN integrity gate fails on the original rows — so
    imputation must happen upstream, after the split (no test leakage) and
    before encoding.

    Returns
    -------
    X_train, X_test : imputed copies (originals untouched)
    report : dict  {column: {"fill": value, "n_train": int, "n_test": int}}
             empty dict when nothing was imputed
    """
    n_tr = int(X_train.isna().sum().sum())
    n_te = int(X_test.isna().sum().sum())
    if n_tr == 0 and n_te == 0:
        return X_train, X_test, {}

    X_train = X_train.copy()
    X_test  = X_test.copy()
    cat_set = set(cat_cols or [])
    report  = {}

    for col in X_train.columns:
        miss_tr = int(X_train[col].isna().sum())
        miss_te = int(X_test[col].isna().sum())
        if miss_tr == 0 and miss_te == 0:
            continue
        train_vals = X_train[col].dropna()
        if len(train_vals) == 0:
            # Pathological: column entirely NaN in train — fill with 0.
            fill = 0
        elif col in cat_set:
            fill = train_vals.mode().iloc[0]
        else:
            fill = float(train_vals.median())
        X_train[col] = X_train[col].fillna(fill)
        X_test[col]  = X_test[col].fillna(fill)
        report[col]  = {"fill": fill, "n_train": miss_tr, "n_test": miss_te}

    assert X_train.isna().sum().sum() == 0, "Imputation left NaN in X_train"
    assert X_test.isna().sum().sum()  == 0, "Imputation left NaN in X_test"
    return X_train, X_test, report


def stratified_subsample(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         n: int,
                         task: str,
                         seed: int) -> tuple:
    """Reduce training set to at most n rows."""
    if len(X_train) <= n:
        return X_train, y_train
    stratify = y_train if task == "classification" else None
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train,
        train_size=n,
        random_state=seed,
        stratify=stratify,
    )
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


# ─── SDV METADATA ─────────────────────────────────────────────────────────────

def build_sdv_metadata(train_df: pd.DataFrame, cat_cols: list):
    """Build SDV SingleTableMetadata for CTGAN and TVAE."""
    from sdv.metadata import SingleTableMetadata
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(train_df)
    for col in cat_cols:
        if col in train_df.columns:
            meta.update_column(col, sdtype="categorical")
    return meta


# ─── COLUMN INDICES ───────────────────────────────────────────────────────────

def get_cat_indices(X: pd.DataFrame, cat_cols: list) -> list:
    """
    Return the integer column positions of categorical columns in X.
    Used by SMOTE-NC which requires indices, not names.
    Only returns indices for columns that actually exist in X.
    """
    return [X.columns.get_loc(c) for c in cat_cols if c in X.columns]