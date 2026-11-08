"""
shap_analysis.py
================
SHAP value computation using TreeExplainer for RF and XGBoost.

Handles all output formats across SHAP versions:
  - list of arrays  [old SHAP < 0.42, one array per class]
  - 3D ndarray      [new SHAP >= 0.42, shape (n_samples, n_features, n_classes)]
  - 2D ndarray      [regression or binary in some versions]
"""

import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")

import os
import sys
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils.config import SHAP_SAMPLE_N, SHAP_TOP_K


def compute_shap(model,
                 X_test: pd.DataFrame,
                 model_name: str = "rf",
                 sample_n: int = SHAP_SAMPLE_N) -> tuple:
    """
    Compute SHAP values for a fitted tree model on X_test.

    Handles three output formats from shap.TreeExplainer:
      1. list of ndarrays   -> old SHAP format (one array per class)
      2. 3D ndarray         -> new SHAP format (n_samples, n_features, n_classes)
      3. 2D ndarray         -> regression or single-output

    Returns:
        shap_values  : np.ndarray  shape (n_samples, n_features)  — always 2D
        importance   : pd.Series   mean |SHAP| per feature, sorted descending
        explainer    : shap.TreeExplainer
    """
    n = min(sample_n, len(X_test))
    X_sample = X_test.iloc[:n].copy()

    explainer = shap.TreeExplainer(model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = explainer.shap_values(X_sample)

    # ── Normalise to 2D (n_samples, n_features) ───────────────────────────────
    if isinstance(raw, list):
        # Old SHAP (<0.42): list with one array per class
        # Stack → (n_classes, n_samples, n_features), average absolute across classes
        shap_values = np.mean(
            np.abs(np.stack(raw, axis=0)), axis=0
        )

    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        # New SHAP (>=0.42) with tree classifiers:
        # shape = (n_samples, n_features, n_classes)
        # Average absolute SHAP across the class dimension
        shap_values = np.mean(np.abs(raw), axis=-1)

    elif isinstance(raw, np.ndarray) and raw.ndim == 2:
        # Regression or binary (already 2D)
        shap_values = raw

    else:
        # Fallback: force 2D
        shap_values = np.abs(np.array(raw))
        if shap_values.ndim != 2:
            shap_values = shap_values.reshape(len(X_sample), -1)

    # ── Assertions ────────────────────────────────────────────────────────────
    assert shap_values.ndim == 2, \
        f"SHAP normalisation failed: shape={shap_values.shape}"
    assert shap_values.shape == (len(X_sample), X_test.shape[1]), \
        (f"SHAP shape mismatch after normalisation: "
         f"got {shap_values.shape}, expected ({len(X_sample)}, {X_test.shape[1]})")
    assert not np.isnan(shap_values).any(), \
        "NaN values found in SHAP output"

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_test.columns,
        name=f"mean_abs_shap_{model_name}",
    ).sort_values(ascending=False)

    return shap_values, importance, explainer


def shap_for_models(rf_model, xgb_model,
                    X_test: pd.DataFrame,
                    sample_n: int = SHAP_SAMPLE_N) -> dict:
    """
    Compute SHAP for both RF and XGBoost in one call.
    Returns {"rf": (shap_values, importance, explainer),
             "xgb": (shap_values, importance, explainer)}
    """
    print("    Computing SHAP (RF)...")
    rf_shap,  rf_imp,  rf_exp  = compute_shap(rf_model,  X_test, "rf",  sample_n)
    print("    Computing SHAP (XGB)...")
    xgb_shap, xgb_imp, xgb_exp = compute_shap(xgb_model, X_test, "xgb", sample_n)

    return {
        "rf":  (rf_shap,  rf_imp,  rf_exp),
        "xgb": (xgb_shap, xgb_imp, xgb_exp),
    }
