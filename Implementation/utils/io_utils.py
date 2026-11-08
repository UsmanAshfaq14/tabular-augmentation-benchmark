import sys
"""
io_utils.py
─────────────────────────────────────────────────────────────────────
All I/O: folder creation, CSV logging, numpy saving, text reports.
Professor point 4: every single artifact must be saved.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime


# ─── FOLDER CREATION ──────────────────────────────────────────────────────────

def make_output_dirs(root: str, task: str, size: str,
                     dataset: str, methods: list) -> dict:
    """
    Create the complete output folder structure for one dataset.
    Returns a dict of paths keyed by method name.
    """
    paths = {}
    base  = os.path.join(root, task, size, dataset)

    for method in ["baseline"] + methods:
        p = os.path.join(base, method)
        os.makedirs(p, exist_ok=True)
        paths[method] = p

    # Distribution plots subfolder per method
    for method in methods:
        dp = os.path.join(base, method, "distribution_plots")
        os.makedirs(dp, exist_ok=True)

    # Comparison folder
    cmp_path = os.path.join(base, "comparison")
    os.makedirs(cmp_path, exist_ok=True)
    paths["comparison"] = cmp_path

    # Data folder
    data_path = os.path.join(base, "data")
    os.makedirs(data_path, exist_ok=True)
    paths["data"] = data_path

    return paths


# ─── LOGGING SETUP ────────────────────────────────────────────────────────────

def setup_logger(log_path: str, name: str = "experiment") -> logging.Logger:
    """
    Create a logger that writes to both console and a log file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler — force UTF-8 safe output on Windows
    ch = logging.StreamHandler(stream=open(
        sys.stdout.fileno(),
        mode="w",
        encoding="utf-8",
        errors="replace",
        closefd=False,
    ))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ─── SAVE DATA SPLITS ─────────────────────────────────────────────────────────

def save_splits(X_train, X_test, y_train, y_test,
                data_dir: str, seed: int) -> None:
    """Save train/test splits to CSV for reproducibility."""
    X_train.to_csv(os.path.join(data_dir, f"X_train_seed{seed}.csv"), index=False)
    X_test.to_csv( os.path.join(data_dir, f"X_test_seed{seed}.csv"),  index=False)
    y_train.to_csv(os.path.join(data_dir, f"y_train_seed{seed}.csv"), index=False, header=True)
    y_test.to_csv( os.path.join(data_dir, f"y_test_seed{seed}.csv"),  index=False, header=True)


# ─── SAVE AUGMENTED DATA ──────────────────────────────────────────────────────

def save_augmented_data(X_aug, y_aug, X_syn, y_syn,
                         method_dir: str, seed: int) -> None:
    """
    Save combined augmented set and synthetic-only rows.
    Professor point 4: every generated sample must be saved.
    """
    if X_aug is not None:
        aug_df = X_aug.copy()
        aug_df["target"] = y_aug.values
        aug_df.to_csv(
            os.path.join(method_dir, f"X_train_aug_seed{seed}.csv"), index=False
        )

    if X_syn is not None:
        syn_df = X_syn.copy()
        syn_df["target"] = y_syn.values
        syn_df.to_csv(
            os.path.join(method_dir, f"X_synthetic_only_seed{seed}.csv"), index=False
        )


# ─── SAVE METRICS ─────────────────────────────────────────────────────────────

def append_metrics_csv(record: dict, csv_path: str) -> None:
    """
    Append one result row to a CSV file.
    Creates the file with headers on first call.
    """
    df_new = pd.DataFrame([record])
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode="w", header=True, index=False)


def save_metrics_dict(metrics: dict, path: str) -> None:
    """Save a metrics dictionary as a single-row CSV."""
    pd.DataFrame([metrics]).to_csv(path, index=False)


# ─── SAVE SHAP VALUES ─────────────────────────────────────────────────────────

def save_shap_values(shap_values: np.ndarray,
                      importance: pd.Series,
                      method_dir: str,
                      model_name: str,
                      seed: int) -> None:
    """
    Save raw SHAP array as .npy and importance ranking as CSV.
    Professor point 4: all SHAP values must be saved.
    """
    np.save(
        os.path.join(method_dir, f"shap_values_{model_name}_seed{seed}.npy"),
        shap_values,
    )
    importance.to_csv(
        os.path.join(method_dir, f"shap_importance_{model_name}_seed{seed}.csv"),
        header=True,
    )


# ─── SAVE TEXT REPORT ─────────────────────────────────────────────────────────

def save_text_report(text: str, path: str) -> None:
    """Save a plain-text classification or regression report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


# ─── SAVE DISTRIBUTION METRICS ────────────────────────────────────────────────

def save_distribution_metrics(dist_metrics: dict,
                               kl_per_feat: dict,
                               method_dir: str,
                               seed: int) -> None:
    """Save distribution comparison metrics and per-feature KL divergences."""
    # Combined dict
    combined = {**dist_metrics, **{f"kl_{k}": v for k, v in kl_per_feat.items()}}
    pd.DataFrame([combined]).to_csv(
        os.path.join(method_dir, f"distribution_metrics_seed{seed}.csv"),
        index=False,
    )


# ─── AGGREGATE RESULTS ACROSS SEEDS ──────────────────────────────────────────

def aggregate_seeds(csv_glob_pattern: str,
                    group_cols: list,
                    metric_cols: list) -> pd.DataFrame:
    """
    Read all per-seed CSVs matching a pattern and compute mean ± std.
    Returns a summary DataFrame.
    """
    import glob
    files = glob.glob(csv_glob_pattern)
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


# ─── SAVE EXPERIMENT CONFIG ───────────────────────────────────────────────────

def save_experiment_config(cfg: dict, out_dir: str) -> None:
    """Save the full experiment configuration as JSON for reproducibility."""
    path = os.path.join(out_dir, "experiment_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)


# ─── PRINT SECTION BANNER ─────────────────────────────────────────────────────

def banner(text: str, char: str = "=", width: int = 60) -> None:
    line = char * width
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")
