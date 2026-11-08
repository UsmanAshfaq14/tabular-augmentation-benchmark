"""
clf_medium.py
=============================================================================
Classification — Medium Datasets
Datasets  : flare (1,066) | cmc (1,473) | car (1,728) | churn (5,000)
Split     : 80 / 20  (stratified)
Seeds     : 5 (42, 123, 456, 789, 999)
Methods   : baseline | smote_nc | ctgan | tvae | tabddpm | great | dice

─────────────────────────────────────────────────────────────────────────────
IRIT JUPYTER  —  7-HOUR SESSION MANAGEMENT
─────────────────────────────────────────────────────────────────────────────
The server disconnects after 7 hours. Manage this with --dataset:

  Session A  (~3.3h with GPU):
      python clf_medium.py --dataset flare cmc

  Session B  (~2.2h with GPU):
      python clf_medium.py --dataset car

  Session C  (~5.6h with GPU):
      python clf_medium.py --dataset churn

  Or run all at once if time permits:
      python clf_medium.py

CHECKPOINT / RESUME:
  Every completed (dataset, seed, method) is saved immediately to disk.
  If the session disconnects, re-run the exact same command — already-
  finished runs are detected and skipped automatically. No work is lost.

SESSION SAFETY GUARD:
  The script tracks wall-clock time. At 6.5 h it finishes the current
  method and stops. Restart with the same --dataset to continue.

─────────────────────────────────────────────────────────────────────────────
GPU AUTO-DETECTION  (RTX8000 / L40S on IRIT)
─────────────────────────────────────────────────────────────────────────────
When CUDA is available:
  - XGBoost  : tree_method="hist" + device="cuda"   ~10x faster
  - GReaT    : HuggingFace Transformers auto-GPU     ~20x faster
  - RF n_jobs: set to 4 to match Jupyter CPU allocation
On CPU (Windows / no GPU):
  - GReaT excluded  (too slow without GPU)
  - XGBoost uses default CPU settings

─────────────────────────────────────────────────────────────────────────────
MULTICLASS DATASETS
─────────────────────────────────────────────────────────────────────────────
  cmc  → 3 classes  (contraceptive method choice)
  car  → 4 classes  (vehicle acceptability)
  XGBoost uses multi:softprob automatically.
  DiCE targets majority class from minority pool (not "opposite").
  SMOTE-NC uses sampling_strategy="auto" (balances all non-majority).

Run:
    python clf_medium.py
    python clf_medium.py --dataset flare cmc
=============================================================================
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Force UTF-8 output (avoids cp1252 on Windows) ─────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.config import (
    SEEDS, OUTPUT_ROOT, SHAP_TOP_K,
    GAN_EPOCHS, GREAT_EPOCHS, DIFFUSION_STEPS, SMOTE_K,
)
from utils.data_utils import (
    load_dataset, inspect_dataset, encode_features,
    split_dataset, build_sdv_metadata, get_cat_indices,
)
from utils.aug_methods import run_augmentation
from utils.evaluation import (
    clf_metrics, clf_report_str,
    distribution_metrics, kl_divergence, shap_stability,
)
from utils.shap_analysis import shap_for_models
from utils.plotting import (
    plot_feature_distributions, plot_correlation_heatmap,
    plot_correlation_diff, plot_shap_summary, plot_shap_bar,
    plot_shap_comparison, plot_performance_comparison,
    plot_ks_per_feature, plot_shap_stability_heatmap,
)
from utils.io_utils import (
    make_output_dirs, setup_logger, save_splits, save_augmented_data,
    save_metrics_dict, save_shap_values, save_text_report,
    save_distribution_metrics, save_experiment_config, banner,
)

# ═══════════════════════════════════════════════════════════════════════════════
# GPU DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "None"
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = "torch not installed"

# ═══════════════════════════════════════════════════════════════════════════════
# SIZE-SPECIFIC CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SIZE        = "medium"
TASK        = "classification"
TEST_SIZE   = 0.20                        # 80/20 split
ALL_DATASETS = ["flare", "cmc", "car", "churn"]

# GReaT auto-enabled on GPU (~4 min/seed), disabled on CPU (~90 min/seed)
# GReaT is excluded from the main pipeline — it runs as a separate script
# (great_medium.py) to avoid blocking the full pipeline for hours.
# Run: python great_medium.py --dataset flare  (after clf_medium.py finishes)
AUG_METHODS = ["smote_nc", "ctgan", "tvae", "tabddpm", "dice"]

CFG = {
    "size":            SIZE,
    "task":            TASK,
    "test_size":       TEST_SIZE,
    "datasets":        ALL_DATASETS,
    "seeds":           SEEDS,
    "methods":         AUG_METHODS,
    "gpu":             GPU_NAME,
    "gan_epochs":      GAN_EPOCHS[SIZE],        # 300
    "great_epochs":    GREAT_EPOCHS[SIZE],      # 75
    "diffusion_steps": DIFFUSION_STEPS[SIZE],   # 15,000
    "smote_k":         SMOTE_K[SIZE],           # 5
}

# ── Baseline model params (GPU-aware) ─────────────────────────────────────────
# n_jobs=10 matches the 10-CPU allocation on L40S/RTX8000 nodes
RF_PARAMS = {"n_estimators": 100, "random_state": 42, "n_jobs": 10}

if GPU_AVAILABLE:
    XGB_PARAMS = {
        "n_estimators": 100, "random_state": 42,
        "eval_metric": "logloss", "verbosity": 0,
        "tree_method": "hist",  # required for CUDA
        "device": "cuda",       # use RTX8000 / L40S
    }
else:
    XGB_PARAMS = {
        "n_estimators": 100, "random_state": 42,
        "eval_metric": "logloss", "verbosity": 0,
    }

# ── Estimated runtimes per dataset (5 seeds, all methods) ─────────────────────
ESTIMATED_HOURS = {
    "flare": 1.4 if GPU_AVAILABLE else 2.0,
    "cmc":   1.9 if GPU_AVAILABLE else 2.8,
    "car":   2.2 if GPU_AVAILABLE else 3.3,
    "churn": 5.6 if GPU_AVAILABLE else 9.5,
}

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION TIME GUARD  (7-hour IRIT limit)
# ═══════════════════════════════════════════════════════════════════════════════

SESSION_START       = time.time()
SESSION_LIMIT_HOURS = 6.5   # stop 30 min before 7-hour disconnect

def session_elapsed_h():
    return (time.time() - SESSION_START) / 3600.0

def session_ok(logger=None):
    """Returns True if budget remains. Warns at <30 min."""
    elapsed   = session_elapsed_h()
    remaining = SESSION_LIMIT_HOURS - elapsed
    if remaining <= 0:
        msg = (f"SESSION LIMIT REACHED ({elapsed:.1f}h). "
               f"Stopping gracefully — re-run to continue from checkpoint.")
        (logger.warning if logger else print)(msg)
        return False
    if remaining < 0.5:
        msg = f"Session budget low: {remaining*60:.0f} min remaining."
        (logger.warning if logger else print)(msg)
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT / RESUME
# ═══════════════════════════════════════════════════════════════════════════════

def is_completed(dataset, method, seed, model="rf"):
    """True if this (dataset, method, seed) already has saved metrics."""
    path = os.path.join(
        OUTPUT_ROOT, TASK, SIZE, dataset, method,
        f"metrics_{model}_seed{seed}.csv"
    )
    return os.path.exists(path)

def count_completed(dataset):
    """Return (completed, total) for a dataset."""
    total     = (len(AUG_METHODS) + 1) * len(SEEDS)  # +1 baseline
    completed = sum(
        1 for m in ["baseline"] + AUG_METHODS
        for s in SEEDS if is_completed(dataset, m, s)
    )
    return completed, total

def load_saved(dataset, method, seed, rf_list, xgb_list):
    """Load already-saved per-seed metrics into accumulators."""
    for model, store in [("rf", rf_list), ("xgb", xgb_list)]:
        f = os.path.join(
            OUTPUT_ROOT, TASK, SIZE, dataset, method,
            f"metrics_{model}_seed{seed}.csv"
        )
        if os.path.exists(f):
            store.append(pd.read_csv(f).iloc[0].to_dict())

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — n_generate for classification (binary + multiclass)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_n_generate_clf(y_train):
    """
    Rows to add to balance all non-majority classes to the majority count.
    Binary:    n_majority - n_minority
    Multiclass: sum over all deficit classes
    """
    counts = y_train.value_counts()
    n_maj  = int(counts.max())
    return max(10, sum(max(0, n_maj - int(c)) for c in counts.values))

# ═══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE — one dataset × one seed × one method
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_method(method,
                   X_train_enc, X_test_enc,
                   y_train, y_test,
                   X_train_raw, cat_cols, cont_cols,
                   metadata, paths, dataset, seed, logger):
    """
    Augment → train → evaluate → SHAP → save everything.
    Returns (m_rf, m_xgb, dist_result, shap_out)
    or (None, None, None, None) on failure.
    """
    method_dir = paths[method]
    n_cls      = y_train.nunique()

    # ── Step 1: generate augmented data ───────────────────────────────────────
    t0 = time.time()
    if method == "baseline":
        X_aug_enc = X_train_enc.copy()
        y_aug     = y_train.copy()
    else:
        n_gen = compute_n_generate_clf(y_train)
        X_aug_enc, y_aug, X_syn, y_syn = run_augmentation(
            method, X_train_enc, y_train,
            cat_cols, cont_cols, metadata,
            n_generate=n_gen, cfg=CFG, task=TASK,
        )
        if X_aug_enc is None:
            logger.warning(f"  [{method}] Augmentation failed - skipping.")
            return None, None, None, None

        # Soft integrity checks
        if len(X_aug_enc) <= len(X_train_enc):
            logger.warning(f"  [{method}] Added 0 rows - skipping.")
            return None, None, None, None
        if X_aug_enc.isna().sum().sum() > 0:
            logger.warning(f"  [{method}] NaN in augmented set - skipping.")
            return None, None, None, None
        if len(X_aug_enc) != len(y_aug):
            logger.warning(f"  [{method}] X/y length mismatch - skipping.")
            return None, None, None, None

        save_augmented_data(X_aug_enc, y_aug, X_syn, y_syn, method_dir, seed)
        logger.info(f"  [{method}] Generated {len(X_syn)} rows in {time.time()-t0:.1f}s")

    # ── Step 2: coerce dtypes + train ────────────────────────────────────────
    # Safety net: CTGAN/TVAE may produce string values for numeric columns
    # (SDV reverse-maps ordinal integers to original labels). Coerce here too.
    for _col in X_aug_enc.columns:
        if X_train_enc[_col].dtype != object and X_aug_enc[_col].dtype == object:
            _fallback = int(X_train_enc[_col].mode()[0])
            X_aug_enc[_col] = (pd.to_numeric(X_aug_enc[_col], errors="coerce")
                               .fillna(_fallback)
                               .astype(X_train_enc[_col].dtype))
            logger.warning(f"  [{method}] Coerced string→numeric in column '{_col}'")

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_aug_enc, y_aug)

    xgb_params = XGB_PARAMS.copy()
    if n_cls > 2:
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"]  = int(n_cls)
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_aug_enc, y_aug)

    # ── Step 3: evaluate on UNTOUCHED test set ────────────────────────────────
    rf_pred  = rf.predict(X_test_enc)
    xgb_pred = xgb.predict(X_test_enc)
    rf_prob  = rf.predict_proba(X_test_enc)
    xgb_prob = xgb.predict_proba(X_test_enc)

    m_rf  = clf_metrics(y_test, rf_pred,  rf_prob,  "rf",  method)
    m_xgb = clf_metrics(y_test, xgb_pred, xgb_prob, "xgb", method)
    m_rf["seed"]    = seed;  m_rf["dataset"]  = dataset
    m_xgb["seed"]   = seed;  m_xgb["dataset"] = dataset

    save_metrics_dict(m_rf,
        os.path.join(method_dir, f"metrics_rf_seed{seed}.csv"))
    save_metrics_dict(m_xgb,
        os.path.join(method_dir, f"metrics_xgb_seed{seed}.csv"))
    save_text_report(clf_report_str(y_test, rf_pred),
        os.path.join(method_dir, f"classification_report_rf_seed{seed}.txt"))
    save_text_report(clf_report_str(y_test, xgb_pred),
        os.path.join(method_dir, f"classification_report_xgb_seed{seed}.txt"))

    logger.info(f"  [{method}] RF  F1={m_rf['f1_macro']:.4f}  AUC={m_rf['auc_roc']}")
    logger.info(f"  [{method}] XGB F1={m_xgb['f1_macro']:.4f}  AUC={m_xgb['auc_roc']}")

    # ── Step 4: distribution metrics (skip for baseline) ──────────────────────
    dist_result = None
    if method != "baseline":
        dist_m  = distribution_metrics(X_train_enc, X_aug_enc, method)
        kl_dict = kl_divergence(X_train_enc, X_aug_enc)
        dist_m["seed"]    = seed
        dist_m["dataset"] = dataset
        dist_result = {**dist_m, **{f"kl_{k}": v for k, v in kl_dict.items()}}
        save_distribution_metrics(dist_m, kl_dict, method_dir, seed)

        plot_dir = os.path.join(method_dir, "distribution_plots")
        plot_feature_distributions(X_train_enc, X_aug_enc, method, plot_dir)
        plot_correlation_heatmap(X_train_enc,
            f"Correlation - Original - {dataset}",
            os.path.join(method_dir, f"corr_original_seed{seed}.png"))
        plot_correlation_heatmap(X_aug_enc,
            f"Correlation - Augmented ({method}) - {dataset}",
            os.path.join(method_dir, f"corr_augmented_seed{seed}.png"))
        plot_correlation_diff(X_train_enc, X_aug_enc, method,
            os.path.join(method_dir, f"corr_diff_seed{seed}.png"))
        plot_ks_per_feature(dist_m, method, dataset,
            os.path.join(method_dir, f"ks_per_feature_seed{seed}.png"))

    # ── Step 5: SHAP ──────────────────────────────────────────────────────────
    shap_res = shap_for_models(rf, xgb, X_test_enc)
    shap_out = {}
    for mdl_name, (sv, imp, _) in shap_res.items():
        save_shap_values(sv, imp, method_dir, mdl_name, seed)
        plot_shap_bar(imp, mdl_name, method,
            os.path.join(method_dir, f"shap_bar_{mdl_name}_seed{seed}.png"))
        plot_shap_summary(sv, X_test_enc, mdl_name, method,
            os.path.join(method_dir, f"shap_summary_{mdl_name}_seed{seed}.png"))
        shap_out[mdl_name] = imp

    return m_rf, m_xgb, dist_result, shap_out

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET PIPELINE — all seeds × all methods
# ═══════════════════════════════════════════════════════════════════════════════

def run_dataset(dataset, logger):
    """Full pipeline for one dataset across all seeds and methods."""

    completed, total = count_completed(dataset)
    if completed == total:
        logger.info(f"  {dataset}: ALL {total} runs already complete. Skipping.")
        return

    banner(f"Dataset: {dataset.upper()}  |  {completed}/{total} complete  "
           f"|  ~{ESTIMATED_HOURS[dataset]:.1f}h remaining")

    try:
        df = load_dataset(dataset)
    except Exception as e:
        import traceback
        print(f"\n  !! load_dataset('{dataset}') FAILED: {e}")
        traceback.print_exc()
        raise   # re-raise so main() catches it too

    info = inspect_dataset(df, dataset)

    cat_cols  = info["cat_cols"] + info["binary_cols"]
    cont_cols = info["cont_cols"]
    n_classes = info["n_classes"]

    if n_classes > 2:
        logger.info(f"  Multiclass: {n_classes} classes — "
                    f"XGB uses multi:softprob, DiCE targets majority class")

    paths = make_output_dirs(OUTPUT_ROOT, TASK, SIZE, dataset, AUG_METHODS)
    save_experiment_config(
        {**CFG, "dataset": dataset, "n_classes": n_classes,
         "info": {k: v for k, v in info.items() if k != "class_counts"}},
        os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset))

    all_metrics_rf   = []
    all_metrics_xgb  = []
    all_dist         = []
    all_shap_rf      = []
    all_shap_xgb     = []

    for seed in SEEDS:

        # ── Session budget check ─────────────────────────────────────────────
        if not session_ok(logger):
            logger.warning(f"  Stopping at seed {seed}. Re-run to resume from checkpoint.")
            break

        # ── Seed-level checkpoint ─────────────────────────────────────────────
        seed_all_done = all(
            is_completed(dataset, m, seed)
            for m in ["baseline"] + AUG_METHODS
        )
        if seed_all_done:
            logger.info(f"  Seed {seed}: fully checkpointed — loading saved metrics.")
            for m in ["baseline"] + AUG_METHODS:
                load_saved(dataset, m, seed, all_metrics_rf, all_metrics_xgb)
            continue

        banner(f"  Seed {seed}", char="-", width=50)

        # Split
        X_train_raw, X_test_raw, y_train, y_test = split_dataset(
            df, test_size=TEST_SIZE, task=TASK, seed=seed)

        # ── Normalise class labels to 0-indexed ───────────────────────────────
        # XGBoost 3.x multi:softprob requires {0,1,...,n-1}.
        # Datasets like cmc use {1,2,3} natively — fix both train and test.
        _orig_cls = sorted(y_train.unique())
        if _orig_cls[0] != 0:
            _lmap  = {c: i for i, c in enumerate(_orig_cls)}
            y_train = y_train.map(_lmap).astype(int)
            y_test  = y_test.map(_lmap).astype(int)
            logger.info(f"  Labels normalised: {_orig_cls} -> 0-indexed")
        # ─────────────────────────────────────────────────────────────────────

        # Hash the test set — used to verify it's never modified
        test_hash = pd.util.hash_pandas_object(X_test_raw).sum()

        # Encode (fit on train only)
        X_train_enc, X_test_enc, encoder = encode_features(
            X_train_raw, X_test_raw, cat_cols)

        # Build SDV metadata (CTGAN / TVAE)
        train_df_meta = pd.concat(
            [X_train_enc, y_train.rename("target")], axis=1)
        metadata = build_sdv_metadata(train_df_meta, cat_cols)

        # Save splits
        save_splits(X_train_enc, X_test_enc, y_train, y_test,
                    paths["data"], seed)

        baseline_shap_rf  = None
        baseline_shap_xgb = None

        for method in ["baseline"] + AUG_METHODS:

            # ── Method-level checkpoint ──────────────────────────────────────
            if is_completed(dataset, method, seed):
                logger.info(f"  [{method}] seed={seed}: checkpointed — skipping.")
                load_saved(dataset, method, seed, all_metrics_rf, all_metrics_xgb)
                continue

            # ── Session budget check before each method ───────────────────────
            if not session_ok(logger):
                logger.warning(
                    f"  Session limit before [{method}] seed={seed}. "
                    f"Re-run to resume.")
                break

            print(f"\n  -> [{dataset}] seed={seed}  method={method}  "
                  f"({session_elapsed_h():.1f}h elapsed)")

            try:
                m_rf, m_xgb, dist_m, shap_d = run_one_method(
                    method, X_train_enc, X_test_enc, y_train, y_test,
                    X_train_raw, cat_cols, cont_cols, metadata,
                    paths, dataset, seed, logger,
                )
            except Exception as e:
                logger.error(f"  [{method}] Unexpected error: {e}", exc_info=True)
                continue

            if m_rf is None:
                continue

            # Professor point 5: test set unchanged
            assert pd.util.hash_pandas_object(X_test_raw).sum() == test_hash, \
                f"TEST SET MODIFIED after {method}!"

            all_metrics_rf.append(m_rf)
            all_metrics_xgb.append(m_xgb)
            if dist_m:
                all_dist.append(dist_m)

            if method == "baseline" and shap_d:
                baseline_shap_rf  = shap_d.get("rf")
                baseline_shap_xgb = shap_d.get("xgb")

            if method != "baseline" and shap_d and baseline_shap_rf is not None:
                for mdl_name, base_imp, aug_imp in [
                    ("rf",  baseline_shap_rf,  shap_d.get("rf")),
                    ("xgb", baseline_shap_xgb, shap_d.get("xgb")),
                ]:
                    if aug_imp is None:
                        continue
                    stab = shap_stability(
                        base_imp, aug_imp, top_k=SHAP_TOP_K, method=method)
                    stab.update({"seed": seed, "dataset": dataset, "model": mdl_name})
                    (all_shap_rf if mdl_name == "rf" else all_shap_xgb).append(stab)

                    plot_shap_comparison(base_imp, aug_imp, mdl_name, method,
                        os.path.join(paths[method],
                                     f"shap_comparison_{mdl_name}_seed{seed}.png"))

        done_now, _ = count_completed(dataset)
        logger.info(
            f"  Seed {seed} complete. "
            f"Dataset progress: {done_now}/{total}. "
            f"Session: {session_elapsed_h():.1f}h elapsed."
        )

    # ── Aggregate and save cross-seed summaries ────────────────────────────────
    cmp_dir = paths["comparison"]
    for records, fname in [
        (all_metrics_rf,  "all_methods_metrics_rf.csv"),
        (all_metrics_xgb, "all_methods_metrics_xgb.csv"),
        (all_dist,        "all_methods_distribution.csv"),
        (all_shap_rf,     "all_methods_shap_stability_rf.csv"),
        (all_shap_xgb,    "all_methods_shap_stability_xgb.csv"),
    ]:
        if records:
            pd.DataFrame(records).to_csv(
                os.path.join(cmp_dir, fname), index=False)

    if all_metrics_rf:
        for metric in ["f1_macro", "auc_roc", "mcc"]:
            for mdl, recs in [("rf", all_metrics_rf), ("xgb", all_metrics_xgb)]:
                df_m = pd.DataFrame(recs)
                avg_recs = []
                for mth in ["baseline"] + AUG_METHODS:
                    sub = df_m[df_m["method"] == mth]
                    if len(sub):
                        row = sub[["model", "method"]].iloc[0].to_dict()
                        row[metric] = sub[metric].mean()
                        avg_recs.append(row)
                if avg_recs:
                    plot_performance_comparison(
                        avg_recs, metric, TASK, dataset, mdl,
                        os.path.join(cmp_dir, f"perf_{metric}_{mdl}.png"))

    if all_shap_rf:
        plot_shap_stability_heatmap(all_shap_rf, dataset, "rf",
            os.path.join(cmp_dir, "shap_stability_heatmap_rf.png"))
        plot_shap_stability_heatmap(all_shap_xgb, dataset, "xgb",
            os.path.join(cmp_dir, "shap_stability_heatmap_xgb.png"))

    final_done, _ = count_completed(dataset)
    logger.info(f"  {dataset} complete. {final_done}/{total} runs. "
                f"Outputs -> {os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset)}")

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="clf_medium.py — Classification Medium Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Session plan for IRIT 7-hour limit (GPU timings):
  Session A:  python clf_medium.py --dataset flare cmc    (~3.3h)
  Session B:  python clf_medium.py --dataset car          (~2.2h)
  Session C:  python clf_medium.py --dataset churn        (~5.6h)

  Checkpoints are saved after every method. If session disconnects,
  re-run the same command to resume automatically.
        """
    )
    parser.add_argument(
        "--dataset", nargs="+",
        choices=ALL_DATASETS, default=ALL_DATASETS,
        help="Datasets to run this session (default: all 4)"
    )
    args = parser.parse_args()

    log_path = os.path.join(OUTPUT_ROOT, TASK, SIZE, "run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path, name=f"{TASK}_{SIZE}")

    banner(f"CLASSIFICATION - MEDIUM DATASETS  |  Seeds: {SEEDS}")

    # GPU status
    if GPU_AVAILABLE:
        logger.info(f"GPU DETECTED: {GPU_NAME}")
        logger.info(f"XGBoost: tree_method=hist + device=cuda")
        logger.info(f"GReaT: ENABLED")
    else:
        logger.info("GPU: not available — CPU mode (GReaT disabled)")

    logger.info(f"Datasets this session : {args.dataset}")
    logger.info(f"Methods               : {AUG_METHODS}")
    logger.info(f"Split                 : 80/20")
    logger.info(f"GAN epochs            : {CFG['gan_epochs']}")
    logger.info(f"TabDDPM steps         : {CFG['diffusion_steps']}")
    logger.info(f"Session limit         : {SESSION_LIMIT_HOURS}h")
    logger.info("")

    # Checkpoint summary
    logger.info("Checkpoint status:")
    for ds in args.dataset:
        done, total = count_completed(ds)
        pct = int(done / total * 100) if total else 0
        logger.info(
            f"  {ds:<12}: {done:>3}/{total} complete ({pct:>3}%)  "
            f"~{ESTIMATED_HOURS[ds]:.1f}h remaining"
        )
    logger.info("")

    # ── Startup diagnostics ──────────────────────────────────────────────────
    logger.info("Running startup diagnostics...")
    from utils.config import DATASET_PATHS, OUTPUT_ROOT as _OR
    diag_ok = True
    for ds in args.dataset:
        p = DATASET_PATHS.get(ds, "NOT IN CONFIG")
        exists = os.path.isfile(p) if p != "NOT IN CONFIG" else False
        status = "OK" if exists else "MISSING"
        logger.info(f"  {ds:<12}: [{status}] {p}")
        if not exists:
            print(f"  !! FILE MISSING: {p}")
            diag_ok = False
    out_ok = os.access(os.path.dirname(_OR), os.W_OK) or True
    logger.info(f"  Output root: {_OR}")
    if not diag_ok:
        print("\n  !! One or more dataset files are missing. Check paths in utils/config.py")
        print(f"  !! Expected base dir: {os.path.dirname(_OR)}")
    logger.info("")
    # ─────────────────────────────────────────────────────────────────────────

    save_experiment_config(CFG, os.path.join(OUTPUT_ROOT, TASK, SIZE))

    for dataset in args.dataset:
        if not session_ok(logger):
            logger.warning(
                f"Session budget exhausted before {dataset}. "
                f"Re-run: python clf_medium.py --dataset {dataset}")
            break
        try:
            run_dataset(dataset, logger)
        except Exception as e:
            import traceback
            logger.error(f"Dataset {dataset} failed: {e}", exc_info=True)
            print(f"\n  !! ERROR in dataset {dataset}: {e}")
            traceback.print_exc()   # always visible in Jupyter cell output

    banner("CLASSIFICATION MEDIUM - SESSION COMPLETE")
    logger.info(f"Session ended after {session_elapsed_h():.2f}h")
    logger.info(f"Re-run with same command to resume any incomplete datasets.")
    logger.info(f"Outputs -> {os.path.join(OUTPUT_ROOT, TASK, SIZE)}")


if __name__ == "__main__":
    main()