"""
clf_small.py
============
Classification - Small Datasets
Datasets  : heart_c | saheart | australian | german
Split     : 70 / 30  (stratified, professor point 1)
Seeds     : 5 (42, 123, 456, 789, 999)
Methods   : baseline | smote_nc | ctgan | tvae | tabddpm | great | dice

Run:
    python clf_small.py

All outputs -> outputs/classification/small/<dataset>/<method>/
Professor point 4: every metric, SHAP value, and graph is saved.
Professor point 5: assertions check data integrity at every step.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Make sure Implementation/ is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.config import (
    SEEDS, OUTPUT_ROOT, METHODS, SHAP_TOP_K,
    GAN_EPOCHS, GREAT_EPOCHS, DIFFUSION_STEPS, SMOTE_K,
    LARGE_SUBSAMPLE_N, REG_AUG_PCT,
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
    append_metrics_csv, save_metrics_dict, save_shap_values,
    save_text_report, save_distribution_metrics,
    save_experiment_config, banner,
)

# --------------------------------------------------------------------------- #
#  SIZE-SPECIFIC CONFIG                                                         #
# --------------------------------------------------------------------------- #
SIZE        = "small"
TASK        = "classification"
TEST_SIZE   = 0.30                  # 70/30 split (professor point 1)
DATASETS    = ["heart_c", "saheart", "australian", "german"]
AUG_METHODS = ["smote_nc", "ctgan", "tvae", "tabddpm", "great", "dice"]

CFG = {
    "size":            SIZE,
    "task":            TASK,
    "test_size":       TEST_SIZE,
    "datasets":        DATASETS,
    "seeds":           SEEDS,
    "methods":         AUG_METHODS,
    "gan_epochs":      GAN_EPOCHS[SIZE],
    "great_epochs":    GREAT_EPOCHS[SIZE],
    "diffusion_steps": DIFFUSION_STEPS[SIZE],
    "smote_k":         SMOTE_K[SIZE],
}

RF_PARAMS  = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
XGB_PARAMS = {"n_estimators": 100, "random_state": 42,
              "eval_metric": "logloss", "verbosity": 0}


# --------------------------------------------------------------------------- #
#  HELPER - compute how many synthetic rows to generate (classification)       #
# --------------------------------------------------------------------------- #

def compute_n_generate_clf(y_train):
    """Double the minority class: n_generate = n_majority - n_minority."""
    counts     = y_train.value_counts()
    n_maj      = int(counts.max())
    n_min      = int(counts.min())
    n_generate = max(10, n_maj - n_min)
    return n_generate


# --------------------------------------------------------------------------- #
#  CORE PIPELINE - one dataset x one seed x one method                        #
# --------------------------------------------------------------------------- #

def run_one_method(method, X_train_enc, X_test_enc,
                   y_train, y_test,
                   X_train_raw, cat_cols, cont_cols,
                   metadata, paths, dataset, seed, logger):
    """
    Augment -> train -> evaluate -> SHAP -> save everything.
    Returns (metrics_rf, metrics_xgb, dist_metrics, shap_results)
    or (None, None, None, None) if the method fails.
    """
    method_dir = paths[method]

    # ------------------------------------------------------------------ #
    #  Step 1: generate augmented data                                    #
    # ------------------------------------------------------------------ #
    t0 = time.time()

    if method == "baseline":
        X_aug_enc = X_train_enc.copy()
        y_aug     = y_train.copy()
        X_syn     = None
        y_syn     = None
    else:
        n_gen = compute_n_generate_clf(y_train)
        X_aug_enc, y_aug, X_syn, y_syn = run_augmentation(
            method, X_train_enc, y_train,
            cat_cols, cont_cols, metadata,
            n_generate=n_gen,
            cfg=CFG, task=TASK,
        )
        if X_aug_enc is None:
            logger.warning(f"  [{method}] Augmentation failed - skipping.")
            return None, None, None, None

        # Professor point 5: data integrity assertions
        assert len(X_aug_enc) > len(X_train_enc),             f"[{method}] Augmentation added no rows!"
        assert X_aug_enc.isna().sum().sum() == 0,             f"[{method}] NaN values in augmented set!"
        assert len(X_aug_enc) == len(y_aug),             f"[{method}] X/y length mismatch in augmented set!"

        save_augmented_data(X_aug_enc, y_aug, X_syn, y_syn, method_dir, seed)
        logger.info(f"  [{method}] Generated {len(X_syn)} synthetic rows "
                    f"in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    #  Step 2: train RF and XGBoost                                       #
    # ------------------------------------------------------------------ #
    n_cls = y_train.nunique()

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_aug_enc, y_aug)

    xgb_params = XGB_PARAMS.copy()
    if n_cls > 2:
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"] = n_cls
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_aug_enc, y_aug)

    # ------------------------------------------------------------------ #
    #  Step 3: evaluate on UNTOUCHED test set                             #
    # ------------------------------------------------------------------ #
    rf_pred  = rf.predict(X_test_enc)
    xgb_pred = xgb.predict(X_test_enc)
    rf_prob  = rf.predict_proba(X_test_enc)
    xgb_prob = xgb.predict_proba(X_test_enc)

    m_rf  = clf_metrics(y_test, rf_pred,  rf_prob,  "rf",  method)
    m_xgb = clf_metrics(y_test, xgb_pred, xgb_prob, "xgb", method)
    m_rf["seed"]    = seed;  m_rf["dataset"]  = dataset
    m_xgb["seed"]   = seed;  m_xgb["dataset"] = dataset

    for m, name in [(m_rf, "rf"), (m_xgb, "xgb")]:
        save_metrics_dict(m, os.path.join(
            method_dir, f"metrics_{name}_seed{seed}.csv"))

    save_text_report(
        clf_report_str(y_test, rf_pred),
        os.path.join(method_dir, f"classification_report_rf_seed{seed}.txt"))
    save_text_report(
        clf_report_str(y_test, xgb_pred),
        os.path.join(method_dir, f"classification_report_xgb_seed{seed}.txt"))

    logger.info(f"  [{method}] RF  F1={m_rf['f1_macro']:.4f}  "
                f"AUC={m_rf['auc_roc']}")
    logger.info(f"  [{method}] XGB F1={m_xgb['f1_macro']:.4f}  "
                f"AUC={m_xgb['auc_roc']}")

    # ------------------------------------------------------------------ #
    #  Step 4: distribution metrics (skip for baseline)                   #
    # ------------------------------------------------------------------ #
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
        plot_correlation_heatmap(
            X_train_enc,
            f"Correlation - Original - {dataset}",
            os.path.join(method_dir, f"corr_original_seed{seed}.png"))
        plot_correlation_heatmap(
            X_aug_enc,
            f"Correlation - Augmented ({method}) - {dataset}",
            os.path.join(method_dir, f"corr_augmented_seed{seed}.png"))
        plot_correlation_diff(
            X_train_enc, X_aug_enc, method,
            os.path.join(method_dir, f"corr_diff_seed{seed}.png"))
        plot_ks_per_feature(
            dist_m, method, dataset,
            os.path.join(method_dir, f"ks_per_feature_seed{seed}.png"))

    # ------------------------------------------------------------------ #
    #  Step 5: SHAP                                                        #
    # ------------------------------------------------------------------ #
    shap_res = shap_for_models(rf, xgb, X_test_enc)
    shap_out = {}

    for mdl_name, (sv, imp, exp) in shap_res.items():
        save_shap_values(sv, imp, method_dir, mdl_name, seed)
        plot_shap_bar(
            imp, mdl_name, method,
            os.path.join(method_dir, f"shap_bar_{mdl_name}_seed{seed}.png"))
        plot_shap_summary(
            sv, X_test_enc, mdl_name, method,
            os.path.join(method_dir, f"shap_summary_{mdl_name}_seed{seed}.png"))
        shap_out[mdl_name] = imp

    return m_rf, m_xgb, dist_result, shap_out


# --------------------------------------------------------------------------- #
#  DATASET PIPELINE - all seeds x all methods                                 #
# --------------------------------------------------------------------------- #

def run_dataset(dataset, logger):
    """Full pipeline for one dataset across all seeds and methods."""
    banner(f"Dataset: {dataset.upper()}  |  Task: {TASK}  |  Size: {SIZE}")

    # Load and inspect
    df   = load_dataset(dataset)
    info = inspect_dataset(df, dataset)

    cat_cols  = info["cat_cols"]
    cont_cols = info["cont_cols"]
    bin_cols  = info["binary_cols"]
    all_cat   = cat_cols + bin_cols       # both types needed for SMOTE-NC

    # Create output folders
    paths = make_output_dirs(OUTPUT_ROOT, TASK, SIZE, dataset, AUG_METHODS)

    # Save experiment config
    exp_cfg = {**CFG, "dataset": dataset, "info": {
        k: v for k, v in info.items() if k != "class_counts"
    }}
    save_experiment_config(exp_cfg,
                           os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset))

    # Accumulators
    all_metrics_rf       = []
    all_metrics_xgb      = []
    all_dist             = []
    all_shap_stable_rf   = []
    all_shap_stable_xgb  = []

    # Main loop: seed x method
    for seed in SEEDS:
        banner(f"  Seed {seed}", char="-", width=50)

        # Split
        X_train_raw, X_test_raw, y_train, y_test = split_dataset(
            df, test_size=TEST_SIZE, task=TASK, seed=seed)

        # Professor point 5: snapshot test set hash before any processing
        test_hash = pd.util.hash_pandas_object(X_test_raw).sum()

        # Encode (fit on train only)
        X_train_enc, X_test_enc, encoder = encode_features(
            X_train_raw, X_test_raw, all_cat)

        # Build SDV metadata for CTGAN / TVAE
        train_df_meta = pd.concat(
            [X_train_enc, y_train.rename("target")], axis=1)
        metadata = build_sdv_metadata(train_df_meta, all_cat)

        # Save splits
        save_splits(X_train_enc, X_test_enc, y_train, y_test,
                    paths["data"], seed)

        baseline_shap_rf  = None
        baseline_shap_xgb = None

        for method in ["baseline"] + AUG_METHODS:
            print(f"\n  -> [{dataset}] seed={seed}  method={method}")

            # Initialize result so Pyright can trace both paths correctly
            result = (None, None, None, None)
            try:
                result = run_one_method(
                    method, X_train_enc, X_test_enc, y_train, y_test,
                    X_train_raw, all_cat, cont_cols, metadata,
                    paths, dataset, seed, logger,
                )
            except Exception as e:
                logger.error(f"  [{method}] Unexpected error: {e}",
                             exc_info=True)
                # result stays (None, None, None, None) — handled below

            m_rf, m_xgb, dist_m, shap_d = result
            if m_rf is None:
                continue

            # Professor point 5: verify test set unchanged after each method
            assert pd.util.hash_pandas_object(X_test_raw).sum() == test_hash, \
                f"TEST SET WAS MODIFIED after method={method}!"

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
                    stab.update({"seed": seed, "dataset": dataset,
                                 "model": mdl_name})
                    if mdl_name == "rf":
                        all_shap_stable_rf.append(stab)
                    else:
                        all_shap_stable_xgb.append(stab)

                    plot_shap_comparison(
                        base_imp, aug_imp, mdl_name, method,
                        os.path.join(paths[method],
                                     f"shap_comparison_{mdl_name}_seed{seed}.png"),
                    )

    # ------------------------------------------------------------------ #
    #  Aggregate and save cross-seed summaries                            #
    # ------------------------------------------------------------------ #
    cmp_dir = paths["comparison"]
    for records, fname in [
        (all_metrics_rf,      "all_methods_metrics_rf.csv"),
        (all_metrics_xgb,     "all_methods_metrics_xgb.csv"),
        (all_dist,            "all_methods_distribution.csv"),
        (all_shap_stable_rf,  "all_methods_shap_stability_rf.csv"),
        (all_shap_stable_xgb, "all_methods_shap_stability_xgb.csv"),
    ]:
        if records:
            pd.DataFrame(records).to_csv(
                os.path.join(cmp_dir, fname), index=False)

    # Comparison plots (averaged across seeds)
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
                        os.path.join(cmp_dir,
                                     f"perf_{metric}_{mdl}.png"))

    if all_shap_stable_rf:
        plot_shap_stability_heatmap(
            all_shap_stable_rf, dataset, "rf",
            os.path.join(cmp_dir, "shap_stability_heatmap_rf.png"))
        plot_shap_stability_heatmap(
            all_shap_stable_xgb, dataset, "xgb",
            os.path.join(cmp_dir, "shap_stability_heatmap_xgb.png"))

    logger.info(f"  Dataset {dataset} complete. "
                f"Outputs -> {os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset)}")


# --------------------------------------------------------------------------- #
#  ENTRY POINT                                                                 #
# --------------------------------------------------------------------------- #

def main():
    log_path = os.path.join(OUTPUT_ROOT, TASK, SIZE, "run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path, name=f"{TASK}_{SIZE}")

    banner(f"CLASSIFICATION - SMALL DATASETS  |  Seeds: {SEEDS}")
    logger.info(f"Starting {TASK}/{SIZE} experiment")
    logger.info(f"Datasets : {DATASETS}")
    logger.info(f"Methods  : {AUG_METHODS}")
    logger.info(f"Split    : {int((1 - TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")

    save_experiment_config(CFG, os.path.join(OUTPUT_ROOT, TASK, SIZE))

    for dataset in DATASETS:
        try:
            run_dataset(dataset, logger)
        except Exception as e:
            logger.error(f"Dataset {dataset} failed: {e}", exc_info=True)

    banner("ALL SMALL CLASSIFICATION DATASETS COMPLETE")
    logger.info(
        f"All outputs saved under: {os.path.join(OUTPUT_ROOT, TASK, SIZE)}")


if __name__ == "__main__":
    main()