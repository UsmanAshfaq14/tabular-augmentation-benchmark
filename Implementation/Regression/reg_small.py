"""
reg_small.py
============
Regression - Small Datasets
Datasets  : 230_machine_cpu | 560_bodyfat | 547_no2 | 1028_SWD
Split     : 70 / 30  (random; see justification below)
Seeds     : 5 (42, 123, 456, 789, 999)
Methods   : baseline | smote_nc(SMOGN) | ctgan | tvae | tabddpm | great | dice

Run:
    python reg_small.py

All outputs -> outputs/regression/small/<dataset>/<method>/

Mirrors clf_small.py exactly (checkpoint/resume, data-integrity assertions,
SHAP stability, distribution fidelity) with three regression-specific changes:

  1. Models   : RandomForestRegressor + XGBRegressor (reg:squarederror).
  2. Metrics  : RMSE / MAE / R2 (no F1/AUC). Distribution KS+CorrFrob and SHAP
                stability are computed identically to the classification suite.
  3. Rare-region evaluation : in imbalanced regression the quantity that
                matters is error on the *rare extreme* targets, not just the
                overall average. Every method is scored OVERALL, on the RARE
                region, and on the NORMAL region (see utils.evaluation).

70/30 split justification (same as the classification small phase): the four
small regression sets hold ~200-1,000 rows. A 30% test partition keeps the
held-out error estimate stable on so few samples, and matching the
classification small phase's ratio lets the two studies be compared directly.
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

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from utils.config import (
    SEEDS, OUTPUT_ROOT, SHAP_TOP_K,
    GAN_EPOCHS, GREAT_EPOCHS, DIFFUSION_STEPS, SMOTE_K,
    LARGE_SUBSAMPLE_N, REG_AUG_PCT,
)
from utils.data_utils import (
    load_dataset, inspect_dataset, encode_features,
    split_dataset, build_sdv_metadata,
)
from utils.aug_methods import run_augmentation
from utils.evaluation import (
    reg_metrics, reg_residual_report,
    compute_rarity_mask, reg_metrics_regional,
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
    save_metrics_dict, save_shap_values,
    save_text_report, save_distribution_metrics,
    save_experiment_config, banner,
)

# --------------------------------------------------------------------------- #
#  SIZE-SPECIFIC CONFIG                                                         #
# --------------------------------------------------------------------------- #
SIZE        = "small"
TASK        = "regression"
TEST_SIZE   = 0.30                  # 70/30 split (see header justification)
DATASETS    = ["230_machine_cpu", "560_bodyfat", "547_no2", "1028_SWD"]
AUG_METHODS = ["smogn", "ctgan", "tvae", "tabddpm", "great", "dice"]

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
    "reg_aug_pct":     REG_AUG_PCT,
}

RF_PARAMS  = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
XGB_PARAMS = {"n_estimators": 100, "random_state": 42,
              "objective": "reg:squarederror", "eval_metric": "rmse",
              "verbosity": 0}


# --------------------------------------------------------------------------- #
#  SESSION TIME GUARD  (7-hour IRIT limit)                                      #
# --------------------------------------------------------------------------- #
# The IRIT server kills the kernel after 7 h. We stop ~30 min early and exit
# gracefully; because every (dataset, method, seed) cell is checkpointed to
# disk, simply re-running the script resumes exactly where it left off.

SESSION_START       = time.time()
SESSION_LIMIT_HOURS = 6.5          # stop 30 min before the 7-hour disconnect


def session_elapsed_h():
    return (time.time() - SESSION_START) / 3600.0


def session_ok(logger=None):
    """True while session budget remains; warns under 30 min, stops at limit."""
    remaining = SESSION_LIMIT_HOURS - session_elapsed_h()
    if remaining <= 0:
        msg = (f"SESSION LIMIT REACHED ({session_elapsed_h():.1f}h). "
               f"Stopping gracefully — re-run to continue from checkpoint.")
        (logger.warning if logger else print)(msg)
        return False
    if remaining < 0.5:
        msg = f"Session budget low: {remaining*60:.0f} min remaining."
        (logger.warning if logger else print)(msg)
    return True


# --------------------------------------------------------------------------- #
#  CHECKPOINT / RESUME  (mirrors clf_medium.py)                                 #
# --------------------------------------------------------------------------- #
# A cell is "done" once its RF metrics CSV exists. On resume we skip done cells
# and reload their saved per-cell outputs (metrics, distribution, SHAP
# stability) so the cross-seed summary files stay complete regardless of where
# the previous session stopped.

def _cell_dir(dataset, method):
    return os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset, method)


def is_completed(dataset, method, seed, model="rf"):
    """True if this (dataset, method, seed) already has saved metrics."""
    return os.path.exists(os.path.join(
        _cell_dir(dataset, method), f"metrics_{model}_seed{seed}.csv"))


def count_completed(dataset):
    """Return (completed, total) cells for a dataset."""
    total     = (len(AUG_METHODS) + 1) * len(SEEDS)        # +1 baseline
    completed = sum(1 for m in ["baseline"] + AUG_METHODS
                    for s in SEEDS if is_completed(dataset, m, s))
    return completed, total


def load_saved(dataset, method, seed,
               rf_list, xgb_list, dist_list, stab_rf_list, stab_xgb_list):
    """Reload a checkpointed cell's per-cell outputs into the accumulators so
    the cross-seed summaries are complete on resume."""
    cdir = _cell_dir(dataset, method)
    for model, store in [("rf", rf_list), ("xgb", xgb_list)]:
        f = os.path.join(cdir, f"metrics_{model}_seed{seed}.csv")
        if os.path.exists(f):
            store.append(pd.read_csv(f).iloc[0].to_dict())
    fd = os.path.join(cdir, f"distribution_metrics_seed{seed}.csv")
    if method != "baseline" and os.path.exists(fd):
        dist_list.append(pd.read_csv(fd).iloc[0].to_dict())
    for model, store in [("rf", stab_rf_list), ("xgb", stab_xgb_list)]:
        fs = os.path.join(cdir, f"shap_stability_{model}_seed{seed}.csv")
        if method != "baseline" and os.path.exists(fs):
            store.append(pd.read_csv(fs).iloc[0].to_dict())


def load_baseline_shap(dataset, seed):
    """Reload baseline SHAP importance rankings (rf, xgb) from disk so SHAP
    stability can still be computed for methods that run after a resume
    boundary within a partially-finished seed. Returns (rf_imp, xgb_imp)."""
    out = {}
    for model in ("rf", "xgb"):
        f = os.path.join(_cell_dir(dataset, "baseline"),
                         f"shap_importance_{model}_seed{seed}.csv")
        if os.path.exists(f):
            s = pd.read_csv(f, index_col=0).iloc[:, 0]
            out[model] = s
    return out.get("rf"), out.get("xgb")


# --------------------------------------------------------------------------- #
#  HELPER - how many synthetic rows to generate (regression)                   #
# --------------------------------------------------------------------------- #

def compute_n_generate_reg(y_train):
    """
    Regression has no minority class, so we add a fixed fraction
    (REG_AUG_PCT) of extra rows. SMOGN ignores this (it balances by its
    own relevance function); CTGAN/TVAE/TabDDPM/GReaT use it as the row count.
    """
    return max(10, int(round(REG_AUG_PCT * len(y_train))))


# --------------------------------------------------------------------------- #
#  HELPER - regression-aware dataset inspection (target stats, not classes)    #
# --------------------------------------------------------------------------- #

def inspect_regression(df, name):
    """
    Reuse the feature-type detection from inspect_dataset, but report
    continuous-target statistics instead of (meaningless) class counts.
    """
    import io, contextlib
    # inspect_dataset prints a classification-style "Classes : N" line that is
    # meaningless for a continuous target; suppress its stdout and print a clean
    # regression summary instead. We still use its returned column-type dict.
    with contextlib.redirect_stdout(io.StringIO()):
        info = inspect_dataset(df, name)        # detects binary/cat/cont cols

    y = df["target"].astype(float)
    tgt = {
        "target_min":  round(float(y.min()),  4),
        "target_max":  round(float(y.max()),  4),
        "target_mean": round(float(y.mean()), 4),
        "target_std":  round(float(y.std()),  4),
        "target_skew": round(float(y.skew()), 4),
    }
    print(f"\n{'='*60}")
    print(f"  Dataset : {name}  |  Task: regression")
    print(f"  Samples : {info['n_samples']}  |  Features: {info['n_features']}")
    print(f"  binary={len(info['binary_cols'])}  "
          f"cat={len(info['cat_cols'])}  cont={len(info['cont_cols'])}  "
          f"|  Missing: {info['missing']}")
    print(f"  Target  : mean={tgt['target_mean']}  std={tgt['target_std']}  "
          f"range=[{tgt['target_min']}, {tgt['target_max']}]  "
          f"skew={tgt['target_skew']}")
    print(f"{'='*60}\n")
    info.update(tgt)
    return info


# --------------------------------------------------------------------------- #
#  CORE PIPELINE - one dataset x one seed x one method                        #
# --------------------------------------------------------------------------- #

def run_one_method(method, X_train_enc, X_test_enc,
                   y_train, y_test,
                   X_train_raw, cat_cols, cont_cols,
                   metadata, paths, dataset, seed, logger,
                   rare_mask):
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
        n_gen = compute_n_generate_reg(y_train)
        X_aug_enc, y_aug, X_syn, y_syn = run_augmentation(
            method, X_train_enc, y_train,
            cat_cols, cont_cols, metadata,
            n_generate=n_gen,
            cfg=CFG, task=TASK,
        )
        if X_aug_enc is None:
            logger.warning(f"  [{method}] Augmentation failed - skipping.")
            return None, None, None, None

        # Data integrity assertions (same as classification)
        assert len(X_aug_enc) > len(X_train_enc), \
            f"[{method}] Augmentation added no rows!"
        assert X_aug_enc.isna().sum().sum() == 0, \
            f"[{method}] NaN values in augmented set!"
        assert len(X_aug_enc) == len(y_aug), \
            f"[{method}] X/y length mismatch in augmented set!"

        save_augmented_data(X_aug_enc, y_aug, X_syn, y_syn, method_dir, seed)
        logger.info(f"  [{method}] Generated {len(X_syn)} synthetic rows "
                    f"in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    #  Step 2: train RF and XGBoost regressors                            #
    # ------------------------------------------------------------------ #
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_aug_enc, y_aug)

    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X_aug_enc, y_aug)

    # ------------------------------------------------------------------ #
    #  Step 3: evaluate on UNTOUCHED test set                             #
    # ------------------------------------------------------------------ #
    rf_pred  = rf.predict(X_test_enc)
    xgb_pred = xgb.predict(X_test_enc)

    m_rf  = reg_metrics(y_test, rf_pred,  "rf",  method)
    m_xgb = reg_metrics(y_test, xgb_pred, "xgb", method)

    # Rare-region breakdown (overall / rare / normal) — merged into the row
    reg_rf  = reg_metrics_regional(y_test, rf_pred,  rare_mask, "rf",  method)
    reg_xgb = reg_metrics_regional(y_test, xgb_pred, rare_mask, "xgb", method)
    for d, extra in [(m_rf, reg_rf), (m_xgb, reg_xgb)]:
        for k, v in extra.items():
            if k not in ("model", "method"):
                d[k] = v
    m_rf["seed"]    = seed;  m_rf["dataset"]  = dataset
    m_xgb["seed"]   = seed;  m_xgb["dataset"] = dataset

    for m, name in [(m_rf, "rf"), (m_xgb, "xgb")]:
        save_metrics_dict(m, os.path.join(
            method_dir, f"metrics_{name}_seed{seed}.csv"))

    save_text_report(
        reg_residual_report(y_test, rf_pred, rare_mask),
        os.path.join(method_dir, f"residual_report_rf_seed{seed}.txt"))
    save_text_report(
        reg_residual_report(y_test, xgb_pred, rare_mask),
        os.path.join(method_dir, f"residual_report_xgb_seed{seed}.txt"))

    logger.info(f"  [{method}] RF  RMSE={m_rf['rmse']:.4f}  "
                f"R2={m_rf['r2']:.4f}  rareRMSE={m_rf['rare_rmse']}")
    logger.info(f"  [{method}] XGB RMSE={m_xgb['rmse']:.4f}  "
                f"R2={m_xgb['r2']:.4f}  rareRMSE={m_xgb['rare_rmse']}")

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
    info = inspect_regression(df, dataset)

    cat_cols  = info["cat_cols"]
    cont_cols = info["cont_cols"]
    bin_cols  = info["binary_cols"]
    all_cat   = cat_cols + bin_cols       # both types encoded together

    # Create output folders
    paths = make_output_dirs(OUTPUT_ROOT, TASK, SIZE, dataset, AUG_METHODS)

    # Save experiment config
    exp_cfg = {**CFG, "dataset": dataset, "info": info}
    save_experiment_config(exp_cfg,
                           os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset))

    # Accumulators
    all_metrics_rf       = []
    all_metrics_xgb      = []
    all_dist             = []
    all_shap_stable_rf   = []
    all_shap_stable_xgb  = []

    # ── Dataset-level checkpoint ──────────────────────────────────────────────
    done, total = count_completed(dataset)
    if done >= total:
        logger.info(f"  {dataset}: all {total} cells already complete — "
                    f"rebuilding summaries from checkpoint, no recompute.")
    elif done > 0:
        logger.info(f"  {dataset}: resuming — {done}/{total} cells already done.")

    # Main loop: seed x method
    for seed in SEEDS:

        # Stop early if the 7-hour session budget is nearly spent.
        if not session_ok(logger):
            logger.warning(f"  Stopping at seed {seed}. "
                           f"Re-run to resume from checkpoint.")
            break

        # ── Seed-level checkpoint: every method done → reload, don't recompute ─
        if all(is_completed(dataset, m, seed)
               for m in ["baseline"] + AUG_METHODS):
            logger.info(f"  Seed {seed}: fully checkpointed — loading saved.")
            for m in ["baseline"] + AUG_METHODS:
                load_saved(dataset, m, seed, all_metrics_rf, all_metrics_xgb,
                           all_dist, all_shap_stable_rf, all_shap_stable_xgb)
            continue

        banner(f"  Seed {seed}", char="-", width=50)

        # Split (random for regression)
        X_train_raw, X_test_raw, y_train, y_test = split_dataset(
            df, test_size=TEST_SIZE, task=TASK, seed=seed)

        # Snapshot test set hash before any processing (integrity check)
        test_hash = pd.util.hash_pandas_object(X_test_raw).sum()

        # Rare-target region defined on TRAIN targets (no test leakage)
        rare_mask, rare_info = compute_rarity_mask(y_train, y_test)
        logger.info(f"  [seed {seed}] rarity={rare_info['rarity_rule']}  "
                    f"thresholds=[{rare_info['low_threshold']}, "
                    f"{rare_info['high_threshold']}]  "
                    f"rare_test={int(rare_mask.sum())}/{len(y_test)}")

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
        # If baseline already ran in a previous session, reload its SHAP rankings
        # so methods run this session can still compute SHAP stability against it.
        if is_completed(dataset, "baseline", seed):
            baseline_shap_rf, baseline_shap_xgb = load_baseline_shap(dataset, seed)

        for method in ["baseline"] + AUG_METHODS:

            # ── Method-level checkpoint: skip + reload, never recompute ───────
            if is_completed(dataset, method, seed):
                logger.info(f"  [{method}] seed={seed}: checkpointed — skipping.")
                load_saved(dataset, method, seed,
                           all_metrics_rf, all_metrics_xgb, all_dist,
                           all_shap_stable_rf, all_shap_stable_xgb)
                continue

            # ── Session budget check before each (expensive) method ───────────
            if not session_ok(logger):
                logger.warning(f"  Session limit before [{method}] seed={seed}. "
                               f"Re-run to resume from checkpoint.")
                break

            print(f"\n  -> [{dataset}] seed={seed}  method={method}")

            result = (None, None, None, None)
            try:
                result = run_one_method(
                    method, X_train_enc, X_test_enc, y_train, y_test,
                    X_train_raw, all_cat, cont_cols, metadata,
                    paths, dataset, seed, logger, rare_mask,
                )
            except Exception as e:
                logger.error(f"  [{method}] Unexpected error: {e}",
                             exc_info=True)

            m_rf, m_xgb, dist_m, shap_d = result
            if m_rf is None:
                continue

            # Verify test set unchanged after each method
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
                    # Persist per-cell so a resumed run can reload it into the
                    # cross-seed SHAP-stability summary (otherwise the summary
                    # would be partial after a mid-dataset restart).
                    pd.DataFrame([stab]).to_csv(
                        os.path.join(paths[method],
                                     f"shap_stability_{mdl_name}_seed{seed}.csv"),
                        index=False)
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

    # Comparison plots (averaged across seeds) — regression metrics
    if all_metrics_rf:
        for metric in ["rmse", "mae", "r2", "rare_rmse"]:
            for mdl, recs in [("rf", all_metrics_rf), ("xgb", all_metrics_xgb)]:
                df_m = pd.DataFrame(recs)
                if metric not in df_m.columns:
                    continue
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

    banner(f"REGRESSION - SMALL DATASETS  |  Seeds: {SEEDS}")
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

    banner("ALL SMALL REGRESSION DATASETS COMPLETE")
    logger.info(
        f"All outputs saved under: {os.path.join(OUTPUT_ROOT, TASK, SIZE)}")


if __name__ == "__main__":
    main()