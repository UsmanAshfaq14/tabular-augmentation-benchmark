"""
reg_medium.py
=============
Regression - Medium Datasets
Datasets  : titanic | 529_pollen | 503_wind | 294_satellite_image
Split     : 80 / 20  (random; see justification below)
Seeds     : 5 (42, 123, 456, 789, 999)
Methods   : baseline | smogn | ctgan | tvae | tabddpm | dice   (inline)
            great                                              (separate script)

Run:
    python reg_medium.py                 # runs baseline + 5 inline methods
    python reg_great_medium.py           # then GReaT (GPU, own time budget)
    python reg_medium.py                 # re-run: rebuilds summaries incl. great

All outputs -> outputs/regression/medium/<dataset>/<method>/

Mirrors reg_small.py exactly (checkpoint/resume, session guard, data-integrity
assertions, rare-region evaluation, SHAP stability, distribution fidelity)
with the medium-specific changes:

  1. Split    : 80/20 instead of 70/30. The medium sets hold 2,201-6,574 rows,
                so a 20% test partition still gives 440-1,315 held-out points -
                ample for stable error estimates - while the extra 10% of
                training data feeds the generative models, which benefit most
                from more rows. Matches the classification medium phase so the
                two studies stay directly comparable.
  2. Deeper generative training (from utils.config, keyed by SIZE):
                GAN_EPOCHS 100->300, DIFFUSION_STEPS 5000->15000 (n_t 50),
                GREAT_EPOCHS 25->75, SMOGN k 3->5.
  3. GReaT    : excluded from the inline loop and run by reg_great_medium.py
                (same isolation as the classification medium phase - GPT-2
                fine-tuning at 75 epochs on ~2-5k rows takes hours and can
                hang). This script still COUNTS great cells, reloads them on
                resume, and includes them in the cross-seed summaries, so once
                reg_great_medium.py has finished, re-running this script
                produces the complete 7-method comparison files.
  4. DiCE     : per-cell wall-clock budget raised 600s -> 1800s via CFG.
                n_query scales with n_generate (ceil(0.2*n_train / 3)), i.e.
                118-344 genetic searches per cell on these datasets; the
                larger budget keeps the +20% augmentation target reachable
                while the existing 60s per-query SIGALRM still kills any
                single runaway search.
  5. XGBoost  : GPU (tree_method=hist, device=cuda) when available - medium
                data is large enough for GPU training to pay off.

Rare-region note (titanic): compute_rarity_mask falls back to the 10th/90th
decile tails when the IQR rule flags nothing. On a binary {0,1} target both
deciles coincide with the data extremes, so the rare mask can be EMPTY -
rare-region metrics are then NaN by design (reg_metrics_regional) and the
dataset acts as a degenerate-target control case. The seed loop logs a
prominent warning when this happens so the run.log is self-explanatory.
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
    split_dataset, build_sdv_metadata, impute_missing,
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
SIZE        = "medium"
TASK        = "regression"
TEST_SIZE   = 0.20                  # 80/20 split (see header justification)
DATASETS    = ["titanic", "529_pollen", "503_wind", "294_satellite_image"]

# MEDIUM: GReaT runs in its own script (reg_great_medium.py) so GPT-2
# fine-tuning can never block the main pipeline - same isolation as the
# classification medium phase. It stays in AUG_METHODS_ALL so its checkpointed
# cells are counted, reloaded on resume, and included in the summaries.
AUG_METHODS_INLINE = ["smogn", "ctgan", "tvae", "tabddpm", "dice"]
AUG_METHODS_ALL    = AUG_METHODS_INLINE + ["great"]
METHODS_ALL        = ["baseline"] + AUG_METHODS_ALL

CFG = {
    "size":            SIZE,
    "task":            TASK,
    "test_size":       TEST_SIZE,
    "datasets":        DATASETS,
    "seeds":           SEEDS,
    "methods":         AUG_METHODS_ALL,
    "gan_epochs":      GAN_EPOCHS[SIZE],        # 300
    "great_epochs":    GREAT_EPOCHS[SIZE],      # 75 (used by reg_great_medium)
    "diffusion_steps": DIFFUSION_STEPS[SIZE],   # 15000 -> ForestDiffusion n_t=50
    "smote_k":         SMOTE_K[SIZE],           # 5
    "reg_aug_pct":     REG_AUG_PCT,             # 0.20
    # MEDIUM: DiCE wall-clock budget per cell. n_query is 118-344 here
    # (vs 30-141 in small), so the small-phase 600s default would cut
    # generation short of the +20% target. Per-query SIGALRM stays at 60s.
    "dice_max_seconds":   1800,
    "dice_query_timeout": 60,
}

RF_PARAMS  = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
XGB_PARAMS = {"n_estimators": 100, "random_state": 42,
              "objective": "reg:squarederror", "eval_metric": "rmse",
              "verbosity": 0}

# MEDIUM: GPU-aware XGBoost (same pattern as reg_great_small.py /
# great_medium.py). Medium training sets (1.7k-5.1k rows after the 80/20
# split, augmented +20%) are large enough for GPU `hist` to pay off.
try:
    import torch
    if torch.cuda.is_available():
        XGB_PARAMS["tree_method"] = "hist"
        XGB_PARAMS["device"]      = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: not available - XGBoost on CPU")
except ImportError:
    print("torch not installed - XGBoost on CPU")


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
#  CHECKPOINT / RESUME  (mirrors reg_small.py / clf_medium.py)                  #
# --------------------------------------------------------------------------- #
# A cell is "done" once its RF metrics CSV exists. On resume we skip done cells
# and reload their saved per-cell outputs (metrics, distribution, SHAP
# stability) so the cross-seed summary files stay complete regardless of where
# the previous session stopped. GReaT cells (written by reg_great_medium.py)
# participate in exactly the same scheme.

def _cell_dir(dataset, method):
    return os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset, method)


def is_completed(dataset, method, seed, model="rf"):
    """True if this (dataset, method, seed) already has saved metrics."""
    return os.path.exists(os.path.join(
        _cell_dir(dataset, method), f"metrics_{model}_seed{seed}.csv"))


def count_completed(dataset):
    """Return (completed, total) cells for a dataset - great included."""
    total     = len(METHODS_ALL) * len(SEEDS)              # 7 x 5 = 35
    completed = sum(1 for m in METHODS_ALL
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
    Holding the synthetic QUANTITY constant across phases (always +20%) keeps
    the small / medium / large comparison clean - only data scale changes.
    """
    return max(10, int(round(REG_AUG_PCT * len(y_train))))


# --------------------------------------------------------------------------- #
#  HELPER - regression-aware dataset inspection (target stats, not classes)    #
# --------------------------------------------------------------------------- #

def inspect_regression(df, name):
    """
    Reuse the feature-type detection from inspect_dataset, but report
    continuous-target statistics instead of (meaningless) class counts.
    Additionally flags near-discrete / binary targets, which matter for the
    rare-region definition on the medium datasets (titanic, satellite_image).
    """
    import io, contextlib
    # inspect_dataset prints a classification-style "Classes : N" line that is
    # meaningless for a continuous target; suppress its stdout and print a clean
    # regression summary instead. We still use its returned column-type dict.
    with contextlib.redirect_stdout(io.StringIO()):
        info = inspect_dataset(df, name)        # detects binary/cat/cont cols

    y = df["target"].astype(float)
    n_unique = int(y.nunique())
    tgt = {
        "target_min":    round(float(y.min()),  4),
        "target_max":    round(float(y.max()),  4),
        "target_mean":   round(float(y.mean()), 4),
        "target_std":    round(float(y.std()),  4),
        "target_skew":   round(float(y.skew()), 4),
        "target_unique": n_unique,
    }
    print(f"\n{'='*60}")
    print(f"  Dataset : {name}  |  Task: regression")
    print(f"  Samples : {info['n_samples']}  |  Features: {info['n_features']}")
    print(f"  binary={len(info['binary_cols'])}  "
          f"cat={len(info['cat_cols'])}  cont={len(info['cont_cols'])}  "
          f"|  Missing: {info['missing']}")
    print(f"  Target  : mean={tgt['target_mean']}  std={tgt['target_std']}  "
          f"range=[{tgt['target_min']}, {tgt['target_max']}]  "
          f"skew={tgt['target_skew']}  unique={n_unique}")
    if n_unique <= 2:
        print(f"  NOTE    : target has only {n_unique} distinct values - "
              f"degenerate for regression. Rare region will likely be EMPTY "
              f"(rare metrics NaN); SMOGN/DiCE may produce weak or no samples. "
              f"Treated as a degenerate-target control case.")
    elif n_unique <= 10:
        print(f"  NOTE    : near-discrete target ({n_unique} values) - "
              f"decile-tail rarity rule expected (same as SWD in the small "
              f"phase).")
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
    # MEDIUM: rows whose TARGET is missing are unusable for supervised
    # learning — drop them up front (features with NaN are imputed per split).
    n_bad_y = int(df["target"].isna().sum())
    if n_bad_y:
        df = df[df["target"].notna()].reset_index(drop=True)
        logger.warning(f"  {dataset}: dropped {n_bad_y} rows with missing "
                       f"target ({len(df)} rows remain).")
    info = inspect_regression(df, dataset)

    cat_cols  = info["cat_cols"]
    cont_cols = info["cont_cols"]
    bin_cols  = info["binary_cols"]
    all_cat   = cat_cols + bin_cols       # both types encoded together

    # Create output folders (great/ included so reg_great_medium.py and the
    # summary reload share one structure)
    paths = make_output_dirs(OUTPUT_ROOT, TASK, SIZE, dataset, AUG_METHODS_ALL)

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
        if all(is_completed(dataset, m, seed) for m in METHODS_ALL):
            logger.info(f"  Seed {seed}: fully checkpointed — loading saved.")
            for m in METHODS_ALL:
                load_saved(dataset, m, seed, all_metrics_rf, all_metrics_xgb,
                           all_dist, all_shap_stable_rf, all_shap_stable_xgb)
            continue

        banner(f"  Seed {seed}", char="-", width=50)

        # Split (random for regression)
        X_train_raw, X_test_raw, y_train, y_test = split_dataset(
            df, test_size=TEST_SIZE, task=TASK, seed=seed)

        # MEDIUM: leakage-safe imputation (fit on TRAIN only) — must run
        # BEFORE the integrity hash and before encoding/augmentation. titanic
        # carries NaN in 5 feature columns; every augmenter and DiCE's internal
        # model assume complete data (see utils.data_utils.impute_missing for
        # the failure modes). The hash below then guards the IMPUTED test set,
        # which is the frame every method is evaluated on.
        X_train_raw, X_test_raw, imp_report = impute_missing(
            X_train_raw, X_test_raw, all_cat)
        if imp_report:
            tot_tr = sum(v["n_train"] for v in imp_report.values())
            tot_te = sum(v["n_test"]  for v in imp_report.values())
            fills  = {c: v["fill"] for c, v in imp_report.items()}
            logger.info(f"  [seed {seed}] imputed {tot_tr} train + {tot_te} "
                        f"test cells (median/mode from train): {fills}")

        # Snapshot test set hash before any processing (integrity check)
        test_hash = pd.util.hash_pandas_object(X_test_raw).sum()

        # Rare-target region defined on TRAIN targets (no test leakage)
        rare_mask, rare_info = compute_rarity_mask(y_train, y_test)
        logger.info(f"  [seed {seed}] rarity={rare_info['rarity_rule']}  "
                    f"thresholds=[{rare_info['low_threshold']}, "
                    f"{rare_info['high_threshold']}]  "
                    f"rare_test={int(rare_mask.sum())}/{len(y_test)}")
        # MEDIUM: degenerate-target guard (titanic). An empty rare region is
        # handled by design (rare metrics NaN) but must be loud in the log.
        if int(rare_mask.sum()) == 0:
            logger.warning(
                f"  [seed {seed}] RARE REGION IS EMPTY for {dataset} - the "
                f"target distribution has no extreme tail under the IQR/decile "
                f"rules (expected for binary or bounded-discrete targets where both extreme values hold >=10% mass, e.g. titanic, satellite_image). rare_rmse/rare_r2 will "
                f"be NaN; overall metrics, distribution fidelity and SHAP "
                f"stability are unaffected. This dataset serves as the "
                f"degenerate-target control case.")

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

        for method in METHODS_ALL:

            # ── Method-level checkpoint: skip + reload, never recompute ───────
            if is_completed(dataset, method, seed):
                logger.info(f"  [{method}] seed={seed}: checkpointed — skipping.")
                load_saved(dataset, method, seed,
                           all_metrics_rf, all_metrics_xgb, all_dist,
                           all_shap_stable_rf, all_shap_stable_xgb)
                continue

            # ── GReaT is computed by reg_great_medium.py, never inline ────────
            if method == "great":
                logger.info(f"  [great] seed={seed}: deferred — run "
                            f"reg_great_medium.py, then re-run this script to "
                            f"fold great into the summaries.")
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
                for mth in METHODS_ALL:
                    sub = df_m[df_m["method"] == mth]
                    # MEDIUM: drop NaN before averaging — rare_* metrics are
                    # NaN by design when the rare region is empty (titanic).
                    sub = sub[pd.to_numeric(sub[metric],
                                            errors="coerce").notna()]
                    if len(sub):
                        row = sub[["model", "method"]].iloc[0].to_dict()
                        row[metric] = sub[metric].mean()
                        avg_recs.append(row)
                # MEDIUM: skip the plot entirely when no method has a finite
                # value (all-NaN rare_rmse would crash matplotlib axis limits).
                if avg_recs and any(np.isfinite(r[metric]) for r in avg_recs):
                    plot_performance_comparison(
                        avg_recs, metric, TASK, dataset, mdl,
                        os.path.join(cmp_dir, f"perf_{metric}_{mdl}.png"))
                else:
                    logger.info(f"  [plots] {metric}/{mdl}: no finite values "
                                f"(empty rare region) — plot skipped.")

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

    banner(f"REGRESSION - MEDIUM DATASETS  |  Seeds: {SEEDS}")
    logger.info(f"Starting {TASK}/{SIZE} experiment")
    logger.info(f"Datasets : {DATASETS}")
    logger.info(f"Methods  : {AUG_METHODS_INLINE} (inline) + ['great'] "
                f"(reg_great_medium.py)")
    logger.info(f"Split    : {int((1 - TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
    logger.info(f"Epochs   : GAN={GAN_EPOCHS[SIZE]}  "
                f"diffusion_steps={DIFFUSION_STEPS[SIZE]}  "
                f"smogn_k={SMOTE_K[SIZE]}  "
                f"dice_budget={CFG['dice_max_seconds']}s")

    save_experiment_config(CFG, os.path.join(OUTPUT_ROOT, TASK, SIZE))

    for dataset in DATASETS:
        try:
            run_dataset(dataset, logger)
        except Exception as e:
            logger.error(f"Dataset {dataset} failed: {e}", exc_info=True)

    banner("ALL MEDIUM REGRESSION DATASETS COMPLETE (inline methods)")
    done_great = sum(1 for d in DATASETS for s in SEEDS
                     if is_completed(d, "great", s))
    if done_great < len(DATASETS) * len(SEEDS):
        logger.info(
            f"GReaT cells done so far: {done_great}/{len(DATASETS)*len(SEEDS)}. "
            f"Next: python reg_great_medium.py   (then re-run reg_medium.py "
            f"once to fold great into the comparison summaries).")
    logger.info(
        f"All outputs saved under: {os.path.join(OUTPUT_ROOT, TASK, SIZE)}")


if __name__ == "__main__":
    main()