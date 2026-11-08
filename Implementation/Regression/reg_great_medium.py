"""
reg_great_medium.py
=============================================================================
GReaT-only augmentation for reg_medium datasets.
Run this AFTER reg_medium.py has completed (it needs the baseline SHAP
rankings reg_medium.py saves, to compute SHAP stability for GReaT).

Why separate? (same reason as the classification medium phase)
  GReaT fine-tunes GPT-2 — at 75 epochs on 1.7k-5.1k training rows this takes
  on the order of hours per cell and can hang without visible output. Keeping
  it in its own script means it never blocks the main regression pipeline,
  and it resumes from the last finished seed if interrupted.

Compared with reg_great_small.py this script ALSO saves, per cell:
  - distribution metrics (KS / CorrFrob / KL) + the distribution plots
  - SHAP stability vs the baseline (Spearman, Top-5) + comparison plot
so GReaT cells are byte-equivalent in structure to the inline-method cells
and reg_medium.py can fold them into the complete 7-method comparison
summaries on its next run.

Usage:
    python reg_great_medium.py --dataset 529_pollen
    python reg_great_medium.py --dataset titanic 503_wind
    python reg_great_medium.py                      # all 4 datasets

Set HF_TOKEN to avoid HuggingFace rate limits:
    export HF_TOKEN=your_token
=============================================================================
"""

import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from utils.config        import (SEEDS, OUTPUT_ROOT, SHAP_TOP_K,
                                 GREAT_EPOCHS, GAN_EPOCHS, DIFFUSION_STEPS,
                                 SMOTE_K, REG_AUG_PCT)
from utils.data_utils    import (load_dataset, inspect_dataset, encode_features,
                                 split_dataset, impute_missing)
from utils.aug_methods   import apply_great
from utils.evaluation    import (reg_metrics, reg_residual_report,
                                 compute_rarity_mask, reg_metrics_regional,
                                 distribution_metrics, kl_divergence,
                                 shap_stability)
from utils.shap_analysis import shap_for_models
from utils.plotting      import (plot_shap_bar, plot_shap_summary,
                                 plot_shap_comparison,
                                 plot_feature_distributions,
                                 plot_correlation_heatmap,
                                 plot_correlation_diff, plot_ks_per_feature)
from utils.io_utils      import (setup_logger, save_metrics_dict,
                                 save_augmented_data, save_shap_values,
                                 save_text_report, save_distribution_metrics,
                                 banner)

# ── Config ────────────────────────────────────────────────────────────────────
SIZE         = "medium"
TASK         = "regression"
TEST_SIZE    = 0.20
ALL_DATASETS = ["titanic", "529_pollen", "503_wind", "294_satellite_image"]
METHOD       = "great"

RF_PARAMS  = {"n_estimators": 100, "random_state": 42, "n_jobs": 10}
XGB_PARAMS = {"n_estimators": 100, "random_state": 42,
              "objective": "reg:squarederror", "eval_metric": "rmse",
              "verbosity": 0}

# GPU-aware XGBoost (same pattern as great_medium.py)
try:
    import torch
    if torch.cuda.is_available():
        XGB_PARAMS["tree_method"] = "hist"
        XGB_PARAMS["device"]      = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: not available")
except ImportError:
    print("torch not installed")

# Max hours per dataset — GReaT skips ahead if it exceeds this (same value as
# the classification great_medium.py). 75 GPT-2 epochs on 1.7k-5.1k rows is
# the heaviest training in the whole regression study.
MAX_HOURS_PER_DATASET = 3.0

# Overall session guard (7-hour IRIT limit). 4 datasets x 3 h could exceed a
# session, so we also stop gracefully ~30 min before the disconnect; re-running
# resumes from the last finished seed.
SESSION_START       = time.time()
SESSION_LIMIT_HOURS = 6.5


def session_ok(logger=None):
    remaining = SESSION_LIMIT_HOURS - (time.time() - SESSION_START) / 3600.0
    if remaining <= 0:
        msg = ("SESSION LIMIT REACHED — stopping gracefully. "
               "Re-run to continue from the last finished seed.")
        (logger.warning if logger else print)(msg)
        return False
    return True


CFG = {
    "size": SIZE, "task": TASK, "test_size": TEST_SIZE,
    "methods": [METHOD], "seeds": SEEDS,
    "great_epochs": GREAT_EPOCHS[SIZE],          # 75
    "gan_epochs":   GAN_EPOCHS[SIZE],
    "diffusion_steps": DIFFUSION_STEPS[SIZE],
    "smote_k": SMOTE_K[SIZE],
    "reg_aug_pct": REG_AUG_PCT,
}


def _cell_dir(dataset, method):
    return os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset, method)


def is_done(dataset, seed):
    """True only if BOTH rf and xgb metrics are saved for this seed."""
    return all(
        os.path.exists(os.path.join(
            _cell_dir(dataset, METHOD), f"metrics_{model}_seed{seed}.csv"))
        for model in ["rf", "xgb"]
    )


def load_baseline_shap(dataset, seed):
    """Reload the baseline SHAP importance rankings saved by reg_medium.py so
    GReaT's SHAP stability is computed against the exact same baseline as the
    inline methods. Returns (rf_imp, xgb_imp) — None if missing."""
    out = {}
    for model in ("rf", "xgb"):
        f = os.path.join(_cell_dir(dataset, "baseline"),
                         f"shap_importance_{model}_seed{seed}.csv")
        if os.path.exists(f):
            out[model] = pd.read_csv(f, index_col=0).iloc[:, 0]
    return out.get("rf"), out.get("xgb")


def run_great_dataset(dataset, logger):
    done = sum(1 for s in SEEDS if is_done(dataset, s))
    if done == len(SEEDS):
        logger.info(f"  {dataset}/great: all {len(SEEDS)} seeds complete. Skipping.")
        return

    banner(f"GReaT — {dataset.upper()}  |  {done}/{len(SEEDS)} seeds done")

    df   = load_dataset(dataset)
    n_bad_y = int(df["target"].isna().sum())
    if n_bad_y:
        df = df[df["target"].notna()].reset_index(drop=True)
        logger.warning(f"  {dataset}: dropped {n_bad_y} rows with missing "
                       f"target ({len(df)} rows remain).")
    info = inspect_dataset(df, dataset)

    cat_cols  = info["cat_cols"] + info["binary_cols"]
    cont_cols = info["cont_cols"]

    method_dir = _cell_dir(dataset, METHOD)
    os.makedirs(method_dir, exist_ok=True)
    os.makedirs(os.path.join(method_dir, "distribution_plots"), exist_ok=True)

    ds_start = time.time()

    for seed in SEEDS:
        if is_done(dataset, seed):
            logger.info(f"  seed={seed}: already done. Skipping.")
            continue

        if not session_ok(logger):
            break

        elapsed_ds = (time.time() - ds_start) / 3600
        if elapsed_ds > MAX_HOURS_PER_DATASET:
            logger.warning(
                f"  {dataset}/great: time budget ({MAX_HOURS_PER_DATASET}h) "
                f"exceeded. Stopping. Re-run to resume from seed {seed}.")
            break

        print(f"\n  -> [{dataset}] seed={seed}  method=great", flush=True)

        # Random split for regression (no stratification) — identical to
        # reg_medium.py: same seed -> byte-identical split and rare region.
        X_tr_r, X_te_r, y_tr, y_te = split_dataset(
            df, test_size=TEST_SIZE, task=TASK, seed=seed)
        # Leakage-safe imputation — identical call to reg_medium.py so the
        # same seed yields byte-identical train/test frames across scripts.
        X_tr_r, X_te_r, imp_report = impute_missing(X_tr_r, X_te_r, cat_cols)
        if imp_report:
            logger.info(f"  [seed {seed}] imputed "
                        f"{sum(v['n_train'] for v in imp_report.values())} train + "
                        f"{sum(v['n_test'] for v in imp_report.values())} test cells")
        X_tr, X_te, _ = encode_features(X_tr_r, X_te_r, cat_cols)

        # NO label normalisation — the target is continuous.

        # Rare-target region from TRAIN targets (no test leakage)
        rare_mask, rare_info = compute_rarity_mask(y_tr, y_te)
        if int(rare_mask.sum()) == 0:
            logger.warning(f"  [seed {seed}] rare region EMPTY for {dataset} "
                           f"(binary / bounded-discrete target) — rare metrics "
                           f"will be NaN.")

        # n_generate: fixed fraction of the training set (REG_AUG_PCT)
        n_gen = max(10, int(round(REG_AUG_PCT * len(X_tr))))

        # ── Run GReaT ─────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            print("  [GReaT] Starting training...", flush=True)
            X_aug, y_aug, X_syn, y_syn = apply_great(
                X_tr, y_tr, n_generate=n_gen,
                epochs=GREAT_EPOCHS[SIZE], task=TASK,
            )
        except Exception as e:
            logger.error(f"  [great] seed={seed} failed: {e}", exc_info=True)
            print(f"  [GReaT] Error: {e}", flush=True)
            continue

        if X_aug is None or len(X_aug) <= len(X_tr):
            logger.warning(f"  [great] seed={seed}: generated 0 rows — skipping.")
            continue

        elapsed_method = time.time() - t0
        logger.info(f"  [great] Generated {len(X_syn)} rows in {elapsed_method:.1f}s")

        # ── Filter invalid GReaT-generated target values (regression) ─────────
        # apply_great already coerces non-numeric tokens to NaN, drops them and
        # range-filters; here we additionally guard against any non-finite value
        # that slipped through (same belt-and-braces as reg_great_small.py).
        finite_aug = np.isfinite(y_aug.astype(float))
        finite_syn = np.isfinite(y_syn.astype(float))
        if not finite_aug.all():
            n_bad = int((~finite_aug).sum())
            print(f"  [GReaT] Removed {n_bad} rows with non-finite y", flush=True)
            X_aug = X_aug[finite_aug].reset_index(drop=True)
            y_aug = y_aug[finite_aug].reset_index(drop=True)
            X_syn = X_syn[finite_syn].reset_index(drop=True)
            y_syn = y_syn[finite_syn].reset_index(drop=True)
        if len(X_aug) <= len(X_tr):
            logger.warning(f"  [great] seed={seed}: too few valid rows after "
                           f"filtering — skipping.")
            continue
        # ─────────────────────────────────────────────────────────────────────

        save_augmented_data(X_aug, y_aug, X_syn, y_syn, method_dir, seed)

        # ── Train & Evaluate (regressors) ─────────────────────────────────────
        try:
            rf = RandomForestRegressor(**RF_PARAMS)
            rf.fit(X_aug, y_aug)

            xgb = XGBRegressor(**XGB_PARAMS)
            xgb.fit(X_aug, y_aug)

            rf_pred  = rf.predict(X_te)
            xgb_pred = xgb.predict(X_te)

            m_rf  = reg_metrics(y_te, rf_pred,  "rf",  METHOD)
            m_xgb = reg_metrics(y_te, xgb_pred, "xgb", METHOD)

            reg_rf  = reg_metrics_regional(y_te, rf_pred,  rare_mask, "rf",  METHOD)
            reg_xgb = reg_metrics_regional(y_te, xgb_pred, rare_mask, "xgb", METHOD)
            for d, extra in [(m_rf, reg_rf), (m_xgb, reg_xgb)]:
                for k, v in extra.items():
                    if k not in ("model", "method"):
                        d[k] = v
            m_rf["seed"]  = seed; m_rf["dataset"]  = dataset
            m_xgb["seed"] = seed; m_xgb["dataset"] = dataset

            save_metrics_dict(m_rf,
                os.path.join(method_dir, f"metrics_rf_seed{seed}.csv"))
            save_metrics_dict(m_xgb,
                os.path.join(method_dir, f"metrics_xgb_seed{seed}.csv"))
            save_text_report(reg_residual_report(y_te, rf_pred, rare_mask),
                os.path.join(method_dir, f"residual_report_rf_seed{seed}.txt"))
            save_text_report(reg_residual_report(y_te, xgb_pred, rare_mask),
                os.path.join(method_dir, f"residual_report_xgb_seed{seed}.txt"))

            logger.info(f"  [great] RF  RMSE={m_rf['rmse']:.4f}  R2={m_rf['r2']:.4f}")
            logger.info(f"  [great] XGB RMSE={m_xgb['rmse']:.4f}  R2={m_xgb['r2']:.4f}")

        except Exception as train_err:
            logger.error(f"  [great] seed={seed} train/eval failed: {train_err}",
                         exc_info=True)
            print(f"  [GReaT] seed={seed} train error: {train_err}", flush=True)
            continue

        # ── Distribution metrics + plots (parity with inline-method cells) ────
        try:
            dist_m  = distribution_metrics(X_tr, X_aug, METHOD)
            kl_dict = kl_divergence(X_tr, X_aug)
            dist_m["seed"]    = seed
            dist_m["dataset"] = dataset
            save_distribution_metrics(dist_m, kl_dict, method_dir, seed)

            plot_dir = os.path.join(method_dir, "distribution_plots")
            plot_feature_distributions(X_tr, X_aug, METHOD, plot_dir)
            plot_correlation_heatmap(
                X_tr, f"Correlation - Original - {dataset}",
                os.path.join(method_dir, f"corr_original_seed{seed}.png"))
            plot_correlation_heatmap(
                X_aug, f"Correlation - Augmented ({METHOD}) - {dataset}",
                os.path.join(method_dir, f"corr_augmented_seed{seed}.png"))
            plot_correlation_diff(
                X_tr, X_aug, METHOD,
                os.path.join(method_dir, f"corr_diff_seed{seed}.png"))
            plot_ks_per_feature(
                dist_m, METHOD, dataset,
                os.path.join(method_dir, f"ks_per_feature_seed{seed}.png"))
        except Exception as dist_err:
            logger.error(f"  [great] seed={seed} distribution metrics failed: "
                         f"{dist_err}", exc_info=True)

        # ── SHAP + stability vs reg_medium.py's baseline ──────────────────────
        try:
            shap_res = shap_for_models(rf, xgb, X_te)
            base_rf, base_xgb = load_baseline_shap(dataset, seed)
            if base_rf is None:
                logger.warning(
                    f"  [great] seed={seed}: baseline SHAP not found — run "
                    f"reg_medium.py first. Stability skipped for this seed "
                    f"(metrics + distribution are saved; re-running this "
                    f"script after reg_medium.py will fill stability in).")

            for mdl, (sv, imp, _) in shap_res.items():
                save_shap_values(sv, imp, method_dir, mdl, seed)
                plot_shap_bar(imp, mdl, METHOD,
                    os.path.join(method_dir, f"shap_bar_{mdl}_seed{seed}.png"))
                plot_shap_summary(sv, X_te, mdl, METHOD,
                    os.path.join(method_dir, f"shap_summary_{mdl}_seed{seed}.png"))

                base_imp = base_rf if mdl == "rf" else base_xgb
                if base_imp is None:
                    continue
                stab = shap_stability(base_imp, imp,
                                      top_k=SHAP_TOP_K, method=METHOD)
                stab.update({"seed": seed, "dataset": dataset, "model": mdl})
                pd.DataFrame([stab]).to_csv(
                    os.path.join(method_dir,
                                 f"shap_stability_{mdl}_seed{seed}.csv"),
                    index=False)
                plot_shap_comparison(
                    base_imp, imp, mdl, METHOD,
                    os.path.join(method_dir,
                                 f"shap_comparison_{mdl}_seed{seed}.png"))
        except Exception as shap_err:
            logger.error(f"  [great] seed={seed} SHAP failed: {shap_err}",
                         exc_info=True)

        done_now = sum(1 for s in SEEDS if is_done(dataset, s))
        logger.info(f"  [{dataset}/great] {done_now}/{len(SEEDS)} seeds complete")
        print(f"  [{dataset}/great] seed={seed} complete", flush=True)

    final = sum(1 for s in SEEDS if is_done(dataset, s))
    logger.info(f"  {dataset}/great DONE: {final}/{len(SEEDS)} seeds")


def main():
    parser = argparse.ArgumentParser(
        description="reg_great_medium.py — GReaT augmentation for medium "
                    "regression datasets",
        epilog="Run AFTER reg_medium.py. Examples:\n"
               "    python reg_great_medium.py\n"
               "    python reg_great_medium.py --dataset 529_pollen\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", nargs="+",
                        choices=ALL_DATASETS, default=ALL_DATASETS)
    args = parser.parse_args()

    log_path = os.path.join(OUTPUT_ROOT, TASK, SIZE, "great_run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path, name="great_reg_medium")

    banner("GReaT REGRESSION MEDIUM — Dedicated GReaT Run")
    logger.info(f"Datasets : {args.dataset}")
    logger.info(f"Epochs   : {GREAT_EPOCHS[SIZE]}")
    logger.info(f"Time limit per dataset: {MAX_HOURS_PER_DATASET}h "
                f"(session guard {SESSION_LIMIT_HOURS}h)")

    for ds in args.dataset:
        if not session_ok(logger):
            break
        try:
            run_great_dataset(ds, logger)
        except Exception as e:
            logger.error(f"Dataset {ds} failed: {e}", exc_info=True)
            print(f"  ERROR in {ds}: {e}", flush=True)

    banner("GReaT REGRESSION MEDIUM — COMPLETE")
    logger.info("Done. Re-run reg_medium.py once to fold great into the "
                "comparison summaries.")


if __name__ == "__main__":
    main()