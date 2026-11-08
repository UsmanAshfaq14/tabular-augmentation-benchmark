"""
great_medium.py
=============================================================================
GReaT-only augmentation for clf_medium datasets.
Run this AFTER clf_medium.py has completed all other methods.

Why separate?
  GReaT fine-tunes GPT-2 which can hang without visible output in Jupyter.
  Keeping it separate means it never blocks the main pipeline.

Usage:
    python great_medium.py --dataset flare
    python great_medium.py --dataset flare cmc car churn
    python great_medium.py                        # all 4 datasets

Time limit:
    Each dataset gets a MAX_HOURS_PER_DATASET budget.
    If GReaT hasn't finished within that time, it skips to the next dataset.
    Set HF_TOKEN environment variable to avoid HuggingFace rate limits:
        export HF_TOKEN=your_token_here
=============================================================================
"""

import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.config        import (SEEDS, OUTPUT_ROOT, SHAP_TOP_K,
                                 GREAT_EPOCHS, GAN_EPOCHS, DIFFUSION_STEPS, SMOTE_K)
from utils.data_utils    import (load_dataset, inspect_dataset, encode_features,
                                 split_dataset, build_sdv_metadata)
from utils.aug_methods   import apply_great
from utils.evaluation    import clf_metrics, clf_report_str, shap_stability
from utils.shap_analysis import shap_for_models
from utils.plotting      import (plot_shap_bar, plot_shap_summary,
                                 plot_shap_comparison)
from utils.io_utils      import (make_output_dirs, setup_logger, save_metrics_dict,
                                 save_augmented_data, save_shap_values,
                                 save_text_report, banner)

# ── Config ────────────────────────────────────────────────────────────────────
SIZE         = "medium"
TASK         = "classification"
TEST_SIZE    = 0.20
ALL_DATASETS = ["flare", "cmc", "car", "churn"]
METHOD       = "great"

RF_PARAMS  = {"n_estimators": 100, "random_state": 42, "n_jobs": 10}
XGB_PARAMS = {"n_estimators": 100, "random_state": 42,
              "eval_metric": "logloss", "verbosity": 0}

# GPU-aware XGBoost
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

# Max hours per dataset — GReaT skips if it exceeds this
# Base limit 1.5h, but churn GReaT takes ~30min/seed = 2.5h for 5 seeds
MAX_HOURS_PER_DATASET = 3.0

CFG = {
    "size": SIZE, "task": TASK, "test_size": TEST_SIZE,
    "methods": [METHOD], "seeds": SEEDS,
    "great_epochs": GREAT_EPOCHS[SIZE],    # 75
    "gan_epochs":   GAN_EPOCHS[SIZE],
    "diffusion_steps": DIFFUSION_STEPS[SIZE],
    "smote_k": SMOTE_K[SIZE],
}


def is_done(dataset, seed):
    """True only if BOTH rf and xgb metrics are saved for this seed."""
    return all(
        os.path.exists(os.path.join(
            OUTPUT_ROOT, TASK, SIZE, dataset, METHOD,
            f"metrics_{model}_seed{seed}.csv"))
        for model in ["rf", "xgb"]
    )


def run_great_dataset(dataset, logger):
    done = sum(1 for s in SEEDS if is_done(dataset, s))
    if done == len(SEEDS):
        logger.info(f"  {dataset}/great: all {len(SEEDS)} seeds complete. Skipping.")
        return

    banner(f"GReaT — {dataset.upper()}  |  {done}/{len(SEEDS)} seeds done")

    df   = load_dataset(dataset)
    info = inspect_dataset(df, dataset)

    cat_cols  = info["cat_cols"] + info["binary_cols"]
    cont_cols = info["cont_cols"]
    n_cls     = info["n_classes"]

    # Find comparison dir (should already exist from clf_medium.py run)
    base_dir = os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset)
    method_dir = os.path.join(base_dir, METHOD)
    os.makedirs(method_dir, exist_ok=True)

    ds_start = time.time()

    for seed in SEEDS:
        if is_done(dataset, seed):
            logger.info(f"  seed={seed}: already done. Skipping.")
            continue

        # Per-dataset time budget check
        elapsed_ds = (time.time() - ds_start) / 3600
        if elapsed_ds > MAX_HOURS_PER_DATASET:
            logger.warning(
                f"  {dataset}/great: time budget ({MAX_HOURS_PER_DATASET}h) exceeded. "
                f"Stopping. Re-run to resume from seed {seed}.")
            break

        print(f"\n  -> [{dataset}] seed={seed}  method=great", flush=True)

        X_tr_r, X_te_r, y_tr, y_te = split_dataset(
            df, test_size=TEST_SIZE, task=TASK, seed=seed)
        X_tr, X_te, _ = encode_features(X_tr_r, X_te_r, cat_cols)

        # ── Normalise class labels to 0-indexed ───────────────────────────────
        # XGBoost 3.x multi:softprob requires labels in {0, 1, ..., n-1}.
        # Datasets like cmc use {1, 2, 3} natively — remap both y_tr and y_te.
        _orig_classes = sorted(y_tr.unique())
        if _orig_classes[0] != 0:
            _lmap = {c: i for i, c in enumerate(_orig_classes)}
            y_tr  = y_tr.map(_lmap).astype(int)
            y_te  = y_te.map(_lmap).astype(int)
            print(f"  [Label normalisation] {_orig_classes} → "
                  f"{sorted(y_tr.unique())}  (0-indexed for XGBoost)", flush=True)
        # ─────────────────────────────────────────────────────────────────────

        # Compute n_generate
        counts = y_tr.value_counts()
        n_gen  = max(10, sum(max(0, int(counts.max()) - int(c))
                             for c in counts.values))

        # ── Run GReaT ─────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            print("  [GReaT] Starting training...", flush=True)
            # apply_great does not need cat_cols/cont_cols —
            # GReaT serialises all columns as text automatically
            X_aug, y_aug, X_syn, y_syn = apply_great(
                X_tr, y_tr, n_generate=n_gen,
                epochs=GREAT_EPOCHS[SIZE],
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

        # ── Filter invalid GReaT-generated target values ─────────────────────
        # GReaT can generate y values outside the training class set
        # (e.g. y=40 for a binary {0,1} dataset) by confusing target with
        # other numerical context. Remove those rows before training.
        valid_classes = set(y_tr.unique())
        valid_mask_aug = y_aug.isin(valid_classes)
        valid_mask_syn = y_syn.isin(valid_classes)
        if not valid_mask_aug.all():
            n_bad = int((~valid_mask_aug).sum())
            bad_vals = sorted(y_aug[~valid_mask_aug].unique().tolist())
            print(f"  [GReaT] Removed {n_bad} rows with invalid y values: {bad_vals}",
                  flush=True)
            X_aug = X_aug[valid_mask_aug].reset_index(drop=True)
            y_aug = y_aug[valid_mask_aug].reset_index(drop=True)
            X_syn = X_syn[valid_mask_syn].reset_index(drop=True) if hasattr(X_syn, '__len__') else X_syn
            y_syn = y_syn[valid_mask_syn].reset_index(drop=True)
        if len(X_aug) <= len(X_tr):
            logger.warning(f"  [great] seed={seed}: too few valid rows after filtering — skipping.")
            continue
        # ─────────────────────────────────────────────────────────────────────

        save_augmented_data(X_aug, y_aug, X_syn, y_syn, method_dir, seed)

        # ── Train & Evaluate ──────────────────────────────────────────────────
        try:
            rf  = RandomForestClassifier(**RF_PARAMS)
            rf.fit(X_aug, y_aug)

            xgb_p = XGB_PARAMS.copy()
            if n_cls > 2:
                xgb_p["objective"] = "multi:softprob"
                xgb_p["num_class"]  = int(n_cls)
            xgb = XGBClassifier(**xgb_p)
            xgb.fit(X_aug, y_aug)

            rf_pred  = rf.predict(X_te);  xgb_pred  = xgb.predict(X_te)
            rf_prob  = rf.predict_proba(X_te); xgb_prob = xgb.predict_proba(X_te)

            m_rf  = clf_metrics(y_te, rf_pred,  rf_prob,  "rf",  METHOD)
            m_xgb = clf_metrics(y_te, xgb_pred, xgb_prob, "xgb", METHOD)
            m_rf["seed"]  = seed; m_rf["dataset"]  = dataset
            m_xgb["seed"] = seed; m_xgb["dataset"] = dataset

            save_metrics_dict(m_rf,
                os.path.join(method_dir, f"metrics_rf_seed{seed}.csv"))
            save_metrics_dict(m_xgb,
                os.path.join(method_dir, f"metrics_xgb_seed{seed}.csv"))
            save_text_report(clf_report_str(y_te, rf_pred),
                os.path.join(method_dir, f"classification_report_rf_seed{seed}.txt"))
            save_text_report(clf_report_str(y_te, xgb_pred),
                os.path.join(method_dir, f"classification_report_xgb_seed{seed}.txt"))

            logger.info(f"  [great] RF  F1={m_rf['f1_macro']:.4f}  AUC={m_rf['auc_roc']}")
            logger.info(f"  [great] XGB F1={m_xgb['f1_macro']:.4f}  AUC={m_xgb['auc_roc']}")

        except Exception as train_err:
            logger.error(f"  [great] seed={seed} train/eval failed: {train_err}",
                         exc_info=True)
            print(f"  [GReaT] seed={seed} train error: {train_err}", flush=True)
            continue   # skip to next seed, don't save as complete

        # ── SHAP ──────────────────────────────────────────────────────────────
        shap_res = shap_for_models(rf, xgb, X_te)
        for mdl, (sv, imp, _) in shap_res.items():
            save_shap_values(sv, imp, method_dir, mdl, seed)
            plot_shap_bar(imp, mdl, METHOD,
                os.path.join(method_dir, f"shap_bar_{mdl}_seed{seed}.png"))
            plot_shap_summary(sv, X_te, mdl, METHOD,
                os.path.join(method_dir, f"shap_summary_{mdl}_seed{seed}.png"))

        done_now = sum(1 for s in SEEDS if is_done(dataset, s))
        logger.info(f"  [{dataset}/great] {done_now}/{len(SEEDS)} seeds complete")
        print(f"  [{dataset}/great] seed={seed} complete", flush=True)

    final = sum(1 for s in SEEDS if is_done(dataset, s))
    logger.info(f"  {dataset}/great DONE: {final}/{len(SEEDS)} seeds")


def main():
    parser = argparse.ArgumentParser(
        description="great_medium.py — GReaT augmentation for medium datasets",
        epilog="""
Run AFTER clf_medium.py has completed all other methods.

Examples:
    python great_medium.py                    # all 4 datasets
    python great_medium.py --dataset flare    # one dataset
    python great_medium.py --dataset flare cmc

Set HF_TOKEN to avoid HuggingFace rate limits:
    export HF_TOKEN=your_token
    python great_medium.py
        """
    )
    parser.add_argument("--dataset", nargs="+",
                        choices=ALL_DATASETS, default=ALL_DATASETS)
    args = parser.parse_args()

    log_path = os.path.join(OUTPUT_ROOT, TASK, SIZE, "great_run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path, name="great_medium")

    banner("GReaT MEDIUM — Dedicated GReaT Run")
    logger.info(f"Datasets : {args.dataset}")
    logger.info(f"Epochs   : {GREAT_EPOCHS[SIZE]}")
    logger.info(f"Time limit per dataset: {MAX_HOURS_PER_DATASET}h")
    logger.info("")

    for ds in args.dataset:
        try:
            run_great_dataset(ds, logger)
        except Exception as e:
            logger.error(f"Dataset {ds} failed: {e}", exc_info=True)
            print(f"  ERROR in {ds}: {e}", flush=True)

    banner("GReaT MEDIUM — COMPLETE")
    logger.info("Done. Outputs merged with clf_medium.py results.")


if __name__ == "__main__":
    main()