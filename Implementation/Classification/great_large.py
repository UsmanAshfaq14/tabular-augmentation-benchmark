"""
great_large.py
=============================================================================
GReaT-only augmentation for clf_large datasets.
Run this AFTER clf_large.py has completed all other methods.

Why separate?
  GReaT fine-tunes GPT-2. Keeping it in the main pipeline risks blocking
  the session for hours without visible progress. Running it separately
  means the main pipeline always completes even if GReaT fails.

Subsampling (new for large):
  GReaT does not scale to 38K+ rows in a 7h session.
  Training data is subsampled to GREAT_SUBSAMPLE rows (10,000) before
  fine-tuning. This is applied consistently per seed for reproducibility.

Usage:
    python great_large.py                          # all 4 datasets
    python great_large.py --dataset mushroom       # one dataset
    python great_large.py --dataset mushroom nursery adult connect_4

Timing (L40S GPU, 10K subsample):
    mushroom  : ~10 min/seed  × 5 = ~50 min
    nursery   : ~12 min/seed  × 5 = ~60 min
    adult     : ~25 min/seed  × 5 = ~2.1h
    connect_4 : ~30 min/seed  × 5 = ~2.5h

ALL clf_medium GReaT bugs fixed from the start:
  1. apply_great() called WITHOUT cat_cols/cont_cols keywords
  2. Label normalisation ({0,…,n-1} for XGBoost 3.x)
  3. Invalid y filter (e.g. y=40 for binary, y=99 for 5-class)
  4. is_done checks BOTH rf and xgb metric files (catches partial failures)
  5. Per-dataset time budget (dataset-specific, not one-size-fits-all)
=============================================================================
"""

import os, sys, time, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost                 import XGBClassifier

from utils.config        import SEEDS, OUTPUT_ROOT, SHAP_TOP_K
from utils.data_utils    import (load_dataset, inspect_dataset, encode_features,
                                  split_dataset, build_sdv_metadata)
from utils.aug_methods   import apply_great
from utils.evaluation    import clf_metrics, clf_report_str
from utils.shap_analysis import shap_for_models
from utils.plotting      import plot_shap_bar, plot_shap_summary
from utils.io_utils      import (setup_logger, save_metrics_dict,
                                  save_augmented_data, save_shap_values,
                                  save_text_report, banner)

# ── Config ────────────────────────────────────────────────────────────────────
SIZE             = "large"
TASK             = "classification"
TEST_SIZE        = 0.20
ALL_DATASETS     = ["mushroom", "nursery", "adult", "connect_4"]
METHOD           = "great"
GREAT_EPOCHS_L   = 50        # fewer epochs than medium (50 vs 75) — subsampled data
GREAT_SUBSAMPLE  = 10_000    # max training rows for GReaT fine-tuning

# Per-dataset time budgets (generous — GReaT on 10K subsample)
TIME_BUDGET = {
    "mushroom":  1.0,    # ~50 min total
    "nursery":   1.5,    # ~60 min total
    "adult":     3.0,    # ~2.1h total
    "connect_4": 3.5,    # ~2.5h total
}

RF_PARAMS  = {"n_estimators": 100, "random_state": 42, "n_jobs": 10}
XGB_PARAMS = {"n_estimators": 100, "random_state": 42,
               "eval_metric": "logloss", "verbosity": 0}

try:
    import torch
    if torch.cuda.is_available():
        XGB_PARAMS["tree_method"] = "hist"
        XGB_PARAMS["device"]      = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("GPU: not available", flush=True)
except ImportError:
    print("torch not installed", flush=True)


# ── Checkpoint ────────────────────────────────────────────────────────────────
def is_done(dataset, seed):
    """True only if BOTH rf and xgb metrics are saved for this seed."""
    return all(
        os.path.exists(os.path.join(
            OUTPUT_ROOT, TASK, SIZE, dataset, METHOD,
            f"metrics_{model}_seed{seed}.csv"))
        for model in ["rf", "xgb"]
    )


# ── Per-dataset runner ────────────────────────────────────────────────────────
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

    method_dir = os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset, METHOD)
    os.makedirs(method_dir, exist_ok=True)

    ds_start = time.time()
    budget   = TIME_BUDGET.get(dataset, 2.0)

    for seed in SEEDS:
        if is_done(dataset, seed):
            logger.info(f"  seed={seed}: already done. Skipping.")
            continue

        elapsed_ds = (time.time() - ds_start) / 3600
        if elapsed_ds > budget:
            logger.warning(
                f"  {dataset}/great: time budget ({budget}h) exceeded. "
                f"Re-run to resume from seed {seed}.")
            break

        print(f"\n  -> [{dataset}] seed={seed}  method=great", flush=True)

        # ── Split ─────────────────────────────────────────────────────────────
        X_tr_r, X_te_r, y_tr, y_te = split_dataset(
            df, test_size=TEST_SIZE, task=TASK, seed=seed)

        # Bug fix #2: normalise labels to 0-indexed for XGBoost 3.x
        _orig_classes = sorted(y_tr.unique())
        if _orig_classes != list(range(len(_orig_classes))):
            _lmap = {c: i for i, c in enumerate(_orig_classes)}
            y_tr  = y_tr.map(_lmap).astype(int)
            y_te  = y_te.map(_lmap).astype(int)
            print(f"  [Label norm] {_orig_classes} → "
                  f"{list(range(len(_orig_classes)))} (0-indexed contiguous)",
                  flush=True)

        # ── Subsample for GReaT ───────────────────────────────────────────────
        n_tr = len(X_tr_r)
        if n_tr > GREAT_SUBSAMPLE:
            X_tr_sub, _, y_tr_sub, _ = train_test_split(
                X_tr_r, y_tr,
                train_size=GREAT_SUBSAMPLE,
                stratify=y_tr,
                random_state=seed,
            )
            X_tr_sub = X_tr_sub.reset_index(drop=True)
            y_tr_sub = y_tr_sub.reset_index(drop=True)
            print(f"  [GReaT] Subsampled: {n_tr:,} → {GREAT_SUBSAMPLE:,} training rows",
                  flush=True)
        else:
            X_tr_sub, y_tr_sub = X_tr_r, y_tr

        X_tr_enc, X_te_enc, _ = encode_features(X_tr_sub, X_te_r, cat_cols)

        # Compute n_generate from the subsample
        counts = y_tr_sub.value_counts()
        n_gen  = max(10, sum(max(0, int(counts.max()) - int(c))
                             for c in counts.values))

        # ── Run GReaT ─────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            print(f"  [GReaT] Training ({GREAT_EPOCHS_L} epochs, "
                  f"n={len(X_tr_sub):,})...", flush=True)
            # Bug fix #1: apply_great does NOT accept cat_cols / cont_cols
            X_aug, y_aug, X_syn, y_syn = apply_great(
                X_tr_enc, y_tr_sub,
                n_generate=n_gen,
                epochs=GREAT_EPOCHS_L,
            )
        except Exception as e:
            logger.error(f"  [great] seed={seed} failed: {e}", exc_info=True)
            print(f"  [GReaT] Error: {e}", flush=True)
            continue

        if X_aug is None or len(X_aug) <= len(X_tr_enc):
            logger.warning(f"  [great] seed={seed}: generated 0 rows — skipping.")
            continue

        logger.info(f"  [great] Generated {len(X_syn)} rows in {time.time()-t0:.1f}s")

        # Bug fix #3: filter invalid GReaT-generated target values
        # (GPT-2 can confuse feature values with the target — e.g. y=40 on churn,
        #  or generate unseen class labels on multiclass datasets)
        valid_classes  = set(y_tr_sub.unique())
        valid_mask_aug = y_aug.isin(valid_classes)
        valid_mask_syn = y_syn.isin(valid_classes)
        if not valid_mask_aug.all():
            n_bad    = int((~valid_mask_aug).sum())
            bad_vals = sorted(y_aug[~valid_mask_aug].unique().tolist())
            print(f"  [GReaT] Removed {n_bad} invalid y rows: {bad_vals}", flush=True)
            logger.warning(f"  [great] seed={seed}: removed {n_bad} rows "
                           f"with invalid y values {bad_vals}")
            X_aug = X_aug[valid_mask_aug].reset_index(drop=True)
            y_aug = y_aug[valid_mask_aug].reset_index(drop=True)
            if hasattr(X_syn, "__len__"):
                X_syn = X_syn[valid_mask_syn].reset_index(drop=True)
            y_syn = y_syn[valid_mask_syn].reset_index(drop=True)

        if len(X_aug) <= len(X_tr_enc):
            logger.warning(f"  [great] seed={seed}: too few valid rows — skipping.")
            continue

        save_augmented_data(X_aug, y_aug, X_syn, y_syn, method_dir, seed)

        # ── Train & evaluate ──────────────────────────────────────────────────
        try:
            rf  = RandomForestClassifier(**RF_PARAMS)
            rf.fit(X_aug, y_aug)

            xgb_p = XGB_PARAMS.copy()
            if n_cls > 2:
                xgb_p["objective"] = "multi:softprob"
                xgb_p["num_class"] = int(n_cls)
            xgb = XGBClassifier(**xgb_p)
            xgb.fit(X_aug, y_aug)

            rf_pred  = rf.predict(X_te_enc)
            xgb_pred = xgb.predict(X_te_enc)
            rf_prob  = rf.predict_proba(X_te_enc)
            xgb_prob = xgb.predict_proba(X_te_enc)

            m_rf  = clf_metrics(y_te, rf_pred,  rf_prob,  "rf",  METHOD)
            m_xgb = clf_metrics(y_te, xgb_pred, xgb_prob, "xgb", METHOD)
            m_rf["seed"]  = seed;  m_rf["dataset"]  = dataset
            m_xgb["seed"] = seed;  m_xgb["dataset"] = dataset

            save_metrics_dict(m_rf,
                os.path.join(method_dir, f"metrics_rf_seed{seed}.csv"))
            save_metrics_dict(m_xgb,
                os.path.join(method_dir, f"metrics_xgb_seed{seed}.csv"))
            save_text_report(clf_report_str(y_te, rf_pred),
                os.path.join(method_dir, f"classification_report_rf_seed{seed}.txt"))
            save_text_report(clf_report_str(y_te, xgb_pred),
                os.path.join(method_dir, f"classification_report_xgb_seed{seed}.txt"))

            logger.info(f"  [great] RF  F1={m_rf['f1_macro']:.4f}  "
                        f"AUC={m_rf['auc_roc']:.4f}")
            logger.info(f"  [great] XGB F1={m_xgb['f1_macro']:.4f}  "
                        f"AUC={m_xgb['auc_roc']:.4f}")

        except Exception as train_err:
            logger.error(f"  [great] seed={seed} train/eval failed: {train_err}",
                         exc_info=True)
            print(f"  [GReaT] seed={seed} train error: {train_err}", flush=True)
            continue   # don't save as complete; is_done will return False

        # ── SHAP ──────────────────────────────────────────────────────────────
        shap_res = shap_for_models(rf, xgb, X_te_enc)
        for mdl, (sv, imp, _) in shap_res.items():
            save_shap_values(sv, imp, method_dir, mdl, seed)
            plot_shap_bar(imp, mdl, METHOD,
                          os.path.join(method_dir,
                                       f"shap_bar_{mdl}_seed{seed}.png"))
            plot_shap_summary(sv, X_te_enc, mdl, METHOD,
                              os.path.join(method_dir,
                                           f"shap_summary_{mdl}_seed{seed}.png"))

        done_now = sum(1 for s in SEEDS if is_done(dataset, s))
        logger.info(f"  [{dataset}/great] {done_now}/{len(SEEDS)} seeds complete")
        print(f"  [{dataset}/great] seed={seed} complete", flush=True)

    final = sum(1 for s in SEEDS if is_done(dataset, s))
    logger.info(f"  {dataset}/great DONE: {final}/{len(SEEDS)} seeds")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="great_large.py — GReaT augmentation for large datasets",
        epilog="""
Run AFTER clf_large.py has completed all other methods.

  python great_large.py                          # all 4 datasets
  python great_large.py --dataset mushroom       # one dataset

Timing (L40S GPU):
  mushroom  : ~50 min    nursery : ~60 min
  adult     : ~2.1h      connect_4: ~2.5h
        """,
    )
    parser.add_argument("--dataset", nargs="+",
                        choices=ALL_DATASETS, default=ALL_DATASETS)
    args = parser.parse_args()

    log_path = os.path.join(OUTPUT_ROOT, TASK, SIZE, "great_run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path, name="great_large")

    banner("GReaT LARGE — Dedicated GReaT Run")
    logger.info(f"Datasets  : {args.dataset}")
    logger.info(f"Epochs    : {GREAT_EPOCHS_L}")
    logger.info(f"Subsample : {GREAT_SUBSAMPLE:,} rows max per dataset")
    logger.info("")
    for ds in args.dataset:
        done = sum(1 for s in SEEDS if is_done(ds, s))
        logger.info(f"  {ds:<14}: {done}/{len(SEEDS)} seeds done  "
                    f"(budget: {TIME_BUDGET.get(ds, 2.0):.1f}h)")
    logger.info("")

    for ds in args.dataset:
        try:
            run_great_dataset(ds, logger)
        except Exception as e:
            logger.error(f"Dataset {ds} failed: {e}", exc_info=True)
            print(f"  ERROR in {ds}: {e}", flush=True)

    banner("GReaT LARGE — COMPLETE")
    logger.info("Done. Outputs merged with clf_large.py results.")


if __name__ == "__main__":
    main()