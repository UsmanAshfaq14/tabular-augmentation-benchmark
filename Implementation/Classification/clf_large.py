"""
clf_large.py
=============================================================================
Classification — Large Datasets
Datasets  : mushroom (8,124) | nursery (12,960) | adult (48,842) | connect_4 (67,557)
Split     : 80 / 20  (stratified)
Seeds     : 5  (42, 123, 456, 789, 999)
Methods   : baseline | smote_nc | ctgan | tvae | tabddpm | dice
            (GReaT runs separately via great_large.py)

─────────────────────────────────────────────────────────────────────────────
SESSION PLAN  (7-hour IRIT limit → stop at 6.5h)
─────────────────────────────────────────────────────────────────────────────
  Session A  (~5.0h with GPU):
      python clf_large.py --dataset mushroom nursery

  Session B  (~4.5h with GPU):
      python clf_large.py --dataset adult

  Session C  (~5.5h with GPU):
      python clf_large.py --dataset connect_4

If a session disconnects: re-run the same command.  The checkpoint system
detects every completed (dataset, seed, method) and skips it automatically.

─────────────────────────────────────────────────────────────────────────────
SUBSAMPLING (new in clf_large)
─────────────────────────────────────────────────────────────────────────────
adult (38.9K train) and connect_4 (54.1K train) exceed SUBSAMPLE_LIMIT.
For these, ALL methods (including baseline) train on a stratified subsample
of SUBSAMPLE_LIMIT rows drawn from the training split.  This keeps the
comparison fair: every method sees the same amount of training data.

GReaT uses GREAT_SUBSAMPLE_LIMIT (≤ 10 000 rows) because GPT-2 fine-tuning
does not scale well beyond that on a 7-hour session.

─────────────────────────────────────────────────────────────────────────────
N/A METHODS (pre-computed — applied from the first run)
─────────────────────────────────────────────────────────────────────────────
  mushroom   : smote_nc  → all 22 features are categorical (no continuous)
  nursery    : smote_nc  → all 8  features are categorical
               dice      → 5-class all-categorical space (0 CFs expected)
  adult      : (none)    → mixed features, binary target
  connect_4  : smote_nc  → all 42 features are categorical
               dice      → 3-class all-categorical space (0 CFs expected)

─────────────────────────────────────────────────────────────────────────────
ALL clf_medium BUGS FIXED FROM DAY 1
─────────────────────────────────────────────────────────────────────────────
  1. Label normalisation  → XGBoost 3.x needs {0,…,n-1} (nursery=5-class,
                            connect_4=3-class)
  2. SMOTE-NC all-cat     → aug_methods.py detects 0 continuous features
  3. DiCE all-cat multi   → graceful skip when 0 CFs generated
  4. SDV dtype reversal   → _combine() in aug_methods.py coerces strings back
  5. GReaT separate       → great_large.py (never blocks the main pipeline)
  6. GReaT invalid y      → invalid y filter in great_large.py
  7. is_done RF+XGB       → great_large.py checks both metric files
  8. cat_cols kwarg       → apply_great() called without cat_cols/cont_cols
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
from utils.aug_methods   import run_augmentation
from utils.evaluation    import (clf_metrics, clf_report_str,
                                  distribution_metrics, kl_divergence, shap_stability)
from utils.shap_analysis import shap_for_models
from utils.plotting      import (plot_feature_distributions, plot_correlation_heatmap,
                                  plot_correlation_diff, plot_shap_summary, plot_shap_bar,
                                  plot_shap_comparison, plot_performance_comparison,
                                  plot_ks_per_feature, plot_shap_stability_heatmap)
from utils.io_utils      import (make_output_dirs, setup_logger, save_splits,
                                  save_augmented_data, save_metrics_dict, save_shap_values,
                                  save_text_report, save_distribution_metrics,
                                  save_experiment_config, banner)

# ═══════════════════════════════════════════════════════════════════
# GPU DETECTION
# ═══════════════════════════════════════════════════════════════════
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME      = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "None"
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME      = "torch not installed"

# ═══════════════════════════════════════════════════════════════════
# SIZE-SPECIFIC CONFIG
# ═══════════════════════════════════════════════════════════════════
SIZE         = "large"
TASK         = "classification"
TEST_SIZE    = 0.20
ALL_DATASETS = ["mushroom", "nursery", "adult", "connect_4"]

# Subsampling thresholds
SUBSAMPLE_LIMIT       = 15_000   # any dataset with n_train > this is subsampled
GREAT_SUBSAMPLE_LIMIT = 10_000   # GReaT specifically (handled in great_large.py)

# Hyperparameters
GAN_EPOCHS_L       = 300
DIFFUSION_STEPS_L  = 15_000
SMOTE_K_L          = 5

# ── Pre-computed N/A methods per dataset ─────────────────────────────────────
# Applied from the very first run — no silent skips, no misleading log entries.
NA_METHODS = {
    "mushroom":  ["smote_nc", "dice"],   # 22 categorical, 0 continuous; DiCE=0 CFs confirmed
    "nursery":   ["smote_nc", "dice"],   # 8  categorical, 0 continuous; 5-class
    "adult":     [],                      # mixed features, binary target
    "connect_4": ["smote_nc", "dice"],   # 42 categorical, 0 continuous; 3-class
}

# GReaT excluded from main pipeline — runs via great_large.py
AUG_METHODS = ["smote_nc", "ctgan", "tvae", "tabddpm", "dice"]

CFG = {
    "size": SIZE, "task": TASK, "test_size": TEST_SIZE,
    "datasets": ALL_DATASETS, "seeds": SEEDS, "methods": AUG_METHODS,
    "gpu": GPU_NAME,
    "gan_epochs": GAN_EPOCHS_L, "diffusion_steps": DIFFUSION_STEPS_L,
    "smote_k": SMOTE_K_L,
    "subsample_limit": SUBSAMPLE_LIMIT,
}

RF_PARAMS  = {"n_estimators": 100, "random_state": 42, "n_jobs": 10}
XGB_PARAMS = {"n_estimators": 100, "random_state": 42,
               "eval_metric": "logloss", "verbosity": 0}
if GPU_AVAILABLE:
    XGB_PARAMS["tree_method"] = "hist"
    XGB_PARAMS["device"]      = "cuda"

ESTIMATED_HOURS = {
    "mushroom":  2.0 if GPU_AVAILABLE else 4.0,
    "nursery":   3.0 if GPU_AVAILABLE else 6.0,
    "adult":     4.5 if GPU_AVAILABLE else 10.0,
    "connect_4": 5.5 if GPU_AVAILABLE else 12.0,
}

# ═══════════════════════════════════════════════════════════════════
# SESSION TIME GUARD  (7-hour IRIT limit → stop at 6.5h)
# ═══════════════════════════════════════════════════════════════════
SESSION_START       = time.time()
SESSION_LIMIT_HOURS = 6.5

def session_elapsed_h():
    return (time.time() - SESSION_START) / 3600.0

def session_ok(logger=None):
    elapsed   = session_elapsed_h()
    remaining = SESSION_LIMIT_HOURS - elapsed
    if remaining <= 0:
        msg = f"SESSION LIMIT REACHED ({elapsed:.1f}h). Stopping — re-run to resume."
        (logger.warning if logger else print)(msg)
        return False
    if remaining < 0.5:
        (logger.warning if logger else print)(
            f"Session budget low: {remaining*60:.0f} min remaining.")
    return True

# ═══════════════════════════════════════════════════════════════════
# CHECKPOINT / RESUME
# ═══════════════════════════════════════════════════════════════════
def is_completed(dataset, method, seed, model="rf"):
    path = os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset, method,
                        f"metrics_{model}_seed{seed}.csv")
    return os.path.exists(path)

def count_completed(dataset):
    na      = NA_METHODS.get(dataset, [])
    active  = [m for m in AUG_METHODS if m not in na]
    total   = (len(active) + 1) * len(SEEDS)   # +1 for baseline
    completed = sum(1 for m in ["baseline"] + active
                    for s in SEEDS if is_completed(dataset, m, s))
    return completed, total

def load_saved(dataset, method, seed, rf_list, xgb_list):
    for model, store in [("rf", rf_list), ("xgb", xgb_list)]:
        f = os.path.join(OUTPUT_ROOT, TASK, SIZE, dataset, method,
                         f"metrics_{model}_seed{seed}.csv")
        if os.path.exists(f):
            store.append(pd.read_csv(f).iloc[0].to_dict())

# ═══════════════════════════════════════════════════════════════════
# SUBSAMPLING
# ═══════════════════════════════════════════════════════════════════
def subsample_if_needed(X_train, y_train, seed, limit=SUBSAMPLE_LIMIT, logger=None):
    """
    If n_train > limit, take a stratified random subsample of `limit` rows.
    Applied to ALL methods (including baseline) for fair comparison.
    Returns (X_sub, y_sub, was_subsampled).
    """
    n = len(X_train)
    if n <= limit:
        return X_train, y_train, False

    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train,
        train_size=limit,
        stratify=y_train,
        random_state=seed,
    )
    msg = (f"  Subsampled training data: {n:,} → {limit:,} rows "
           f"(stratified, seed={seed})")
    if logger:
        logger.info(msg)
    else:
        print(msg, flush=True)
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True), True

# ═══════════════════════════════════════════════════════════════════
# HELPER — n_generate
# ═══════════════════════════════════════════════════════════════════
def compute_n_generate_clf(y_train):
    counts = y_train.value_counts()
    n_maj  = int(counts.max())
    return max(10, sum(max(0, n_maj - int(c)) for c in counts.values))

# ═══════════════════════════════════════════════════════════════════
# CORE PIPELINE — one dataset × one seed × one method
# ═══════════════════════════════════════════════════════════════════
def run_one_method(method, X_train_enc, X_test_enc, y_train, y_test,
                   cat_cols, cont_cols, metadata,
                   paths, dataset, seed, n_cls, logger):
    """
    Run a single (dataset, seed, method) experiment cell.
    Returns (metrics_rf, metrics_xgb, dist_metrics, shap_dict) or (None,…).
    """
    method_dir = paths[method]
    na         = NA_METHODS.get(dataset, [])

    # ── Skip N/A methods cleanly ──────────────────────────────────────────────
    if method in na:
        logger.info(f"  [{method}] N/A for {dataset} — skipping.")
        return None, None, None, None

    # ── Step 1: augment ───────────────────────────────────────────────────────
    t0 = time.time()
    if method == "baseline":
        X_aug_enc = X_train_enc.copy()
        y_aug     = y_train.copy()
    else:
        n_gen = compute_n_generate_clf(y_train)
        X_aug_enc, y_aug, X_syn, y_syn = run_augmentation(
            method, X_train_enc, y_train, cat_cols, cont_cols,
            metadata, n_generate=n_gen,
            cfg=CFG,   # CFG includes "size", "gan_epochs", "diffusion_steps", etc.
            task=TASK,
        )
        if X_aug_enc is None:
            logger.warning(f"  [{method}] Augmentation failed — skipping.")
            return None, None, None, None
        if len(X_aug_enc) <= len(X_train_enc):
            logger.warning(f"  [{method}] Added 0 rows — skipping.")
            return None, None, None, None
        save_augmented_data(X_aug_enc, y_aug, X_syn, y_syn, method_dir, seed)
        logger.info(f"  [{method}] Generated {len(X_syn)} rows in {time.time()-t0:.1f}s")

    # ── Step 2: dtype safety net + train ─────────────────────────────────────
    # Bug fix #4: SDV reverses ordinal encoding → strings appear in numeric cols
    for _col in X_aug_enc.columns:
        if X_train_enc[_col].dtype != object and X_aug_enc[_col].dtype == object:
            _fallback = int(X_train_enc[_col].mode()[0])
            X_aug_enc[_col] = (pd.to_numeric(X_aug_enc[_col], errors="coerce")
                               .fillna(_fallback)
                               .astype(X_train_enc[_col].dtype))
            logger.warning(f"  [{method}] Coerced string→numeric in '{_col}'")

    rf  = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_aug_enc, y_aug)

    xgb_p = XGB_PARAMS.copy()
    if n_cls > 2:                          # multiclass: nursery (5), connect_4 (3)
        xgb_p["objective"] = "multi:softprob"
        xgb_p["num_class"] = int(n_cls)
    xgb = XGBClassifier(**xgb_p)
    xgb.fit(X_aug_enc, y_aug)

    # ── Step 3: evaluate on held-out test set ─────────────────────────────────
    rf_pred  = rf.predict(X_test_enc);   xgb_pred  = xgb.predict(X_test_enc)
    rf_prob  = rf.predict_proba(X_test_enc); xgb_prob = xgb.predict_proba(X_test_enc)

    m_rf  = clf_metrics(y_test, rf_pred,  rf_prob,  "rf",  method)
    m_xgb = clf_metrics(y_test, xgb_pred, xgb_prob, "xgb", method)
    m_rf["seed"]  = seed;  m_rf["dataset"]  = dataset
    m_xgb["seed"] = seed;  m_xgb["dataset"] = dataset

    save_metrics_dict(m_rf,  os.path.join(method_dir, f"metrics_rf_seed{seed}.csv"))
    save_metrics_dict(m_xgb, os.path.join(method_dir, f"metrics_xgb_seed{seed}.csv"))
    save_text_report(clf_report_str(y_test, rf_pred),
                     os.path.join(method_dir, f"classification_report_rf_seed{seed}.txt"))
    save_text_report(clf_report_str(y_test, xgb_pred),
                     os.path.join(method_dir, f"classification_report_xgb_seed{seed}.txt"))
    logger.info(f"  [{method}] RF  F1={m_rf['f1_macro']:.4f}  AUC={m_rf['auc_roc']:.4f}")
    logger.info(f"  [{method}] XGB F1={m_xgb['f1_macro']:.4f}  AUC={m_xgb['auc_roc']:.4f}")

    # ── Step 4: distribution metrics (skip for baseline) ─────────────────────
    dist_result = None
    if method != "baseline":
        dist_m  = distribution_metrics(X_train_enc, X_aug_enc, method)
        kl_dict = kl_divergence(X_train_enc, X_aug_enc)
        dist_m.update({"seed": seed, "dataset": dataset})
        dist_result = {**dist_m, **{f"kl_{k}": v for k, v in kl_dict.items()}}
        save_distribution_metrics(dist_m, kl_dict, method_dir, seed)
        plot_dir = os.path.join(method_dir, "distribution_plots")
        plot_feature_distributions(X_train_enc, X_aug_enc, method, plot_dir)
        plot_ks_per_feature(dist_m, method, dataset,
                            os.path.join(method_dir, f"ks_per_feature_seed{seed}.png"))

    # ── Step 5: SHAP (200-row subsample — consistent across all methods) ──────
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

# ═══════════════════════════════════════════════════════════════════
# DATASET PIPELINE — all seeds × all methods
# ═══════════════════════════════════════════════════════════════════
def run_dataset(dataset, logger):
    na      = NA_METHODS.get(dataset, [])
    active  = [m for m in AUG_METHODS if m not in na]
    completed, total = count_completed(dataset)

    if completed == total:
        logger.info(f"  {dataset}: ALL {total} runs already complete. Skipping.")
        return

    banner(f"Dataset: {dataset.upper()}  |  {completed}/{total} complete  "
           f"|  ~{ESTIMATED_HOURS[dataset]:.1f}h remaining")
    if na:
        logger.info(f"  N/A methods for {dataset}: {na}")

    # ── Load and inspect ──────────────────────────────────────────────────────
    try:
        df = load_dataset(dataset)
    except Exception as e:
        import traceback
        print(f"\n  !! load_dataset('{dataset}') FAILED: {e}", flush=True)
        traceback.print_exc()
        raise

    info      = inspect_dataset(df, dataset)
    cat_cols  = info["cat_cols"] + info["binary_cols"]
    cont_cols = info["cont_cols"]
    n_classes = info["n_classes"]

    if n_classes > 2:
        logger.info(f"  Multiclass: {n_classes} classes — XGB uses multi:softprob")
    if not cont_cols:
        logger.info(f"  All-categorical dataset — SMOTE-NC is N/A")

    paths = make_output_dirs(OUTPUT_ROOT, TASK, SIZE, dataset, active)

    all_metrics_rf, all_metrics_xgb, all_dist = [], [], []
    all_shap_rf, all_shap_xgb = [], []

    for seed in SEEDS:
        if not session_ok(logger): break

        # ── Check if entire seed is done ──────────────────────────────────────
        seed_all_done = all(is_completed(dataset, m, seed)
                            for m in ["baseline"] + active)
        if seed_all_done:
            for m in ["baseline"] + active:
                load_saved(dataset, m, seed, all_metrics_rf, all_metrics_xgb)
            continue

        banner(f"  Seed {seed}", char="-", width=50)

        # ── Split ─────────────────────────────────────────────────────────────
        X_train_raw, X_test_raw, y_train, y_test = split_dataset(
            df, test_size=TEST_SIZE, task=TASK, seed=seed)

        # ── Bug fix #1: label normalisation for XGBoost 3.x ──────────────────
        # nursery labels may be {0,1,2,3,4} or {1,2,3,4,5} depending on PMLB version
        # connect_4 labels may be {0,1,2} or {1,2,3}
        # XGBoost 3.x requires CONTIGUOUS 0-indexed labels {0,1,...,n-1}.
        # Handles: {1,2,3}→ non-zero start | {0,1,3,4}→ gap (nursery pattern)
        _orig_cls = sorted(y_train.unique())
        if _orig_cls != list(range(len(_orig_cls))):
            _lmap   = {c: i for i, c in enumerate(_orig_cls)}
            y_train = y_train.map(_lmap).astype(int)
            y_test  = y_test.map(_lmap).astype(int)
            logger.info(f"  Labels normalised: {_orig_cls} → "
                        f"{list(range(len(_orig_cls)))} (0-indexed contiguous)")

        # ── Subsample for large datasets ──────────────────────────────────────
        X_train_sub, y_train_sub, was_subsampled = subsample_if_needed(
            X_train_raw, y_train, seed, logger=logger)

        # ── Encode (fit on subsample only — never on test) ────────────────────
        X_train_enc, X_test_enc, encoder = encode_features(
            X_train_sub, X_test_raw, cat_cols)

        # ── SDV metadata (for CTGAN / TVAE) ──────────────────────────────────
        train_df_meta = pd.concat(
            [X_train_enc, y_train_sub.rename("target")], axis=1)
        metadata = build_sdv_metadata(train_df_meta, cat_cols)

        # Save the (possibly subsampled) split
        save_splits(X_train_enc, X_test_enc, y_train_sub, y_test,
                    paths["data"], seed)

        test_hash = pd.util.hash_pandas_object(X_test_raw).sum()
        n_cls_seed = y_train_sub.nunique()

        baseline_shap_rf = baseline_shap_xgb = None

        for method in ["baseline"] + active:
            if is_completed(dataset, method, seed):
                logger.info(f"  [{method}] seed={seed}: checkpointed — skipping.")
                load_saved(dataset, method, seed, all_metrics_rf, all_metrics_xgb)
                continue

            if not session_ok(logger): break

            print(f"\n  -> [{dataset}] seed={seed}  method={method}  "
                  f"({session_elapsed_h():.1f}h elapsed)", flush=True)

            try:
                m_rf, m_xgb, dist_m, shap_d = run_one_method(
                    method, X_train_enc, X_test_enc,
                    y_train_sub, y_test,
                    cat_cols, cont_cols, metadata,
                    paths, dataset, seed, n_cls_seed, logger)
            except Exception as e:
                import traceback
                logger.error(f"  [{method}] Unexpected error: {e}", exc_info=True)
                print(f"\n  !! ERROR [{dataset}/{method}/seed{seed}]: {e}", flush=True)
                traceback.print_exc()
                continue

            if m_rf is None:
                continue

            # Verify test set was never touched
            assert pd.util.hash_pandas_object(X_test_raw).sum() == test_hash, \
                f"TEST SET MODIFIED after {method}!"

            all_metrics_rf.append(m_rf)
            all_metrics_xgb.append(m_xgb)
            if dist_m: all_dist.append(dist_m)

            if method == "baseline" and shap_d:
                baseline_shap_rf  = shap_d.get("rf")
                baseline_shap_xgb = shap_d.get("xgb")

            if method != "baseline" and shap_d and baseline_shap_rf is not None:
                for mdl_name, base_imp, aug_imp in [
                    ("rf",  baseline_shap_rf,  shap_d.get("rf")),
                    ("xgb", baseline_shap_xgb, shap_d.get("xgb")),
                ]:
                    if aug_imp is None: continue
                    stab = shap_stability(base_imp, aug_imp,
                                          top_k=SHAP_TOP_K, method=method)
                    stab.update({"seed": seed, "dataset": dataset,
                                  "model": mdl_name})
                    (all_shap_rf if mdl_name == "rf" else all_shap_xgb).append(stab)
                    plot_shap_comparison(
                        base_imp, aug_imp, mdl_name, method,
                        os.path.join(paths[method],
                                     f"shap_comparison_{mdl_name}_seed{seed}.png"))

        done_now, _ = count_completed(dataset)
        logger.info(f"  Seed {seed} complete. Dataset: {done_now}/{total}. "
                    f"Session: {session_elapsed_h():.1f}h elapsed.")

    # ── Aggregate and plot ────────────────────────────────────────────────────
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
                df_m    = pd.DataFrame(recs)
                avg_recs = []
                for mth in ["baseline"] + active:
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
        plot_shap_stability_heatmap(
            all_shap_rf,  dataset, "rf",
            os.path.join(cmp_dir, "shap_stability_heatmap_rf.png"))
        plot_shap_stability_heatmap(
            all_shap_xgb, dataset, "xgb",
            os.path.join(cmp_dir, "shap_stability_heatmap_xgb.png"))

    final_done, _ = count_completed(dataset)
    logger.info(f"  {dataset} complete. {final_done}/{total} runs. "
                f"Outputs → {os.path.join(OUTPUT_ROOT, TASK, SIZE)}")

# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="clf_large.py — Classification Large Datasets",
        epilog="""
Session plan (L40S GPU, 7h limit):
  python clf_large.py --dataset mushroom nursery   # Session A (~5.0h)
  python clf_large.py --dataset adult              # Session B (~4.5h)
  python clf_large.py --dataset connect_4          # Session C (~5.5h)

If disconnected: re-run the same command — checkpoint resumes automatically.
After all sessions complete, run:
  python great_large.py                            # GReaT on all datasets
        """,
    )
    parser.add_argument("--dataset", nargs="+",
                        choices=ALL_DATASETS, default=ALL_DATASETS)
    args = parser.parse_args()

    log_path = os.path.join(OUTPUT_ROOT, TASK, SIZE, "run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path, name=f"{TASK}_{SIZE}")

    banner(f"CLASSIFICATION - LARGE DATASETS  |  Seeds: {SEEDS}")

    if GPU_AVAILABLE:
        logger.info(f"GPU DETECTED: {GPU_NAME}")
        logger.info("XGBoost: tree_method=hist + device=cuda")
    else:
        logger.info("GPU: not available — CPU mode")

    logger.info(f"Datasets this session : {args.dataset}")
    logger.info(f"Methods               : {AUG_METHODS}")
    logger.info(f"Split                 : 80/20")
    logger.info(f"GAN epochs            : {GAN_EPOCHS_L}")
    logger.info(f"TabDDPM steps         : {DIFFUSION_STEPS_L}")
    logger.info(f"Subsample limit       : {SUBSAMPLE_LIMIT:,} rows")
    logger.info(f"Session limit         : {SESSION_LIMIT_HOURS}h")
    logger.info("")

    # ── Checkpoint overview ───────────────────────────────────────────────────
    logger.info("Checkpoint status:")
    for ds in args.dataset:
        done, total = count_completed(ds)
        na  = NA_METHODS.get(ds, [])
        pct = 100 * done / total if total else 0
        logger.info(f"  {ds:<14}: {done:>3}/{total} complete ({pct:>3.0f}%)"
                    f"  ~{ESTIMATED_HOURS[ds]:.1f}h remaining"
                    + (f"  [N/A: {na}]" if na else ""))
    logger.info("")

    # ── Startup diagnostics ───────────────────────────────────────────────────
    from utils.config import DATASET_PATHS
    diag_ok = True
    for ds in args.dataset:
        p = DATASET_PATHS.get(ds, "NOT IN CONFIG")
        exists = os.path.isfile(p) if p != "NOT IN CONFIG" else False
        status = "OK" if exists else "MISSING"
        logger.info(f"  {ds:<14}: [{status}] {p}")
        if not exists:
            print(f"  !! FILE MISSING: {p}", flush=True)
            diag_ok = False
    if not diag_ok:
        print("\n  !! One or more dataset files are missing. Check paths in utils/config.py",
              flush=True)
    logger.info("")

    save_experiment_config(CFG, os.path.join(OUTPUT_ROOT, TASK, SIZE))

    for dataset in args.dataset:
        if not session_ok(logger): break
        try:
            run_dataset(dataset, logger)
        except Exception as e:
            import traceback
            logger.error(f"Dataset {dataset} failed: {e}", exc_info=True)
            print(f"\n  !! ERROR in dataset {dataset}: {e}", flush=True)
            traceback.print_exc()

    banner("CLASSIFICATION LARGE - SESSION COMPLETE")
    logger.info(f"Session ended after {session_elapsed_h():.2f}h")
    logger.info("Re-run with same command to resume any incomplete datasets.")
    logger.info(f"Outputs → {os.path.join(OUTPUT_ROOT, TASK, SIZE)}")

if __name__ == "__main__":
    main()