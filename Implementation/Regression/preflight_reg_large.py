"""
preflight_reg_large.py
=======================
30-second go/no-go check for the MEDIUM regression phase. Run on the server
BEFORE launching reg_large.py:

    python preflight_reg_large.py

For each of the four medium datasets this script verifies the file loads
through the real pipeline code (utils.data_utils / utils.evaluation), then
reports:

  - shape + feature-type mix (binary / categorical / continuous)
  - target statistics (mean, std, range, skew, #unique values)
  - the 80/20 split sizes
  - the rare-region rule chosen and the rare-test count FOR EVERY SEED
    (boxplot_iqr vs decile_tail — and whether the rare region is EMPTY,
    which is expected for titanic and 294_satellite_image: their targets
    are binary / bounded-discrete with >=10% mass at both extremes, so they
    act as degenerate-target control cases and their rare metrics are NaN)
  - the +20% synthetic row count (n_generate) and the DiCE query-pool size
  - method-applicability notes (SMOGN/DiCE need >=1 continuous feature)

Nothing is written to disk. If every dataset prints OK (warnings about empty
rare regions are informational, not failures), launch:

    python reg_large.py            # baseline + smogn/ctgan/tvae/tabddpm/dice
    python reg_great_large.py      # GReaT (after reg_large.py)
    python reg_large.py            # once more: folds great into summaries
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from utils.config     import SEEDS, REG_AUG_PCT, SMOTE_K, GAN_EPOCHS, \
                             DIFFUSION_STEPS, GREAT_EPOCHS, LARGE_SUBSAMPLE_N
from utils.data_utils import load_dataset, inspect_dataset, split_dataset
from utils.evaluation import compute_rarity_mask

SIZE      = "large"
TEST_SIZE = 0.20
DATASETS  = ["225_puma8NH", "197_cpu_act", "537_houses", "344_mv"]

DICE_NUM_CFS = 3   # mirrors utils.config


def check(dataset):
    print("=" * 72)
    print(f"  {dataset}")
    print("=" * 72)
    try:
        df = load_dataset(dataset)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return False

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        info = inspect_dataset(df, dataset)

    y = df["target"].astype(float)
    n_unique = int(y.nunique())
    print(f"  shape      : {df.shape[0]} rows x {df.shape[1]-1} features "
          f"(binary={len(info['binary_cols'])} cat={len(info['cat_cols'])} "
          f"cont={len(info['cont_cols'])})  missing={info['missing']}")

    # Missing-value report + imputation preview (the pipeline imputes per
    # split, fit on TRAIN only — median for continuous, mode for cat/binary).
    n_y_missing = int(df["target"].isna().sum())
    if n_y_missing:
        print(f"  MISSING    : {n_y_missing} rows have a missing TARGET — "
              f"these rows are dropped at load.")
    feat_miss = df.drop(columns=["target"]).isna().sum()
    feat_miss = feat_miss[feat_miss > 0]
    if len(feat_miss):
        cat_set = set(info["cat_cols"]) | set(info["binary_cols"])
        plan = {}
        for col, k in feat_miss.items():
            vals = df[col].dropna()
            fill = (vals.mode().iloc[0] if col in cat_set
                    else float(vals.median()))
            plan[col] = f"{int(k)} -> {round(fill, 3)}"
        print(f"  MISSING    : {int(feat_miss.sum())} feature cells will be "
              f"IMPUTED per split (train-fitted median/mode). "
              f"Per column (count -> fill preview): {plan}")
    print(f"  target     : mean={y.mean():.4f} std={y.std():.4f} "
          f"range=[{y.min():.4g}, {y.max():.4g}] skew={y.skew():+.3f} "
          f"unique={n_unique}")

    n_train_full = int(round(len(df) * (1 - TEST_SIZE)))
    n_test  = len(df) - n_train_full
    # LARGE: inline methods cap training at LARGE_SUBSAMPLE_N (15k), GReaT at 10k
    n_train = min(n_train_full, LARGE_SUBSAMPLE_N)
    n_great = min(n_train_full, 10_000)
    n_gen   = max(10, int(round(REG_AUG_PCT * n_train)))
    n_query = max(8, -(-n_gen // DICE_NUM_CFS))
    sub = ""
    if n_train_full > LARGE_SUBSAMPLE_N:
        sub = f"  -> SUBSAMPLED to {LARGE_SUBSAMPLE_N:,} (inline) / {n_great:,} (GReaT)"
    print(f"  split 80/20: train~{n_train_full:,}  test~{n_test:,}{sub}")
    print(f"  effective train: inline~{n_train:,}  great~{n_great:,}   "
          f"n_generate(+20%)~{n_gen:,}   DiCE n_query~{n_query:,}")

    # Rare-region rule for every seed (defined on TRAIN target, no leakage)
    empty_all = True
    for seed in SEEDS:
        _, _, y_tr, y_te = split_dataset(df, test_size=TEST_SIZE,
                                         task="regression", seed=seed)
        mask, ri = compute_rarity_mask(y_tr, y_te)
        n_rare = int(mask.sum())
        empty_all &= (n_rare == 0)
        print(f"    seed {seed:>3}: rule={ri['rarity_rule']:<12} "
              f"thresholds=[{ri['low_threshold']}, {ri['high_threshold']}]  "
              f"rare_test={n_rare}/{len(y_te)} "
              f"({100*n_rare/len(y_te):.1f}%)")

    notes = []
    if len(info["cont_cols"]) == 0:
        notes.append("no continuous features -> SMOGN N/A (and DiCE will "
                     "likely yield no counterfactuals); both skip gracefully")
    if n_unique <= 2:
        notes.append("binary target -> degenerate-target CONTROL case")
    if empty_all:
        notes.append("rare region EMPTY on every seed -> rare_rmse/rare_r2 "
                     "will be NaN by design; overall metrics, distribution "
                     "fidelity and SHAP stability are unaffected")
    elif n_unique <= 10:
        notes.append(f"near-discrete target ({n_unique} values) -> "
                     f"decile-tail rule, same handling as SWD in small phase")
    for n in notes:
        print(f"  NOTE       : {n}")
    print(f"  STATUS     : OK")
    return True


def main():
    print(f"PREFLIGHT — regression/{SIZE}  "
          f"(GAN_EPOCHS={GAN_EPOCHS[SIZE]}  "
          f"DIFFUSION_STEPS={DIFFUSION_STEPS[SIZE]}  "
          f"GREAT_EPOCHS={GREAT_EPOCHS[SIZE]}  SMOGN_k={SMOTE_K[SIZE]}  "
          f"split=80/20  aug=+{int(REG_AUG_PCT*100)}%)\n")
    ok = all([check(d) for d in DATASETS])
    print("=" * 72)
    if ok:
        print("ALL DATASETS LOAD AND PROFILE CLEANLY.")
        print("Expect rare-region analysis to rest on all four continuous targets;")
        print("datasets above 15k train rows are subsampled (537_houses, 344_mv).")
        print("Launch:  python reg_large.py")
    else:
        print("AT LEAST ONE DATASET FAILED TO LOAD — fix before launching.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
