"""
fix_mushroom_shap_stability.py
=============================================================================
Regenerates the 42 SHAP stability files missing for mushroom.

Why they are missing:
  Mushroom baseline ran in Session 1 (before bug fixes).
  Augmentation methods ran in Session 2 (after fixes, via checkpoint).
  In Session 2, baseline was checkpointed/skipped so `baseline_shap_rf`
  was never loaded into memory. SHAP stability requires comparing
  baseline vs augmented SHAP arrays in-memory during the run.
  All raw .npy arrays ARE on disk — we just need to recompute.

What this script generates:
  Per-method, per-seed (4 methods × 5 seeds × 2 models = 40 plots):
    mushroom/{method}/shap_comparison_{model}_seed{seed}.png

  Per-dataset comparison (2 plots + 2 CSVs):
    mushroom/comparison/shap_stability_heatmap_rf.png
    mushroom/comparison/shap_stability_heatmap_xgb.png
    mushroom/comparison/all_methods_shap_stability_rf.csv   (may already exist)
    mushroom/comparison/all_methods_shap_stability_xgb.csv  (may already exist)

Runtime: < 2 minutes on CPU, < 30s with GPU-accelerated matplotlib.

Usage:
    python fix_mushroom_shap_stability.py
    (Run from the Implementation/ directory on the IRIT server)
=============================================================================
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import OUTPUT_ROOT, SHAP_TOP_K

# ── Inline plot_shap_comparison ───────────────────────────────────────────────
# Re-implemented here to avoid the 'nlargest' bug when importance is ndarray.
def plot_shap_comparison(bl_imp, aug_imp, model, method, out_path, top_k=10):
    """Bar chart: baseline vs augmented top-K SHAP feature importance."""
    bl_s  = pd.Series(bl_imp)
    aug_s = pd.Series(aug_imp)
    n     = min(top_k, len(bl_s))
    idx   = bl_s.nlargest(n).index      # top-K by baseline importance
    bl_top  = bl_s.iloc[idx].values
    aug_top = aug_s.iloc[idx].values
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, bl_top,  0.4, label="Baseline", color="#4A90D9", alpha=0.85)
    ax.bar(x + 0.2, aug_top, 0.4, label=method.upper(), color="#E8974A", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i}" for i in idx], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("|SHAP|")
    ax.set_title(f"SHAP comparison — mushroom / {method} / {model}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

# ── Inline plot_shap_stability_heatmap ───────────────────────────────────────
# Re-implemented to avoid FixedLocator matplotlib version bug.
def plot_shap_stability_heatmap(records, dataset, model, out_path):
    """Heatmap: Spearman ρ and Top-5 overlap per method (mean over seeds)."""
    df = pd.DataFrame(records)
    methods = sorted(df["method"].unique())
    spear   = [df[df["method"]==m]["spearman_corr"].mean() for m in methods]
    top5    = [df[df["method"]==m]["top5_overlap"].mean()  for m in methods]

    data = np.array([spear, top5])          # shape (2, n_methods)
    fig, ax = plt.subplots(figsize=(max(5, len(methods)*1.2 + 1), 3))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Spearman ρ", "Top-5 overlap"])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_title(f"SHAP stability — {dataset} / {model}")
    for j, m in enumerate(methods):
        ax.text(j, 0, f"{spear[j]:.3f}", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(j, 1, f"{top5[j]:.3f}",  ha="center", va="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

# ── Config ────────────────────────────────────────────────────────────────────
SIZE    = "large"
TASK    = "classification"
DATASET = "mushroom"
AUG_METHODS = ["ctgan", "tvae", "tabddpm", "great"]
SEEDS   = [42, 123, 456, 789, 999]
MODELS  = ["rf", "xgb"]
TOP_K   = SHAP_TOP_K

BASE = os.path.join(OUTPUT_ROOT, TASK, SIZE, DATASET)
print(f"Base directory: {BASE}")
print(f"Generating SHAP stability for: {DATASET}")
print(f"Methods: {AUG_METHODS}")
print(f"Seeds:   {SEEDS}")
print()


def load_importance(method, model, seed):
    """Load shap_importance CSV and return the importance array."""
    f = os.path.join(BASE, method, f"shap_importance_{model}_seed{seed}.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing: {f}")
    df = pd.read_csv(f)
    # importance CSV has columns: [feature, importance] or similar
    # Take the numeric column (second column)
    return df.iloc[:, 1].values.astype(float)


def shap_stability_from_arrays(bl_imp, aug_imp, top_k=5):
    """Compute Spearman ρ and Top-K overlap between baseline and augmented."""
    rho, _ = spearmanr(bl_imp, aug_imp)
    n_top   = min(top_k, len(bl_imp))
    top_bl  = set(np.argsort(bl_imp)[-n_top:])
    top_aug = set(np.argsort(aug_imp)[-n_top:])
    top_k_ov = len(top_bl & top_aug) / n_top
    return float(rho), float(top_k_ov)


# ── Main loop ─────────────────────────────────────────────────────────────────
records_rf, records_xgb = [], []
total_plots = 0
errors = []

for seed in SEEDS:
    print(f"  Seed {seed}:")

    for model in MODELS:
        # Load baseline importance
        try:
            bl_imp = load_importance("baseline", model, seed)
        except FileNotFoundError as e:
            print(f"    SKIP (baseline missing): {e}")
            continue

        for method in AUG_METHODS:
            # Load augmented importance
            try:
                aug_imp = load_importance(method, model, seed)
            except FileNotFoundError as e:
                print(f"    SKIP ({method}/{model}): {e}")
                continue

            # 1. Compute stability metrics
            rho, top_k_ov = shap_stability_from_arrays(bl_imp, aug_imp, top_k=TOP_K)
            rec = {"dataset": DATASET, "method": method, "seed": seed,
                   "model": model, "spearman_corr": rho, "top5_overlap": top_k_ov}
            (records_rf if model == "rf" else records_xgb).append(rec)

            # 2. Generate shap_comparison plot
            out_png = os.path.join(BASE, method, f"shap_comparison_{model}_seed{seed}.png")
            if os.path.exists(out_png):
                print(f"    {model} {method} seed{seed}: plot already exists — skipping")
                continue

            try:
                plot_shap_comparison(
                    bl_imp, aug_imp, model, method, out_png
                )
                total_plots += 1
                print(f"    {model} {method} seed{seed}: "
                      f"ρ={rho:.4f}  top5={top_k_ov:.2f}  → plot saved")
            except Exception as e:
                errors.append(f"    ERROR generating {out_png}: {e}")
                print(f"    ERROR: {e}")

# ── Save stability CSVs ───────────────────────────────────────────────────────
cmp_dir = os.path.join(BASE, "comparison")
os.makedirs(cmp_dir, exist_ok=True)

for records, fname in [
    (records_rf,  "all_methods_shap_stability_rf.csv"),
    (records_xgb, "all_methods_shap_stability_xgb.csv"),
]:
    fpath = os.path.join(cmp_dir, fname)
    pd.DataFrame(records).to_csv(fpath, index=False)
    print(f"\n  Saved: {fname}  ({len(records)} rows)")

# ── Generate heatmap plots ────────────────────────────────────────────────────
print()
for records, model in [(records_rf, "rf"), (records_xgb, "xgb")]:
    if not records:
        print(f"  No records for {model} — heatmap skipped")
        continue
    heatmap_path = os.path.join(cmp_dir, f"shap_stability_heatmap_{model}.png")
    try:
        plot_shap_stability_heatmap(records, DATASET, model, heatmap_path)
        print(f"  Saved heatmap: shap_stability_heatmap_{model}.png")
    except Exception as e:
        errors.append(f"  ERROR heatmap {model}: {e}")
        print(f"  ERROR generating heatmap {model}: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("="*60)
print(f"  DONE")
print(f"  Comparison plots generated: {total_plots}")
print(f"  Stability CSV rows (RF):    {len(records_rf)}")
print(f"  Stability CSV rows (XGB):   {len(records_xgb)}")
print(f"  Errors:                     {len(errors)}")
if errors:
    print("  ERROR DETAILS:")
    for e in errors: print(e)
else:
    print()
    print("  All 42 missing mushroom SHAP stability files regenerated.")
    print("  Results summary (RF, mean over 5 seeds):")
    df = pd.DataFrame(records_rf)
    for m in AUG_METHODS:
        sub = df[df["method"] == m]
        if len(sub):
            print(f"    {m:<12}: ρ={sub['spearman_corr'].mean():.4f}  "
                  f"top5={sub['top5_overlap'].mean():.2f}")
print("="*60)