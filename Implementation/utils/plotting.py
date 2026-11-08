"""
plotting.py
─────────────────────────────────────────────────────────────────────
All graph generation for the research paper.
Every function saves to a given path and returns the path.
Professor point 4: every single graph must be saved.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family":    "DejaVu Serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "figure.dpi":      150,
    "savefig.dpi":     150,
    "savefig.bbox":    "tight",
})

METHODS_LABELS = {
    "baseline":  "Baseline\n(No Aug)",
    "smote_nc":  "SMOTE-NC",
    "smogn":     "SMOGN",
    "ctgan":     "CTGAN",
    "tvae":      "TVAE",
    "tabddpm":   "TabDDPM",
    "great":     "GReaT",
    "dice":      "DiCE",
}

METHODS_COLORS = {
    "baseline": "#555555",
    "smote_nc": "#2196F3",
    "smogn":    "#2196F3",
    "ctgan":    "#FF9800",
    "tvae":     "#9C27B0",
    "tabddpm":  "#03A9F4",
    "great":    "#4CAF50",
    "dice":     "#F44336",
}


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ─── 1. Feature distribution overlay ─────────────────────────────────────────

def plot_feature_distributions(X_orig: pd.DataFrame,
                                X_aug:  pd.DataFrame,
                                method: str,
                                save_dir: str) -> list:
    """
    One density/histogram overlay per feature: original vs augmented.
    Saves one PNG per feature. Returns list of saved paths.
    """
    cols  = X_orig.select_dtypes(include=[np.number]).columns.tolist()
    paths = []
    for col in cols:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(X_orig[col].dropna(), bins=30, density=True,
                alpha=0.55, label="Original", color="#1565C0", edgecolor="white")
        ax.hist(X_aug[col].dropna(),  bins=30, density=True,
                alpha=0.55, label=f"Augmented ({method})",
                color=METHODS_COLORS.get(method, "#888"), edgecolor="white")
        ax.set_title(f"{col}  —  distribution comparison ({method})")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        p = os.path.join(save_dir, f"dist_{col}.png")
        paths.append(_save(fig, p))
    return paths


# ─── 2. Correlation heatmaps ──────────────────────────────────────────────────

def plot_correlation_heatmap(X: pd.DataFrame,
                              title: str,
                              save_path: str) -> str:
    """Single correlation heatmap."""
    corr = X.select_dtypes(include=[np.number]).corr()
    n    = len(corr)
    sz   = max(5, min(n, 14))
    fig, ax = plt.subplots(figsize=(sz, sz * 0.85))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True,
                annot=n <= 12, fmt=".1f", annot_kws={"size": 7},
                linewidths=0.4, cbar_kws={"shrink": 0.8})
    ax.set_title(title, pad=10)
    return _save(fig, save_path)


def plot_correlation_diff(X_orig: pd.DataFrame,
                           X_aug:  pd.DataFrame,
                           method: str,
                           save_path: str) -> str:
    """Heatmap of (corr_orig - corr_aug) — shows where relationships changed."""
    cols      = X_orig.select_dtypes(include=[np.number]).columns
    corr_orig = X_orig[cols].corr().fillna(0)
    corr_aug  = X_aug[cols].corr().fillna(0)
    diff      = corr_orig - corr_aug
    n         = len(diff)
    sz        = max(5, min(n, 14))
    fig, ax   = plt.subplots(figsize=(sz, sz * 0.85))
    lim       = max(0.1, float(np.abs(diff.values).max()))
    sns.heatmap(diff, ax=ax, cmap="RdBu_r", center=0,
                vmin=-lim, vmax=lim, square=True,
                annot=n <= 12, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.4, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Correlation difference (Original − {method})", pad=10)
    return _save(fig, save_path)


# ─── 3. SHAP plots ────────────────────────────────────────────────────────────

def plot_shap_summary(shap_values: np.ndarray,
                      X_test: pd.DataFrame,
                      model_name: str,
                      method: str,
                      save_path: str) -> str:
    """SHAP beeswarm summary plot."""
    n = min(200, len(X_test))
    fig, ax = plt.subplots(figsize=(8, max(4, X_test.shape[1] * 0.35)))
    shap.summary_plot(
        shap_values[:n], X_test.iloc[:n],
        show=False, plot_type="dot",
        max_display=min(20, X_test.shape[1]),
    )
    plt.title(f"SHAP Summary — {model_name.upper()} — {method}", pad=8)
    return _save(plt.gcf(), save_path)


def plot_shap_bar(importance: pd.Series,
                  model_name: str,
                  method: str,
                  save_path: str,
                  top_n: int = 15) -> str:
    """Horizontal SHAP bar chart (mean |SHAP|)."""
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(top) * 0.38)))
    bars = ax.barh(range(len(top)), top.values[::-1],
                   color="#1565C0", edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance (SHAP) — {model_name.upper()} — {method}")
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    return _save(fig, save_path)


def plot_shap_comparison(imp_baseline: pd.Series,
                          imp_augmented: pd.Series,
                          model_name: str,
                          method: str,
                          save_path: str,
                          top_n: int = 10) -> str:
    """
    Side-by-side horizontal bars: baseline vs augmented feature importance.
    Directly answers RQ3 (SHAP stability).
    """
    feats   = imp_baseline.nlargest(top_n).index.tolist()
    base_v  = imp_baseline[feats].values
    aug_v   = imp_augmented.reindex(feats).fillna(0).values

    x     = np.arange(len(feats))
    w     = 0.38
    fig, ax = plt.subplots(figsize=(8, max(4, len(feats) * 0.45)))
    ax.barh(x - w/2, base_v, w, label="Baseline", color="#1565C0",
            edgecolor="white", linewidth=0.4)
    ax.barh(x + w/2, aug_v,  w, label=f"Augmented ({method})",
            color=METHODS_COLORS.get(method, "#888"),
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(x)
    ax.set_yticklabels(feats, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance: Baseline vs {method} "
                 f"({model_name.upper()})")
    ax.legend(fontsize=9)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    return _save(fig, save_path)


# ─── 4. Performance bar charts ────────────────────────────────────────────────

def plot_performance_comparison(results: list,
                                 metric: str,
                                 task: str,
                                 dataset_name: str,
                                 model_name: str,
                                 save_path: str) -> str:
    """
    Bar chart comparing one metric across all augmentation methods.
    results : list of dicts (each with 'method' and metric keys).
    """
    df   = pd.DataFrame(results)
    df   = df[df["model"] == model_name].copy()
    methods = [m for m in METHODS_LABELS if m in df["method"].values]
    vals    = [df[df["method"] == m][metric].values[0]
               if len(df[df["method"] == m]) > 0 else 0
               for m in methods]
    colors  = [METHODS_COLORS.get(m, "#888") for m in methods]
    labels  = [METHODS_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars    = ax.bar(range(len(methods)), vals, color=colors,
                     edgecolor="white", linewidth=0.5, width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(metric.upper().replace("_", " "))
    ax.set_title(f"{metric.upper()} — {dataset_name} — {model_name.upper()}")
    # Y-limits: classification metrics live in [0,1]; regression metrics
    # (rmse/mae) are unbounded above and R² can be negative, so the axis
    # must adapt to the data range rather than being capped at 1.05.
    bounded = task == "classification" and metric in (
        "f1_macro", "auc_roc", "accuracy", "precision", "recall")
    if bounded:
        ax.set_ylim(0, min(1.05, max(vals) * 1.15) if vals else 1.0)
    elif vals:
        vmin, vmax = min(vals), max(vals)
        pad = (vmax - vmin) * 0.15 if vmax > vmin else (abs(vmax) * 0.15 or 1.0)
        ax.set_ylim(min(0, vmin) - pad, vmax + pad)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Annotate bars
    for bar, val in zip(bars, vals):
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}", ha="center", va=va, fontsize=8)
    return _save(fig, save_path)


# ─── 5. Distribution shift summary ───────────────────────────────────────────

def plot_ks_per_feature(ks_dict: dict,
                         method: str,
                         dataset_name: str,
                         save_path: str) -> str:
    """Bar chart of KS statistic per feature."""
    feats = [k for k in ks_dict if k.startswith("ks_")]
    if not feats:
        return None
    labels = [f.replace("ks_", "") for f in feats]
    vals   = [ks_dict[f] for f in feats]

    fig, ax = plt.subplots(figsize=(max(7, len(feats) * 0.6), 4))
    colors  = ["#D32F2F" if v > 0.2 else "#1565C0" for v in vals]
    ax.bar(range(len(feats)), vals, color=colors,
           edgecolor="white", linewidth=0.4)
    ax.axhline(0.2, color="#FF6F00", linewidth=1.2,
               linestyle="--", label="KS = 0.2 (caution threshold)")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("KS Statistic")
    ax.set_title(f"Feature-wise KS (Original vs Augmented) — {dataset_name} — {method}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    return _save(fig, save_path)


# ─── 6. Multi-method SHAP stability heatmap ───────────────────────────────────

def plot_shap_stability_heatmap(stability_records: list,
                                 dataset_name: str,
                                 model_name: str,
                                 save_path: str) -> str:
    """
    Heatmap: rows = augmentation methods, columns = stability metrics.
    Directly addresses RQ3 and RQ4.
    """
    if not stability_records:
        return None

    df = pd.DataFrame(stability_records)
    df = df[df.columns.intersection(
        ["method", "spearman_corr", "mean_abs_diff", "top5_overlap"]
    )].set_index("method")

    fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.5)))
    sns.heatmap(df.astype(float), ax=ax, cmap="RdYlGn",
                vmin=0, vmax=1, annot=True, fmt=".3f",
                annot_kws={"size": 9}, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title(f"SHAP Stability Summary — {dataset_name} — {model_name.upper()}")
    ax.set_xticklabels(
        ["Spearman ρ", "Mean |ΔSHAP|", "Top-5 Overlap"],
        fontsize=9,
    )
    return _save(fig, save_path)


# ─── 7. Radar chart (aggregate across all methods) ───────────────────────────

def plot_radar(method_scores: dict,
               save_path: str,
               title: str = "Augmentation Method Comparison") -> str:
    """
    Radar / spider chart.
    method_scores : {method: {dim: score}} where dims are performance,
                    fidelity, shap_stability (all normalised to [0,1]).
    """
    dims     = ["Performance", "Fidelity", "SHAP Stability"]
    n_dims   = len(dims)
    methods  = list(method_scores.keys())
    angles   = [n / float(n_dims) * 2 * np.pi for n in range(n_dims)]
    angles  += angles[:1]   # close polygon

    fig, ax = plt.subplots(figsize=(7, 6),
                            subplot_kw={"polar": True})

    for method in methods:
        vals   = [method_scores[method].get(d, 0) for d in dims]
        vals  += vals[:1]
        color  = METHODS_COLORS.get(method, "#888")
        ax.plot(angles, vals, "o-", linewidth=1.5,
                label=METHODS_LABELS.get(method, method), color=color)
        ax.fill(angles, vals, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title(title, pad=15, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(linewidth=0.5, alpha=0.6)

    return _save(fig, save_path)