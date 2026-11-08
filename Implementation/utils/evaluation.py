"""
evaluation.py
─────────────────────────────────────────────────────────────────────
All metric computation: classification, regression, and distribution.
Every function returns a plain dictionary so results can be
saved directly to CSV without extra conversion.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def clf_metrics(y_true, y_pred, y_prob=None,
                model_name: str = "",
                method: str = "") -> dict:
    """
    Compute all classification metrics.
    y_prob : predicted probabilities (for AUC); pass None if not available.
    Returns a flat dict ready for CSV logging.
    """
    n_classes = len(np.unique(y_true))

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)

    auc = np.nan
    if y_prob is not None:
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob,
                                    multi_class="ovr", average="macro")
        except Exception:
            auc = np.nan

    return {
        "model":     model_name,
        "method":    method,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_macro":  round(f1,   4),
        "mcc":       round(mcc,  4),
        "auc_roc":   round(auc,  4) if not np.isnan(auc) else np.nan,
    }


def clf_report_str(y_true, y_pred) -> str:
    """Full sklearn classification report as a string."""
    return classification_report(y_true, y_pred, zero_division=0)


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def reg_metrics(y_true, y_pred,
                model_name: str = "",
                method: str = "") -> dict:
    """
    Compute MAE, RMSE, R², and MAPE (where applicable).
    Returns a flat dict ready for CSV logging.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # MAPE — avoid division by zero
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    nonzero    = y_true_arr != 0
    mape = np.mean(np.abs((y_true_arr[nonzero] - y_pred_arr[nonzero])
                           / y_true_arr[nonzero])) * 100 if nonzero.any() else np.nan

    return {
        "model":  model_name,
        "method": method,
        "mae":    round(mae,  4),
        "rmse":   round(rmse, 4),
        "r2":     round(r2,   4),
        "mape":   round(mape, 2) if not np.isnan(mape) else np.nan,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def distribution_metrics(X_orig: pd.DataFrame,
                         X_aug:  pd.DataFrame,
                         method: str = "") -> dict:
    """
    Compare the original training set to the augmented training set.

    Returns:
        ks_mean     : mean KS statistic across continuous features
        ks_max      : worst-case KS statistic (most distorted feature)
        ks_per_feat : dict {feature: ks_stat}
        corr_frob   : Frobenius norm of correlation matrix difference
        method      : augmentation method name
    """
    # Only use numeric columns
    cols = X_orig.select_dtypes(include=[np.number]).columns.tolist()

    ks_stats = {}
    for col in cols:
        stat, _ = ks_2samp(X_orig[col].dropna(), X_aug[col].dropna())
        ks_stats[col] = round(stat, 4)

    ks_vals = list(ks_stats.values())

    # Correlation matrix difference (Frobenius norm)
    corr_orig  = X_orig[cols].corr().fillna(0).values
    corr_aug   = X_aug[cols].corr().fillna(0).values
    corr_diff  = corr_orig - corr_aug
    corr_frob  = round(np.linalg.norm(corr_diff, "fro"), 4)

    result = {
        "method":    method,
        "ks_mean":   round(np.mean(ks_vals), 4),
        "ks_max":    round(np.max(ks_vals),  4),
        "corr_frob": corr_frob,
        "n_features_compared": len(cols),
    }
    # Add per-feature KS
    for feat, ks in ks_stats.items():
        result[f"ks_{feat}"] = ks

    return result


def kl_divergence(X_orig: pd.DataFrame,
                  X_aug:  pd.DataFrame,
                  bins: int = 30) -> dict:
    """
    Estimate KL divergence per numeric feature using histograms.
    Returns {feature: kl_div}.
    """
    cols = X_orig.select_dtypes(include=[np.number]).columns.tolist()
    kl_per_feat = {}

    for col in cols:
        orig_vals = X_orig[col].dropna().values
        aug_vals  = X_aug[col].dropna().values

        lo = min(orig_vals.min(), aug_vals.min())
        hi = max(orig_vals.max(), aug_vals.max())
        if lo == hi:
            kl_per_feat[col] = 0.0
            continue

        bin_edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(orig_vals, bins=bin_edges, density=True)
        q, _ = np.histogram(aug_vals,  bins=bin_edges, density=True)

        # Smooth to avoid log(0)
        eps = 1e-10
        p   = p + eps
        q   = q + eps
        p   = p / p.sum()
        q   = q / q.sum()

        kl  = float(np.sum(p * np.log(p / q)))
        kl_per_feat[col] = round(kl, 4)

    return kl_per_feat


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP STABILITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def shap_stability(importance_baseline: pd.Series,
                   importance_augmented: pd.Series,
                   top_k: int = 5,
                   method: str = "") -> dict:
    """
    Compare SHAP feature importance rankings before and after augmentation.

    importance_baseline / importance_augmented :
        pd.Series indexed by feature name, values = mean |SHAP|

    Returns:
        spearman_corr : Spearman rank correlation of feature orderings
        mean_abs_diff : mean absolute difference of normalised SHAP values
        top_k_overlap : fraction of top-k features shared
        method        : augmentation method name
    """
    # Align features
    features = importance_baseline.index.tolist()
    imp_b = importance_baseline[features]
    imp_a = importance_augmented.reindex(features).fillna(0)

    # Spearman rank correlation
    corr, _ = stats.spearmanr(imp_b.values, imp_a.values)

    # Normalise to [0,1] and compute mean absolute difference
    def _norm(s):
        total = s.sum()
        return s / total if total > 0 else s

    norm_b  = _norm(imp_b)
    norm_a  = _norm(imp_a)
    mad     = float(np.mean(np.abs(norm_b.values - norm_a.values)))

    # Top-k overlap
    top_b  = set(imp_b.nlargest(top_k).index)
    top_a  = set(imp_a.nlargest(top_k).index)
    overlap = len(top_b & top_a) / top_k

    return {
        "method":        method,
        "spearman_corr": round(corr,    4),
        "mean_abs_diff": round(mad,     4),
        f"top{top_k}_overlap": round(overlap, 4),
        "top_baseline":   ", ".join(list(top_b)),
        "top_augmented":  ", ".join(list(top_a)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RARE-TARGET-REGION METRICS  (imbalanced regression — Branco et al. 2017)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rarity_mask(y_reference, y_values, k_iqr: float = 1.5,
                        min_frac: float = 0.05):
    """
    Identify 'rare' (extreme) target values for imbalanced regression.

    Rarity is defined on y_reference (the TRAINING target distribution, so no
    information leaks from the test set) and then applied to y_values. This is
    the same notion of rarity that SMOGN's default relevance function targets:
    the extreme high/low tail of the target.

    Primary rule (boxplot extremes, matching the standard phi-relevance):
        rare  if  y < Q1 - k_iqr*IQR   OR   y > Q3 + k_iqr*IQR

    Fallback: if the boxplot rule flags fewer than `min_frac` of the reference
    points (i.e. the target has no natural outliers), the rare region is taken
    as the bottom and top decile of y_reference. This guarantees a non-empty,
    well-defined rare region for every dataset.

    Returns
    -------
    mask : np.ndarray[bool]   aligned with y_values (True = rare)
    info : dict               thresholds + which rule was used (for logging)
    """
    yref = np.asarray(y_reference, dtype=float)
    yval = np.asarray(y_values,    dtype=float)

    q1, q3 = np.percentile(yref, [25, 75])
    iqr    = q3 - q1
    lo_w   = q1 - k_iqr * iqr
    hi_w   = q3 + k_iqr * iqr

    ref_rare_frac = float(np.mean((yref < lo_w) | (yref > hi_w)))

    if ref_rare_frac >= min_frac and iqr > 0:
        rule = "boxplot_iqr"
        lo, hi = lo_w, hi_w
    else:
        # Fallback: bottom + top decile of the reference distribution
        rule   = "decile_tail"
        lo, hi = np.percentile(yref, [10, 90])

    mask = (yval < lo) | (yval > hi)
    info = {
        "rarity_rule":   rule,
        "low_threshold": round(float(lo), 4),
        "high_threshold":round(float(hi), 4),
        "ref_rare_frac": round(ref_rare_frac, 4),
    }
    return mask, info


def reg_metrics_regional(y_true, y_pred, rare_mask,
                         model_name: str = "", method: str = "") -> dict:
    """
    RMSE / MAE / R² reported OVERALL, on the RARE region, and on the NORMAL
    region. Rare-region error is the key quantity in imbalanced regression:
    a method can lower overall RMSE while making the rare extremes worse, so
    the two must be reported side by side.

    R² is only reported for a region when it has >= 5 points and non-zero
    target variance; otherwise it is NaN (R² is undefined / unstable on a
    tiny or constant region). RMSE and MAE are always well defined.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rare_mask = np.asarray(rare_mask, dtype=bool)

    def _block(yt, yp, prefix):
        if len(yt) == 0:
            return {f"{prefix}_rmse": np.nan, f"{prefix}_mae": np.nan,
                    f"{prefix}_r2": np.nan, f"{prefix}_n": 0}
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae  = float(mean_absolute_error(yt, yp))
        if len(yt) >= 5 and np.var(yt) > 1e-12:
            r2 = float(r2_score(yt, yp))
        else:
            r2 = np.nan
        return {f"{prefix}_rmse": round(rmse, 4),
                f"{prefix}_mae":  round(mae, 4),
                f"{prefix}_r2":   round(r2, 4) if not np.isnan(r2) else np.nan,
                f"{prefix}_n":    int(len(yt))}

    out = {"model": model_name, "method": method}
    out.update(_block(y_true,             y_pred,             "overall"))
    out.update(_block(y_true[rare_mask],  y_pred[rare_mask],  "rare"))
    out.update(_block(y_true[~rare_mask], y_pred[~rare_mask], "normal"))
    return out


def reg_residual_report(y_true, y_pred, rare_mask=None) -> str:
    """Plain-text residual summary (regression analogue of clf_report_str)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    res    = y_pred - y_true
    lines  = [
        "Regression residual report",
        "=" * 40,
        f"n_test            : {len(y_true)}",
        f"target mean / std : {y_true.mean():.4f} / {y_true.std():.4f}",
        f"RMSE              : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}",
        f"MAE               : {mean_absolute_error(y_true, y_pred):.4f}",
        f"R2                : {r2_score(y_true, y_pred):.4f}",
        f"residual mean     : {res.mean():.4f}",
        f"residual std      : {res.std():.4f}",
        f"residual min/max  : {res.min():.4f} / {res.max():.4f}",
    ]
    if rare_mask is not None:
        rare_mask = np.asarray(rare_mask, dtype=bool)
        n_rare = int(rare_mask.sum())
        lines.append(f"rare-region n     : {n_rare}")
        if n_rare > 0:
            lines.append(
                f"rare-region RMSE  : "
                f"{np.sqrt(mean_squared_error(y_true[rare_mask], y_pred[rare_mask])):.4f}")
            lines.append(
                f"rare-region MAE   : "
                f"{mean_absolute_error(y_true[rare_mask], y_pred[rare_mask]):.4f}")
    return "\n".join(lines) + "\n"