"""
aug_methods.py
==============
All 6 augmentation methods + single dispatcher function.
Every method receives TRAINING DATA ONLY and returns:
    (X_aug, y_aug, X_synthetic_only, y_synthetic_only)
Returns (None, None, None, None) on failure — caller checks and skips.

The TEST SET is NEVER passed into any function here.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings

# Build tag — verify the kernel loaded THIS file (not a cached old module).
#   >>> import utils.aug_methods as A; print(A.AUG_BUILD)
AUG_BUILD = "rowhash-antijoin+dice-cfg-2026-06-10"
warnings.filterwarnings("ignore")

# Ensure Implementation/ is on sys.path
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils.config import (
    CLF_AUG_STRATEGY, LARGE_SUBSAMPLE_N,
    DICE_NUM_CFS, DICE_PROXIMITY_W, DICE_DIVERSITY_W,
    DICE_REG_DELTA_MIN, DICE_REG_DELTA_COEF,
)


# ─── HELPER ───────────────────────────────────────────────────────────────────

def _combine(X_orig, y_orig, X_syn, y_syn):
    """Concatenate original + synthetic. Returns combined + synthetic-only."""
    X_syn = pd.DataFrame(X_syn, columns=X_orig.columns)
    y_syn = pd.Series(y_syn, name=y_orig.name)
    X_aug = pd.concat([X_orig, X_syn], ignore_index=True)
    y_aug = pd.concat([y_orig, y_syn], ignore_index=True)
    assert len(X_aug) == len(X_orig) + len(X_syn), "Row count mismatch"
    assert X_aug.isna().sum().sum() == 0, "NaN in augmented set"
    return X_aug, y_aug, X_syn, y_syn


# ─── SMOGN / NumPy 2.x COMPATIBILITY SHIM ─────────────────────────────────────
# The smogn package (last released 2020; the maintained successor
# ImbalancedLearningRegression shares the same legacy code) breaks on NumPy
# >= 1.22 / 2.x in two places:
#   1. box_plot_stats.py calls np.quantile(..., interpolation="midpoint");
#      NumPy renamed `interpolation` -> `method` (removed in 2.0).
#   2. over_sampling*.py does int(np.random.choice(..., size=1)); NumPy 2.0
#      forbids converting a 1-element array to a Python scalar.
# Rather than edit the installed package on the server, we translate both for
# the duration of the smoter() call. The shim is a no-op on older NumPy, so it
# is safe across versions.

import contextlib as _contextlib

def _smogn_safe_int(x):
    if hasattr(x, "size") and getattr(x, "size", None) == 1:
        return int(np.asarray(x).reshape(-1)[0])
    return int(x)


def _smogn_safe_float(x):
    if hasattr(x, "size") and getattr(x, "size", None) == 1:
        return float(np.asarray(x).reshape(-1)[0])
    return float(x)


@_contextlib.contextmanager
def _smogn_numpy_compat(drop_nan_synth: bool = False):
    """
    Make the (2020-era) smogn package run on NumPy >= 1.22 / 2.x without editing
    the installed package. Fixes, scoped to the smoter() call:

      1. box_plot_stats.py calls np.quantile/np.percentile(interpolation="..."),
         a keyword NumPy renamed to `method` (removed in 2.0). We translate it.

      2. over_sampling.py does int(np.random.choice(..., size=1)) and
         float(np.random.normal(..., size=1)); NumPy 2.0 forbids int()/float()
         on a 1-element array. We inject array-safe int/float into smogn's
         module namespace (module globals shadow builtins), which fixes exactly
         those conversions. Crucially we do NOT touch np.random itself, so
         smogn's `r_index = np.random.choice(..., size=n_synth)` stays an
         iterable array even when n_synth == 1.

      3. (drop_nan_synth=True) SMOGN raises "synthetic data contains missing
         values" and discards the WHOLE augmentation if any single synthetic
         row has a NaN (a numerical edge case in its per-bump interpolation).
         When enabled we swap in a source-patched over_sampling that drops only
         the offending rows and keeps the rest — used as a fallback so one bad
         row never wipes out an otherwise valid SMOGN result.
    """
    import smogn.over_sampling as _os_mod
    import smogn.box_plot_stats as _bps_mod
    import importlib
    _smoter_mod = importlib.import_module("smogn.smoter")

    _oq, _op = np.quantile, np.percentile

    def _kw_wrap(fn):
        def inner(*a, **kw):
            if "interpolation" in kw:
                kw["method"] = kw.pop("interpolation")
            return fn(*a, **kw)
        return inner

    np.quantile, np.percentile = _kw_wrap(_oq), _kw_wrap(_op)
    _patched = [_os_mod, _bps_mod]
    _saved = [(m, m.__dict__.get("int", None), m.__dict__.get("float", None))
              for m in _patched]
    for m in _patched:
        m.int = _smogn_safe_int
        m.float = _smogn_safe_float

    _orig_over_sampling = None
    if drop_nan_synth:
        patched = _build_nan_dropping_over_sampling()
        if patched is not None:
            _orig_over_sampling = _smoter_mod.over_sampling
            _smoter_mod.over_sampling = patched
    try:
        yield
    finally:
        np.quantile, np.percentile = _oq, _op
        for m, old_int, old_float in _saved:
            if old_int is None:
                m.__dict__.pop("int", None)
            else:
                m.int = old_int
            if old_float is None:
                m.__dict__.pop("float", None)
            else:
                m.float = old_float
        if _orig_over_sampling is not None:
            _smoter_mod.over_sampling = _orig_over_sampling


def _build_nan_dropping_over_sampling():
    """
    Return a copy of smogn.over_sampling.over_sampling whose NaN 'quality check'
    drops the offending synthetic rows instead of raising ValueError. Built by
    source-patching the installed function (version-robust). Returns None if the
    expected line isn't found, in which case the caller falls back gracefully.
    """
    import smogn.over_sampling as _osm
    import inspect, textwrap
    try:
        src = textwrap.dedent(inspect.getsource(_osm.over_sampling))
    except Exception:
        return None
    needle = ('raise ValueError("oops! synthetic data contains missing '
              'values")')
    if needle not in src:
        return None
    # Drop NaN synthetic rows (keep the clean ones) and continue. dropna() here
    # is before label-decoding / constant-feature reintroduction, so the rest of
    # over_sampling proceeds normally on the surviving rows.
    src = src.replace(
        needle,
        "data_new = data_new.dropna().reset_index(drop=True)")
    ns = dict(_osm.__dict__)          # mirror the module globals (np, pd, tqdm…)
    ns["int"] = _smogn_safe_int
    ns["float"] = _smogn_safe_float
    try:
        exec(src, ns)
        return ns["over_sampling"]
    except Exception:
        return None


def _smogn_manual_relevance(y_values, rare_mask=None):
    """
    Build SMOGN manual-relevance control points [[x, phi, slope], ...] so SMOGN
    oversamples the rare target region and never hits its automatic-relevance
    degeneracy ('all points are 0/1'), which fails on near-discrete targets
    (e.g. SWD) and some continuous ones.

    Relevance is 1 at the rare-region boundary/extreme(s) and 0 at the median.
    Boundaries come from the evaluation rare-mask when it is non-empty (so
    SMOGN's region matches what we score on); otherwise they fall back to the
    10th/90th-percentile tails so a valid spec still exists. Returns None only
    when the target is constant.
    """
    y = np.asarray(y_values, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return None
    med = float(np.median(y))

    lo_pt = hi_pt = None
    if rare_mask is not None and np.asarray(rare_mask, dtype=bool).sum() > 0:
        rare = y[np.asarray(rare_mask, dtype=bool)]
        low, high = rare[rare < med], rare[rare > med]
        if low.size:
            lo_pt = float(low.max())
        if high.size:
            hi_pt = float(high.min())
    if lo_pt is None and hi_pt is None:          # empty/median-only mask
        lo_pt, hi_pt = (float(np.percentile(y, 10)),
                        float(np.percentile(y, 90)))

    def _assemble(lo, hi):
        pts = []
        if lo is not None and lo < med:
            pts.append([lo, 1.0, 0.0])
        pts.append([med, 0.0, 0.0])
        if hi is not None and hi > med:
            pts.append([hi, 1.0, 0.0])
        xs = [p[0] for p in pts]
        if len(pts) >= 2 and all(b > a for a, b in zip(xs, xs[1:])):
            return pts
        return None

    pts = _assemble(lo_pt, hi_pt)
    if pts is None:                              # last resort: data extremes
        pts = _assemble(float(y.min()), float(y.max()))
    return pts


def _run_smogn(train_df, k_neighbors, rel_ctrl_pts=None):
    """
    Run smogn.smoter robustly.

    If rel_ctrl_pts is given, SMOGN uses MANUAL relevance (our rare-region
    control points) instead of its automatic boxplot relevance. This both keeps
    SMOGN's oversampled region consistent with the evaluation rare region and
    avoids the 'all points are 0/1' degeneracy that fails on near-discrete or
    awkward target distributions.

    Attempt 1 is the standard call. If SMOGN raises "synthetic data contains
    missing values" (a numerical edge case in its per-bump interpolation that
    discards the entire result), attempt 2 retries with a source-patched
    over_sampling that drops only the NaN synthetic rows and keeps the rest.

    Returns (df_res, base_df) — base_df is the frame passed to SMOGN, needed so
    the caller's anti-join can separate synthetic rows. (None, None) on failure.
    """
    import smogn
    base_df = train_df.copy()
    rel_kwargs = ({"rel_method": "manual", "rel_ctrl_pts_rg": rel_ctrl_pts}
                  if rel_ctrl_pts is not None else {"rel_method": "auto"})

    def _call():
        return smogn.smoter(
            data=base_df, y="target", k=k_neighbors,
            samp_method="balance", rel_thres=0.5, **rel_kwargs,
        ).reset_index(drop=True)

    # Attempt 1: standard SMOGN (exact for the seeds that work).
    np.random.seed(42)
    try:
        with _smogn_numpy_compat():
            return _call(), base_df
    except ValueError as e:
        if "missing values" not in str(e):
            raise

    # Attempt 2: NaN-tolerant SMOGN — drop only the offending synthetic rows.
    np.random.seed(42)
    try:
        with _smogn_numpy_compat(drop_nan_synth=True):
            df_res = _call()
        print("  [SMOGN] Recovered by dropping NaN synthetic rows "
              "(SMOGN's missing-value edge case)")
        return df_res, base_df
    except Exception as e:
        print(f"  [SMOGN] NaN-tolerant retry also failed: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# A. SMOTE-NC  /  SMOGN
# ═══════════════════════════════════════════════════════════════════════════════

def apply_smote_nc(X_train, y_train, cat_cols, k_neighbors=5,
                   task="classification"):
    # SMOTE-NC needs at least one continuous feature to interpolate on.
    # All-categorical datasets (e.g. car: 6 cat, 0 cont) cannot use it.
    all_cols = list(X_train.columns) if hasattr(X_train, "columns") else []
    n_cat    = len([c for c in all_cols if c in cat_cols]) if cat_cols else 0
    if n_cat == len(all_cols) and len(all_cols) > 0:
        print(f"  [SMOTE-NC] Dataset is all-categorical ({n_cat} cols, 0 continuous)."
              " SMOTE-NC requires ≥1 numerical feature — skipping.")
        return None, None, None, None
    if task == "classification":
        try:
            from imblearn.over_sampling import SMOTENC, SMOTE
            X_np = X_train.values
            y_np = y_train.values
            if cat_cols:
                cat_idx = [X_train.columns.get_loc(c) for c in cat_cols
                           if c in X_train.columns]
                sm = SMOTENC(categorical_features=cat_idx,
                             sampling_strategy=CLF_AUG_STRATEGY,
                             k_neighbors=k_neighbors, random_state=42)
            else:
                sm = SMOTE(sampling_strategy=CLF_AUG_STRATEGY,
                           k_neighbors=k_neighbors, random_state=42)
            X_res, y_res = sm.fit_resample(X_np, y_np)
            n = len(X_train)
            X_aug = pd.DataFrame(X_res,   columns=X_train.columns)
            y_aug = pd.Series(y_res,      name=y_train.name)
            X_syn = pd.DataFrame(X_res[n:], columns=X_train.columns)
            y_syn = pd.Series(y_res[n:],  name=y_train.name)
            return X_aug, y_aug, X_syn, y_syn
        except Exception as e:
            print(f"  [SMOTE-NC] Error: {e}")
            return None, None, None, None
    else:
        try:
            train_df = pd.concat([X_train, y_train.rename("target")], axis=1
                                 ).reset_index(drop=True)
            # Drive SMOGN with MANUAL relevance built from the same rare-region
            # definition used for evaluation. This (a) makes SMOGN oversample
            # exactly the rare target region we score on, and (b) avoids SMOGN's
            # automatic-relevance degeneracy ("all points are 0/1") that fails
            # outright on near-discrete targets (e.g. SWD) and some continuous
            # ones. Falls back to auto relevance if a valid spec can't be built.
            try:
                from utils.evaluation import compute_rarity_mask
                rare_mask, _ = compute_rarity_mask(y_train, y_train)
                rel_ctrl_pts = _smogn_manual_relevance(y_train, rare_mask)
            except Exception:
                rel_ctrl_pts = None

            df_res, base_df = _run_smogn(train_df, k_neighbors, rel_ctrl_pts)
            if df_res is None:
                print("  [SMOGN] Failed after retries — skipping")
                return None, None, None, None

            # SMOGN returns a *resampled* set (some originals under-sampled away
            # plus new synthetic rare-region rows), NOT original-then-synthetic.
            # Identify the synthetic rows (those not present in the data SMOGN
            # was given) and append them to the full original training set via
            # _combine, matching the generative methods.
            #
            # We hash each row instead of using a merge/anti-join. A merge
            # explodes on datasets with many duplicate rows — e.g. SWD has ~83%
            # duplicate integer rows, so each original row in df_res matches many
            # identical base rows, the join balloons (1500+ rows vs ~430), and
            # the boolean mask comes out the wrong length ("Item wrong length N
            # instead of M"). Row-hashing yields a mask of exactly len(df_res)
            # every time, regardless of duplicates. All columns are numeric here
            # (features are encoded; target is numeric), so we round to absorb
            # float noise and compare tuples.
            cols       = list(base_df.columns)
            base_keys  = set(map(tuple, np.round(
                base_df[cols].to_numpy(dtype=float), 8)))
            res_arr    = np.round(df_res[cols].to_numpy(dtype=float), 8)
            syn_mask   = np.fromiter(
                (tuple(r) not in base_keys for r in res_arr),
                dtype=bool, count=len(res_arr))
            synth      = df_res[syn_mask].reset_index(drop=True)
            if len(synth) == 0:
                print("  [SMOGN] No synthetic rows generated — skipping")
                return None, None, None, None
            return _combine(X_train, y_train,
                            synth.drop(columns=["target"]), synth["target"])
        except Exception as e:
            print(f"  [SMOGN] Error: {e}")
            return None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# B. CTGAN
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ctgan(X_train, y_train, metadata, n_generate,
                epochs=300, task="classification"):
    try:
        from sdv.single_table import CTGANSynthesizer
        train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
        m = CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
        m.fit(train_df)
        synth = m.sample(num_rows=n_generate)
        return _combine(X_train, y_train,
                        synth.drop(columns=["target"]), synth["target"])
    except Exception as e:
        print(f"  [CTGAN] Error: {e}")
        return None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# C. TVAE
# ═══════════════════════════════════════════════════════════════════════════════

def apply_tvae(X_train, y_train, metadata, n_generate,
               epochs=300, task="classification"):
    try:
        from sdv.single_table import TVAESynthesizer
        train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
        m = TVAESynthesizer(metadata, epochs=epochs)
        m.fit(train_df)
        synth = m.sample(num_rows=n_generate)
        return _combine(X_train, y_train,
                        synth.drop(columns=["target"]), synth["target"])
    except Exception as e:
        print(f"  [TVAE] Error: {e}")
        return None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# D. TabDDPM  (ForestDiffusion backend — pip installable)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_tabddpm(X_train, y_train, n_generate, cat_cols,
                  diffusion_steps=1000, task="classification"):
    # ── Step 1: import — 4-tier fallback (package → lowercase → local file → local folder)
    import importlib.util, pathlib, sys as _sys

    ForestDiffusionModel = None
    _base = pathlib.Path(__file__).parent.parent  # Implementation/

    # Tier 1: standard pip package (capital letters)
    if ForestDiffusionModel is None:
        try:
            from ForestDiffusion import ForestDiffusionModel
        except (ModuleNotFoundError, ImportError):
            pass

    # Tier 2: lowercase variant (some platforms)
    if ForestDiffusionModel is None:
        try:
            from forestdiffusion import ForestDiffusionModel
        except (ModuleNotFoundError, ImportError):
            pass

    # Tier 3: local ForestDiffusion.py file in Implementation/
    if ForestDiffusionModel is None:
        _f = _base / "ForestDiffusion.py"
        if _f.exists():
            _spec = importlib.util.spec_from_file_location("ForestDiffusion", _f)
            _mod  = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            ForestDiffusionModel = _mod.ForestDiffusionModel
            print("  [TabDDPM] Using local ForestDiffusion.py")

    # Tier 4: cloned folder Implementation/ForestDiffusion/ or Implementation/fd_temp/ForestDiffusion/
    if ForestDiffusionModel is None:
        for candidate in [_base / "ForestDiffusion",
                          _base / "fd_temp" / "ForestDiffusion"]:
            if candidate.is_dir():
                _sys.path.insert(0, str(candidate.parent))
                try:
                    from ForestDiffusion import ForestDiffusionModel
                    print(f"  [TabDDPM] Using cloned folder: {candidate}")
                    break
                except (ModuleNotFoundError, ImportError):
                    _sys.path.pop(0)

    if ForestDiffusionModel is None:
        print("  [TabDDPM] ForestDiffusion not found. Steps to fix:")
        print("  [TabDDPM]   git clone https://github.com/SamsungSAILMontreal/ForestDiffusion.git fd_temp")
        print("  [TabDDPM]   Then copy fd_temp/ForestDiffusion/ into Implementation/")
        return None, None, None, None

    # ── Step 2: run the model — separated so import errors don't hide runtime errors
    try:
        data = np.hstack([X_train.values.astype(float),
                          y_train.values.reshape(-1, 1)])
        n_t = max(1, min(50, diffusion_steps // 100))
        model = ForestDiffusionModel(
            data,
            n_t=n_t,
            duplicate_K=100,
            seed=42,
        )
        synth = model.generate(batch_size=n_generate)
        X_syn = pd.DataFrame(synth[:, :-1], columns=X_train.columns)
        y_syn = pd.Series(synth[:, -1], name=y_train.name)
        if task == "classification":
            y_syn = y_syn.round().astype(int)
        print(f"  [TabDDPM] ForestDiffusion OK — generated {len(X_syn)} rows")
        return _combine(X_train, y_train, X_syn, y_syn)
    except Exception as e:
        print(f"  [TabDDPM] Runtime error ({type(e).__name__}): {e}")
        return None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# E. GReaT
# ═══════════════════════════════════════════════════════════════════════════════

def _great_clean_regression(synth_df, X_train, y_train):
    """
    Coerce a GReaT sample batch to numeric and drop implausible rows for
    regression. GReaT (a GPT-2 text model) emits the continuous target as text
    and occasionally produces non-numeric tokens or grossly out-of-range values
    (e.g. concatenated digits) that wreck a regressor. We drop rows whose target
    or any feature is non-numeric or outside a generous 50%-margin of the
    observed training range.

    Returns (X_kept, y_kept, n_nonnumeric, n_outrange).
    """
    Xs = synth_df.reindex(columns=list(X_train.columns)).reset_index(drop=True)
    ys = pd.to_numeric(synth_df["target"], errors="coerce").reset_index(drop=True) \
        if "target" in synth_df.columns else pd.Series([np.nan] * len(synth_df))
    for c in Xs.columns:
        Xs[c] = pd.to_numeric(Xs[c], errors="coerce")
    valid = ys.notna() & Xs.notna().all(axis=1)
    n_bad = int((~valid).sum())
    Xs, ys = Xs[valid].reset_index(drop=True), ys[valid].reset_index(drop=True).astype(float)
    if len(Xs) == 0:
        return Xs, ys, n_bad, 0

    def _col_in_range(col_syn, col_ref):
        lo, hi = float(col_ref.min()), float(col_ref.max())
        rng = (hi - lo) if hi > lo else (abs(hi) or 1.0)
        return (col_syn >= lo - 0.5 * rng) & (col_syn <= hi + 0.5 * rng)

    keep = _col_in_range(ys, y_train)
    for c in Xs.columns:
        keep &= _col_in_range(Xs[c], X_train[c])
    n_out = int((~keep).sum())
    return (Xs[keep].reset_index(drop=True),
            ys[keep].reset_index(drop=True), n_bad, n_out)


def apply_great(X_train, y_train, n_generate,
                epochs=50, task="classification"):
    try:
        from be_great import GReaT
        train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
        model = GReaT(llm="gpt2", batch_size=32, epochs=epochs,
                      save_steps=10_000, logging_steps=500)
        model.fit(train_df)

        # max_length must be <= 1024 (GPT-2 hard architectural limit). 800
        # leaves room for the conditioning prompt; guided_sampling uses beam
        # search (more reliable, slower) as a fallback.
        def _sample(n):
            try:
                return model.sample(n_samples=n, k=100, max_length=800)
            except Exception:
                print("  [GReaT] Standard sampling failed, trying guided_sampling...")
                return model.sample(n_samples=n, k=100, max_length=800,
                                    guided_sampling=True)

        if task == "classification":
            synth_df = _sample(n_generate)
            for col in train_df.columns:
                if col not in synth_df.columns:
                    synth_df[col] = np.nan
            X_syn = synth_df[X_train.columns].reset_index(drop=True)
            y_syn = synth_df["target"].reset_index(drop=True).round().astype(int)
            return _combine(X_train, y_train, X_syn, y_syn)

        # ── Regression: adaptive over-generation ──────────────────────────────
        # GReaT's plausible-row keep-rate is high for small-range targets
        # (no2, SWD) but low for high-range / multi-scale ones (machine_cpu,
        # bodyfat), where most generations are out-of-range. We sample, clean,
        # and top up only while we're short of n_generate — so good datasets
        # finish in one batch and hard ones still reach a comparable row count
        # instead of 1-3 rows. Capped attempts keep GReaT's runtime bounded.
        GREAT_MAX_ATTEMPTS = 4
        kept_X, kept_y, total, n_bad_all, n_out_all = [], [], 0, 0, 0
        for attempt in range(GREAT_MAX_ATTEMPTS):
            need = n_generate - total
            if need <= 0:
                break
            req = n_generate if attempt == 0 else max(need, need * 3)
            synth_df = _sample(req)
            Xk, yk, n_bad, n_out = _great_clean_regression(
                synth_df, X_train, y_train)
            n_bad_all += n_bad
            n_out_all += n_out
            if len(Xk):
                kept_X.append(Xk)
                kept_y.append(yk)
                total += len(Xk)
            elif attempt >= 1:
                break   # GReaT can't produce in-range rows for this dataset

        if n_bad_all:
            print(f"  [GReaT] Dropped {n_bad_all} non-numeric synthetic rows "
                  f"(regression)")
        if n_out_all:
            print(f"  [GReaT] Dropped {n_out_all} out-of-range synthetic rows "
                  f"(regression)")
        if total == 0:
            print("  [GReaT] No in-range rows after filtering — skipping")
            return None, None, None, None

        X_syn = pd.concat(kept_X, ignore_index=True).iloc[:n_generate]
        y_syn = pd.concat(kept_y, ignore_index=True).iloc[:n_generate]
        return _combine(X_train, y_train, X_syn, y_syn)
    except ImportError:
        print("  [GReaT] be-great not installed: pip install be-great")
        return None, None, None, None
    except Exception as e:
        print(f"  [GReaT] Error: {e}")
        return None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# F. DiCE  — counterfactual augmentation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_dice_delta(y_train):
    """Professor point 7: delta = max(2, round(std(y_train) * 0.10, 2))"""
    return max(DICE_REG_DELTA_MIN,
               round(float(y_train.std()) * DICE_REG_DELTA_COEF, 2))


class _DiceQueryTimeout(BaseException):
    """Raised to abort a single DiCE counterfactual query that runs too long.
    Subclasses BaseException so DiCE's internal `except Exception` can't swallow
    it."""
    pass


@_contextlib.contextmanager
def _query_time_limit(seconds):
    """
    Hard wall-clock limit for a single block, via SIGALRM. DiCE's genetic search
    can spend many minutes on one query; the per-method budget only checks
    BETWEEN queries, so it cannot interrupt a single long one. This does.

    SIGALRM only works in the main thread; off-thread this is a no-op (the
    per-method budget still applies). The alarm fires between Python bytecodes,
    so it interrupts DiCE's prediction loop promptly even though parts run in C.
    """
    import signal, threading
    if seconds is None or seconds <= 0 or \
       threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):
        raise _DiceQueryTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def apply_dice(X_train, y_train, cat_cols, cont_cols,
               n_cfs=DICE_NUM_CFS, task="classification",
               desired_class="opposite", n_query=100,
               dice_max_seconds=600, dice_query_timeout=60):
    try:
        import dice_ml
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        train_df = pd.concat(
            [X_train, y_train.rename("target")], axis=1
        ).reset_index(drop=True)

        # Base model for CF generation only (NOT the evaluation model)
        if task == "classification":
            from sklearn.linear_model import LogisticRegression
            base = Pipeline([
                ("sc",  StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ])
            base.fit(X_train, y_train)
            model_type = "classifier"
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            base = GradientBoostingRegressor(n_estimators=100, random_state=42)
            base.fit(X_train, y_train)
            model_type = "regressor"

        d = dice_ml.Data(
            dataframe=train_df,
            continuous_features=cont_cols if cont_cols else [],
            outcome_name="target",
        )
        m = dice_ml.Model(model=base, backend="sklearn", model_type=model_type)
        # Use "genetic" — more reliable than "random" for sklearn models.
        # "gradient" only works with PyTorch/TF (not sklearn).
        dice_method = "genetic"
        exp = dice_ml.Dice(d, m, method=dice_method)

        # Query pool
        if task == "classification":
            minority_cls = y_train.value_counts().idxmin()
            pool = X_train[y_train == minority_cls]
            pool = pool.sample(
                n=min(n_query, len(pool)), random_state=42
            ).reset_index(drop=True)
            pool_y = None
        else:
            # Regression: keep each query row aligned with ITS OWN target so the
            # desired_range is centred on the row's value. (Previously the loop
            # used y_train.iloc[i] after the pool had been shuffled, so the range
            # was centred on an unrelated target — this made the genetic search
            # thrash, causing the multi-hour runtimes and wild variance.)
            idx  = X_train.sample(
                n=min(n_query, len(X_train)), random_state=42).index
            pool   = X_train.loc[idx].reset_index(drop=True)
            pool_y = y_train.loc[idx].reset_index(drop=True)

        delta = _compute_dice_delta(y_train) if task == "regression" else None
        all_X, all_y = [], []
        import time as _time
        _dice_t0 = _time.time()
        # Suppress tqdm/DiCE progress bars written to stderr
        import tqdm as _tqdm_mod, os as _os
        _old_stderr = None
        try:
            import sys as _sys
            _old_stderr = _sys.stderr
            _sys.stderr = open(_os.devnull, "w")
        except Exception:
            pass

        _n_timeout = 0
        for i in range(len(pool)):
            # Per-method wall-clock budget: stop generating once exceeded and
            # use whatever counterfactuals have accumulated. Protects the session
            # time budget from pathological genetic-search convergence.
            if _time.time() - _dice_t0 > dice_max_seconds:
                print(f"  [DiCE] Time budget ({dice_max_seconds}s) reached at "
                      f"query {i}/{len(pool)} — using CFs generated so far")
                break
            row = pool.iloc[[i]]
            try:
                import io, contextlib
                _sink = contextlib.redirect_stdout(io.StringIO())
                # Hard per-query cap: a single genetic search cannot exceed
                # dice_query_timeout seconds (the per-method budget can't
                # interrupt one long query; this can).
                if task == "classification":
                    with _query_time_limit(dice_query_timeout), _sink:
                        res = exp.generate_counterfactuals(
                            row, total_CFs=n_cfs, desired_class=desired_class,
                            proximity_weight=DICE_PROXIMITY_W,
                            diversity_weight=DICE_DIVERSITY_W,
                        )
                else:
                    y_i = float(pool_y.iloc[i])
                    with _query_time_limit(dice_query_timeout), _sink:
                        res = exp.generate_counterfactuals(
                            row, total_CFs=n_cfs,
                            desired_range=[y_i - delta, y_i + delta],
                            proximity_weight=DICE_PROXIMITY_W,
                            diversity_weight=DICE_DIVERSITY_W,
                        )
                for ex in res.cf_examples_list:
                    cf = ex.final_cfs_df
                    if cf is not None and len(cf):
                        all_X.append(cf.drop(columns=["target"]))
                        all_y.extend(cf["target"].tolist())
            except _DiceQueryTimeout:
                _n_timeout += 1
                continue
            except Exception:
                continue
        if _n_timeout:
            print(f"  [DiCE] {_n_timeout} query(ies) hit the "
                  f"{dice_query_timeout}s per-query cap and were skipped")

        # Restore stderr after DiCE generation
        try:
            if _old_stderr is not None:
                _sys.stderr.close()
                _sys.stderr = _old_stderr
        except Exception:
            pass

        if not all_X:
            print("  [DiCE] No counterfactuals generated — skipping")
            return None, None, None, None

        X_syn = pd.concat(all_X, ignore_index=True)[X_train.columns]
        y_syn = pd.Series(all_y, name=y_train.name)
        if task == "classification":
            y_syn = y_syn.round().astype(int)
        print(f"  [DiCE] Generated {len(X_syn)} CFs  "
              f"(delta={delta if delta else 'N/A'}, method={dice_method})")
        return _combine(X_train, y_train, X_syn, y_syn)

    except ImportError:
        print("  [DiCE] dice-ml not installed: pip install dice-ml")
        return None, None, None, None
    except Exception as e:
        print(f"  [DiCE] Error: {e}")
        return None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCHER  — single entry point used by all 6 codebases
# ═══════════════════════════════════════════════════════════════════════════════

def run_augmentation(method, X_train, y_train,
                     cat_cols, cont_cols,
                     metadata,
                     n_generate,
                     cfg,
                     task="classification"):
    """
    Dispatch to the correct augmentation method.
    Returns (X_aug, y_aug, X_syn, y_syn) or (None,None,None,None).
    """
    size   = cfg["size"]
    epochs = cfg["gan_epochs"]
    steps  = cfg["diffusion_steps"]
    k      = cfg["smote_k"]

    print(f"  Running: {method.upper()}")

    if method == "smote_nc" or method == "smogn":
        # "smote_nc" (classification) and "smogn" (regression) both route here;
        # apply_smote_nc branches on task. The separate key lets the regression
        # pipeline label the method SMOGN everywhere (folders, logs, plots).
        return apply_smote_nc(X_train, y_train, cat_cols,
                              k_neighbors=k, task=task)

    elif method == "ctgan":
        return apply_ctgan(X_train, y_train, metadata,
                           n_generate, epochs, task)

    elif method == "tvae":
        return apply_tvae(X_train, y_train, metadata,
                          n_generate, epochs, task)

    elif method == "tabddpm":
        return apply_tabddpm(X_train, y_train, n_generate,
                             cat_cols, steps, task)

    elif method == "great":
        great_epochs = cfg["great_epochs"]
        # Subsample large datasets before fine-tuning
        if size == "large" and len(X_train) > LARGE_SUBSAMPLE_N:
            from utils.data_utils import stratified_subsample
            X_tr, y_tr = stratified_subsample(
                X_train, y_train, LARGE_SUBSAMPLE_N, task, seed=42
            )
            print(f"  [GReaT] Subsampled training set to {len(X_tr)} rows")
        else:
            X_tr, y_tr = X_train, y_train
        return apply_great(X_tr, y_tr, n_generate, great_epochs, task)

    elif method == "dice":
        # For multiclass (>2 classes): target the majority class from the minority class pool.
        # "opposite" only works for binary classification.
        if task == "classification" and y_train.nunique() > 2:
            majority_cls = int(y_train.value_counts().idxmax())
            return apply_dice(X_train, y_train, cat_cols, cont_cols,
                              task=task, desired_class=majority_cls)
        elif task == "regression":
            # Scale the query-pool size so DiCE generates roughly n_generate
            # counterfactuals (n_query * n_cfs ~= n_generate), matching the
            # augmentation budget of the generative methods instead of the old
            # fixed 100 queries x 3 CFs = 300. With far fewer genetic searches
            # this also turns multi-hour runtimes into minutes.
            n_q = max(8, -(-n_generate // DICE_NUM_CFS))   # ceil division
            # Per-cell wall-clock budget / per-query timeout are overridable
            # via cfg (medium/large pipelines raise the cell budget because
            # n_query grows with the data); defaults reproduce the small-phase
            # behaviour exactly (600s / 60s).
            return apply_dice(X_train, y_train, cat_cols, cont_cols,
                              task=task, n_query=n_q,
                              dice_max_seconds=cfg.get("dice_max_seconds", 600),
                              dice_query_timeout=cfg.get("dice_query_timeout", 60))
        else:
            return apply_dice(X_train, y_train, cat_cols, cont_cols, task=task)

    else:
        raise ValueError(f"Unknown augmentation method: '{method}'")