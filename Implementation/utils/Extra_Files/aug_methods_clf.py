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
            import smogn
            train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
            df_res = smogn.smoter(
                data=train_df, y="target",
                k=k_neighbors, samp_method="balance",
                rel_thres=0.5, random_state=42,
            ).reset_index(drop=True)
            X_aug = df_res.drop(columns=["target"])
            y_aug = df_res["target"]
            n     = len(X_train)
            return (X_aug, y_aug,
                    X_aug.iloc[n:].reset_index(drop=True),
                    y_aug.iloc[n:].reset_index(drop=True))
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

def apply_great(X_train, y_train, n_generate,
                epochs=50, task="classification"):
    try:
        from be_great import GReaT
        train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
        model = GReaT(llm="gpt2", batch_size=32, epochs=epochs,
                      save_steps=10_000, logging_steps=500)
        model.fit(train_df)
        # max_length must be <= 1024 (GPT-2 hard architectural limit).
        # 800 leaves room for the conditioning prompt within the 1024-token window.
        # guided_sampling=True uses beam search — more reliable, slower.
        try:
            synth_df = model.sample(
                n_samples=n_generate,
                k=100,
                max_length=800,
            )
        except Exception:
            # Fallback: guided sampling — more reliable but slower
            print("  [GReaT] Standard sampling failed, trying guided_sampling...")
            synth_df = model.sample(
                n_samples=n_generate,
                k=100,
                max_length=800,
                guided_sampling=True,
            )
        # Align columns
        for col in train_df.columns:
            if col not in synth_df.columns:
                synth_df[col] = np.nan
        X_syn = synth_df[X_train.columns].reset_index(drop=True)
        y_syn = synth_df["target"].reset_index(drop=True)
        if task == "classification":
            y_syn = y_syn.round().astype(int)
        else:
            y_syn = y_syn.astype(float)
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


def apply_dice(X_train, y_train, cat_cols, cont_cols,
               n_cfs=DICE_NUM_CFS, task="classification",
               desired_class="opposite", n_query=100):
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
        else:
            pool = X_train
        pool = pool.sample(
            n=min(n_query, len(pool)), random_state=42
        ).reset_index(drop=True)

        delta = _compute_dice_delta(y_train) if task == "regression" else None
        all_X, all_y = [], []
        # Suppress tqdm/DiCE progress bars written to stderr
        import tqdm as _tqdm_mod, os as _os
        _old_stderr = None
        try:
            import sys as _sys
            _old_stderr = _sys.stderr
            _sys.stderr = open(_os.devnull, "w")
        except Exception:
            pass

        for i in range(len(pool)):
            row = pool.iloc[[i]]
            try:
                import io, contextlib
                _sink = contextlib.redirect_stdout(io.StringIO())
                if task == "classification":
                    with _sink:
                        res = exp.generate_counterfactuals(
                            row, total_CFs=n_cfs, desired_class=desired_class,
                            proximity_weight=DICE_PROXIMITY_W,
                            diversity_weight=DICE_DIVERSITY_W,
                        )
                else:
                    y_i = float(y_train.iloc[i])
                    with _sink:
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
            except Exception:
                continue

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

    if method == "smote_nc":
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
        else:
            return apply_dice(X_train, y_train, cat_cols, cont_cols, task=task)

    else:
        raise ValueError(f"Unknown augmentation method: '{method}'")