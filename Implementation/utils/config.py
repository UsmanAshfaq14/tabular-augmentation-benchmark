"""
config.py
=========
Central configuration. Automatically resolves all paths relative to
this file so the code works on any machine without editing paths.

utils/config.py  →  _HERE = utils/  →  BASE_DIR = Implementation/
"""
import os

_HERE    = os.path.dirname(os.path.abspath(__file__))   # .../utils/
BASE_DIR = os.path.dirname(_HERE)                        # .../Implementation/

CLF_DATA_DIR = os.path.join(BASE_DIR, "Classification", "Datasets")
REG_DATA_DIR = os.path.join(BASE_DIR, "Regression",     "Datasets")
OUTPUT_ROOT  = os.path.join(BASE_DIR, "outputs")

# ── Dataset → local .tsv path ─────────────────────────────────────────────────
# Note: connect-4 and heart-c use hyphens in filename (as downloaded from PMLB).
DATASET_PATHS = {
    # Classification Small
    "heart_c":    os.path.join(CLF_DATA_DIR, "Small_Datasets",  "heart-c.tsv"),
    "saheart":    os.path.join(CLF_DATA_DIR, "Small_Datasets",  "saheart.tsv"),
    "australian": os.path.join(CLF_DATA_DIR, "Small_Datasets",  "australian.tsv"),
    "german":     os.path.join(CLF_DATA_DIR, "Small_Datasets",  "german.tsv"),
    # Classification Medium
    "flare":      os.path.join(CLF_DATA_DIR, "Medium_Datasets", "flare.tsv"),
    "cmc":        os.path.join(CLF_DATA_DIR, "Medium_Datasets", "cmc.tsv"),
    "car":        os.path.join(CLF_DATA_DIR, "Medium_Datasets", "car.tsv"),
    "churn":      os.path.join(CLF_DATA_DIR, "Medium_Datasets", "churn.tsv"),
    # Classification Large
    "mushroom":   os.path.join(CLF_DATA_DIR, "Large_Datasets",  "mushroom.tsv"),
    "nursery":    os.path.join(CLF_DATA_DIR, "Large_Datasets",  "nursery.tsv"),
    "adult":      os.path.join(CLF_DATA_DIR, "Large_Datasets",  "adult.tsv"),
    "connect_4":  os.path.join(CLF_DATA_DIR, "Large_Datasets",  "connect-4.tsv"),
    # Regression Small
    "230_machine_cpu": os.path.join(REG_DATA_DIR, "Small_Datasets",  "230_machine_cpu.tsv"),
    "560_bodyfat":     os.path.join(REG_DATA_DIR, "Small_Datasets",  "560_bodyfat.tsv"),
    "547_no2":         os.path.join(REG_DATA_DIR, "Small_Datasets",  "547_no2.tsv"),
    "1028_SWD":        os.path.join(REG_DATA_DIR, "Small_Datasets",  "1028_SWD.tsv"),
    # Regression Medium
    "titanic":             os.path.join(REG_DATA_DIR, "Medium_Datasets", "titanic.tsv"),
    "529_pollen":          os.path.join(REG_DATA_DIR, "Medium_Datasets", "529_pollen.tsv"),
    "503_wind":            os.path.join(REG_DATA_DIR, "Medium_Datasets", "503_wind.tsv"),
    "294_satellite_image": os.path.join(REG_DATA_DIR, "Medium_Datasets", "294_satellite_image.tsv"),
    # Regression Large
    "225_puma8NH": os.path.join(REG_DATA_DIR, "Large_Datasets", "225_puma8NH.tsv"),
    "197_cpu_act": os.path.join(REG_DATA_DIR, "Large_Datasets", "197_cpu_act.tsv"),
    "537_houses":  os.path.join(REG_DATA_DIR, "Large_Datasets", "537_houses.tsv"),
    "344_mv":      os.path.join(REG_DATA_DIR, "Large_Datasets", "344_mv.tsv"),
}

SEEDS            = [42, 123, 456, 789, 999]
# tabddpm disabled — forestdiffusion environment issue.
# Re-enable: add "tabddpm" back when the package installs correctly.
METHODS          = ["baseline","smote_nc","ctgan","tvae","tabddpm","great","dice"]
SPLIT            = {"small": 0.30, "medium": 0.20, "large": 0.20}
CLF_AUG_STRATEGY = "auto"
REG_AUG_PCT      = 0.20
LARGE_SUBSAMPLE_N = 15_000
GAN_EPOCHS       = {"small": 100, "medium": 300, "large": 300}
GREAT_EPOCHS     = {"small": 25,  "medium": 75,  "large": 50}
DIFFUSION_STEPS  = {"small": 5000,"medium": 15000,"large": 30000}
SMOTE_K          = {"small": 3,   "medium": 5,   "large": 5}
DICE_NUM_CFS        = 3
DICE_PROXIMITY_W    = 0.5
DICE_DIVERSITY_W    = 1.0
DICE_REG_DELTA_MIN  = 2.0
DICE_REG_DELTA_COEF = 0.10
SHAP_SAMPLE_N    = 200
SHAP_TOP_K       = 5
RF_PARAMS  = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
XGB_PARAMS = {"n_estimators": 100, "random_state": 42,
              "eval_metric": "logloss", "verbosity": 0}
CLF_DATASETS = {
    "small":  ["heart_c","saheart","australian","german"],
    "medium": ["flare","cmc","car","churn"],
    "large":  ["mushroom","nursery","adult","connect_4"],
}
REG_DATASETS = {
    "small":  ["230_machine_cpu","560_bodyfat","547_no2","1028_SWD"],
    "medium": ["titanic","529_pollen","503_wind","294_satellite_image"],
    "large":  ["225_puma8NH","197_cpu_act","537_houses","344_mv"],
}

if __name__ == "__main__":
    print(f"BASE_DIR : {BASE_DIR}")
    print(f"OUTPUT   : {OUTPUT_ROOT}")
    print()
    missing = []
    for name, path in DATASET_PATHS.items():
        ok = os.path.isfile(path)
        print(f"  {'OK' if ok else 'MISSING':7s}  {name:30s}")
        if not ok:
            missing.append(name)
    print(f"\n{len(missing)} file(s) missing" if missing else "\nAll 24 dataset files found.")