import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("SEED_DATASET_ROOT", str(ROOT.parent / "SEED-Dataset" / "Data")))
RESULTS = Path(os.environ.get("CCSI_RESULTS", str(ROOT / "results")))
CONFIGS = ROOT / "configs"

CULTURES = ["chinese", "german", "french"]
CODE = {"chinese": 1, "german": 2, "french": 3}
DIRECTIONS = [(s, t) for t in CULTURES for s in CULTURES if s != t]
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
N_CHANNELS = 62
EEG_DIM = N_CHANNELS * len(BAND_ORDER)
EYE_DIM = 33
WINDOW_SECONDS = 4.0
SINGLE_RUN_SEED = 2

DE_FEATURE_DIRS = {
    "chinese": DATASET_ROOT / "SEED" / "SEED_Multimodel" / "02-EEG-DE-feature",
    "german": DATASET_ROOT / "SEED-GER" / "German" / "02-EEG-DE-features",
    "french": DATASET_ROOT / "SEED-FRA" / "French" / "02-EEG-DE-features",
}
EYE_FEATURE_DIRS = {
    "chinese": DATASET_ROOT / "SEED" / "SEED_Multimodel" / "04-Eye-tracking-feature" / "eye_tracking_feature",
    "german": DATASET_ROOT / "SEED-GER" / "German" / "04-Eye-tracking-feature" / "eye_tracking_feature",
    "french": DATASET_ROOT / "SEED-FRA" / "French" / "04-Eye-tracking-features" / "eye_tracking_feature",
}
INFO_XLSX = {
    "chinese": DATASET_ROOT / "SEED" / "SEED_Multimodel" / "China_information.xlsx",
    "german": DATASET_ROOT / "SEED-GER" / "German_information.xlsx",
}
INFO_SHEET = {"chinese": "Chinese_exp_info", "german": "German_exp_info", "french": "French_exp_info"}
SEED_FRA_ZIP = DATASET_ROOT / "SEED-FRA" / "SEED_FRA.zip"
FRENCH_INFO_MEMBER = "French_information.xlsx"
FRENCH_ZIP = DATASET_ROOT / "SEED-FRA" / "French" / "French.zip"
CHANNEL_LOCS = DATASET_ROOT / "SEED" / "channel_62_pos.locs"


def seeds() -> list[int]:
    out = [int(x) for x in (CONFIGS / "seeds.txt").read_text().split()]
    assert len(out) == 5, out
    return out


def results_dir(*parts: str) -> Path:
    d = RESULTS.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
