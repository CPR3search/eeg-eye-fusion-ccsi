import hashlib

import numpy as np

from ccsi import config
from ccsi.features import eye_groups

DROPOUT = {"62": [4, 8, 16, 31], "8": [4, 8]}
SIGMAS = [0.1, 0.25, 0.5, 1.0]
N_DRAWS = 20
SHIFTS = [-2, -1, 1, 2]
N_BANDS = len(config.BAND_ORDER)


def derived_rng(*parts) -> np.random.Generator:
    key = "|".join(str(p) for p in parts)
    return np.random.default_rng(int(hashlib.sha256(f"12345|{key}".encode()).hexdigest()[:8], 16))


def drop_channels(eeg: np.ndarray, channels: list[int]) -> np.ndarray:
    out = eeg.copy()
    for i in channels:
        out[:, i * N_BANDS:(i + 1) * N_BANDS] = 0.0
    return out


def drop_band(eeg: np.ndarray, n_channels: int, band: int) -> np.ndarray:
    out = eeg.copy()
    out[:, [c * N_BANDS + band for c in range(n_channels)]] = 0.0
    return out


def add_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return x + rng.normal(0.0, sigma, size=x.shape)


def drop_features(eye: np.ndarray, idx: list[int]) -> np.ndarray:
    out = eye.copy()
    out[:, idx] = 0.0
    return out


def shift(eye: np.ndarray, k: int) -> np.ndarray:
    T, out = eye.shape[0], np.empty_like(eye)
    if k > 0:
        out[k:] = eye[:T - k]
        out[:k] = eye[0]
    else:
        out[:T + k] = eye[-k:]
        out[T + k:] = eye[-1]
    return out


def fault_specs(montage: str) -> list[dict]:
    """The fault conditions in plan order. make() returns the transforms passed to build_input for one condition."""
    nk, specs = int(montage), []

    def add(fault, modality, level, draw, units, make, draw_seed=""):
        specs.append({"fault": fault, "modality": modality, "level": level, "draw": draw, "affected_units": units,
                      "draw_seed": draw_seed, "make": make})

    for k in DROPOUT[montage]:
        rng = derived_rng("channel_dropout", montage, k)
        for d in range(N_DRAWS):
            drop = sorted(rng.choice(nk, size=k, replace=False).tolist())
            add("channel_dropout", "eeg", k, d, ",".join(map(str, drop)),
                lambda dd=drop: {"eeg_tf": lambda x: drop_channels(x, dd)})
    for b, band in enumerate(config.BAND_ORDER):
        add("band_loss", "eeg", band, 0, band, lambda b=b: {"eeg_tf": lambda x: drop_band(x, nk, b)})
    for modality in ("eeg", "eye"):
        for sigma in SIGMAS:
            rng = derived_rng("gaussian_noise", modality, sigma)
            for d in range(N_DRAWS):
                seed = int(rng.integers(0, 2**31))
                key = "pre_montage_eeg_tf" if modality == "eeg" else "eye_tf"

                def make(seed=seed, sigma=sigma, key=key):
                    r = np.random.default_rng(seed)
                    return {key: lambda x: add_noise(x, sigma, r)}
                add("gaussian_noise", modality, sigma, d, f"sigma={sigma}", make, seed)
    for group, idx in eye_groups().items():
        add("eye_group_loss", "eye", group, 0, group, lambda gi=idx: {"eye_tf": lambda x: drop_features(x, gi)})
    add("whole_modality_loss", "eeg", "eeg_absent", 0, "all_eeg", lambda: {"eeg_tf": np.zeros_like})
    add("whole_modality_loss", "eye", "eye_absent", 0, "all_eye", lambda: {"eye_tf": np.zeros_like})
    for k in SHIFTS:
        add("eeg_eye_misalignment", "eye", k, 0, f"shift={k}", lambda s=k: {"eye_tf": lambda x: shift(x, s)})
    return specs
