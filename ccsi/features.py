import csv
from functools import lru_cache

import numpy as np

from ccsi import config


def add_differences(x: np.ndarray, order: int) -> np.ndarray:
    if order == 0:
        return x
    d1 = np.diff(x, axis=0, prepend=x[0:1])
    if order == 1:
        return np.concatenate([x, d1], axis=1)
    d2 = np.diff(d1, axis=0, prepend=d1[0:1])
    return np.concatenate([x, d1, d2], axis=1)


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if line.strip() and not line.startswith("#")))


@lru_cache(maxsize=None)
def montage_indices(name: str) -> tuple[int, ...] | None:
    if name == "62":
        return None
    return tuple(int(r["index_in_block"]) for r in read_rows(config.CONFIGS / "channels" / f"{name}.txt"))


def montage_size(name: str) -> int:
    idx = montage_indices(name)
    return config.N_CHANNELS if idx is None else len(idx) // len(config.BAND_ORDER)


@lru_cache(maxsize=None)
def channel_names() -> tuple[str, ...]:
    with open(config.CHANNEL_LOCS) as f:
        return tuple(line.split()[3] for line in f if line.strip())


@lru_cache(maxsize=None)
def eye_groups() -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for r in read_rows(config.CONFIGS / "eye_groups.txt"):
        groups.setdefault(r["group"], []).append(int(r["index"]))
    return groups


def build_input(trial: dict, modality: str, order: int, montage: str = "62",
                eeg_tf=None, eye_tf=None, pre_montage_eeg_tf=None) -> np.ndarray:
    eeg = trial["eeg"] if pre_montage_eeg_tf is None else pre_montage_eeg_tf(trial["eeg"])
    idx = montage_indices(montage)
    if idx is not None:
        eeg = eeg[:, list(idx)]
    eye = trial["eye"]
    if eeg_tf is not None:
        eeg = eeg_tf(eeg)
    if eye_tf is not None:
        eye = eye_tf(eye)
    if modality == "eeg":
        return add_differences(eeg, order)
    if modality == "eye":
        return add_differences(eye, order)
    return np.concatenate([add_differences(eeg, order), add_differences(eye, order)], axis=1)


def input_dims(modality: str, montage: str, order: int) -> tuple[int, int]:
    eeg = montage_size(montage) * len(config.BAND_ORDER) * (order + 1) if modality != "eye" else 0
    eye = config.EYE_DIM * (order + 1) if modality != "eeg" else 0
    return eeg + eye, eeg
