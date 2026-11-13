import pickle
import re
import zipfile
from functools import lru_cache
from io import BytesIO

import numpy as np
import pandas as pd

from ccsi import config

SESSION_FILE = re.compile(r"^(\d+)_(\d+)\.npz$")


def sessions(culture: str) -> list[tuple[int, int]]:
    out = []
    for f in (config.DE_FEATURE_DIRS[culture] / "eeg_used_4s").glob("*.npz"):
        m = SESSION_FILE.match(f.name)
        if not m:
            raise ValueError(f"unexpected file {f}")
        out.append((int(m.group(1)), int(m.group(2))))
    return sorted(out)


def load_de(culture: str, subject: int, session: int) -> dict:
    path = config.DE_FEATURE_DIRS[culture] / "eeg_used_4s" / f"{subject}_{session}.npz"
    npz = np.load(path)
    train, test = pickle.loads(npz["train_data"]), pickle.loads(npz["test_data"])
    n_train, n_test = train[config.BAND_ORDER[0]].shape[0], test[config.BAND_ORDER[0]].shape[0]
    power = np.empty((n_train + n_test, config.N_CHANNELS, len(config.BAND_ORDER)), dtype=np.float64)
    for b, band in enumerate(config.BAND_ORDER):
        if train[band].shape != (n_train, config.N_CHANNELS):
            raise ValueError(f"{path}: unexpected shape {train[band].shape} for band {band}")
        power[:n_train, :, b] = train[band]
        power[n_train:, :, b] = test[band]
    label = np.concatenate([npz["train_label"], npz["test_label"]]).astype(int)
    if not set(np.unique(label)) <= {0, 1, 2}:
        raise ValueError(f"{path}: unexpected labels {np.unique(label)}")
    return {"power": power, "label": label}


def load_eye(culture: str, subject: int, session: int) -> np.ndarray:
    with open(config.EYE_FEATURE_DIRS[culture] / f"{subject}_{session}", "rb") as f:
        d = pickle.load(f)
    return np.concatenate([d["train_data_eye"], d["test_data_eye"]], axis=0)


@lru_cache(maxsize=None)
def trial_info(culture: str) -> pd.DataFrame:
    if culture == "french":
        with zipfile.ZipFile(config.SEED_FRA_ZIP) as z:
            xl = pd.ExcelFile(BytesIO(z.read(config.FRENCH_INFO_MEMBER)))
    else:
        xl = pd.ExcelFile(config.INFO_XLSX[culture])
    sheet = config.INFO_SHEET[culture]
    raw = xl.parse(sheet, header=None)
    header = next(i for i in range(min(5, len(raw))) if "No. in stimuli" in raw.iloc[i].values)
    df = xl.parse(sheet, header=header)
    df = df[df["label"].notna()].copy()
    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) > 0 and vals.astype(str).eq("No use").all():
            df = df[df[col].isna()]
            break
    df["duration_seconds"] = df["Trial End second"].astype(float) - df["Trial Start second"].astype(float)
    if (df["duration_seconds"] <= 0).any():
        raise ValueError(f"non-positive clip duration for {culture}")
    out = df[["No. in stimuli", "label", "duration_seconds"]].rename(columns={"No. in stimuli": "trial_index"})
    out["trial_index"] = out["trial_index"].astype(int)
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)


def timing_epochs(culture: str, n_windows: int, scale_to_windows: bool = True) -> list[dict]:
    info = trial_info(culture)
    dur = info.duration_seconds.to_numpy(dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(dur)])
    if scale_to_windows:
        bounds = np.rint(cum / cum[-1] * n_windows).astype(int)
    else:
        bounds = np.minimum(np.rint(cum / config.WINDOW_SECONDS).astype(int), n_windows)
    return [{"label": int(info.label.iloc[k]) + 1, "start_idx": int(bounds[k]), "end_idx": int(bounds[k + 1]),
             "clip_number": int(info.trial_index.iloc[k])} for k in range(len(dur))]


def label_runs(label: np.ndarray) -> list[dict]:
    runs, start = [], 0
    for i in range(1, len(label) + 1):
        if i == len(label) or label[i] != label[start]:
            runs.append({"label": int(label[start]), "start_idx": start, "end_idx": i})
            start = i
    return runs


def check_label_runs(culture: str, runs: list[dict]) -> dict:
    expected = []
    for _, row in trial_info(culture).iterrows():
        lab = row["label"] + 1
        if expected and expected[-1]["label"] == lab:
            expected[-1]["duration_seconds"] += row["duration_seconds"]
        else:
            expected.append({"label": lab, "duration_seconds": row["duration_seconds"]})
    if [e["label"] for e in expected] != [r["label"] for r in runs]:
        return {"passed": False, "reason": "label sequence differs from the information sheet"}
    gap = [abs(e["duration_seconds"] / config.WINDOW_SECONDS - (r["end_idx"] - r["start_idx"]))
           for e, r in zip(expected, runs)]
    return {"passed": float(np.mean(gap)) * config.WINDOW_SECONDS <= 8.0, "reason": "",
            "mean_discrepancy_windows": float(np.mean(gap)), "max_discrepancy_windows": float(np.max(gap))}


def zscore(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - mu) / (sd + eps)


def normalize_session(x: np.ndarray) -> np.ndarray:
    return zscore(x, x.mean(axis=0), x.std(axis=0))


def raw_session(culture: str, subject: int, session: int) -> tuple[np.ndarray, np.ndarray]:
    power = load_de(culture, subject, session)["power"]
    eeg = power.reshape(power.shape[0], -1)
    eye = load_eye(culture, subject, session)
    if eye.shape[0] != eeg.shape[0]:
        raise ValueError(f"{culture} {subject}_{session}: EEG and eye window counts differ")
    return eeg, eye


def split_trials(culture: str, subject: int, session: int, eeg: np.ndarray, eye: np.ndarray) -> list[dict]:
    out = []
    for k, e in enumerate(timing_epochs(culture, eeg.shape[0])):
        s, t = e["start_idx"], e["end_idx"]
        out.append({"culture": culture, "subject": subject, "session": session, "trial_index": k,
                    "label": int(e["label"]), "n_windows": t - s, "eeg": eeg[s:t], "eye": eye[s:t]})
    return out


@lru_cache(maxsize=None)
def _trials(culture: str) -> tuple:
    out = []
    for subject, session in sessions(culture):
        eeg, eye = raw_session(culture, subject, session)
        out += split_trials(culture, subject, session, normalize_session(eeg), normalize_session(eye))
    return tuple(out)


def load_trials(culture: str) -> list[dict]:
    return list(_trials(culture))


def strip_labels(trials: list[dict]) -> list[dict]:
    return [{k: v for k, v in t.items() if k != "label"} for t in trials]


def unlabelled_trials(culture: str) -> list[dict]:
    return strip_labels(load_trials(culture))


def subject_code(culture: str, subject: int) -> int:
    return config.CODE[culture] * 1000 + subject


def holdout_split(culture: str) -> dict:
    by_subject: dict[int, list[int]] = {}
    for subject, session in sessions(culture):
        by_subject.setdefault(subject, []).append(session)
    adaptation, other_stats, own_stats = [], [], []
    for subject, ss in by_subject.items():
        ss = sorted(ss)
        held, adapt = ss[-1], ss[:-1]
        raw = [raw_session(culture, subject, s) for s in adapt]
        pool_eeg = np.concatenate([r[0] for r in raw], axis=0)
        pool_eye = np.concatenate([r[1] for r in raw], axis=0)
        for (eeg, eye), s in zip(raw, adapt):
            adaptation += split_trials(culture, subject, s, normalize_session(eeg), normalize_session(eye))
        eeg, eye = raw_session(culture, subject, held)
        other_stats += split_trials(culture, subject, held, zscore(eeg, pool_eeg.mean(0), pool_eeg.std(0)),
                                    zscore(eye, pool_eye.mean(0), pool_eye.std(0)))
        own_stats += split_trials(culture, subject, held, normalize_session(eeg), normalize_session(eye))
    assert not {(t["subject"], t["session"]) for t in adaptation} & {(t["subject"], t["session"]) for t in own_stats}
    return {"adaptation": adaptation, "other_sessions": other_stats, "own_session": own_stats}
