import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ccsi import config
from ccsi.data import subject_code
from ccsi.features import build_input, input_dims
from ccsi.models import EncoderFusionModel, matched_branch_width
from ccsi.training import predict_proba, train

KEYS = ["culture", "subject", "session", "trial_index"]
PROBS = ["p_negative", "p_neutral", "p_positive"]
GRID = [(32, 1), (32, 2), (64, 1), (64, 2), (128, 1), (128, 2)]

# modality, EEG montage, architecture (feature- or encoder-level fusion), domain alignment
ARMS = {
    "feat62": ("fused", "62", "feature", True),
    "enc62": ("fused", "62", "encoder", True),
    "eeg62": ("eeg", "62", "feature", True),
    "eye": ("eye", "62", "feature", True),
    "feat8": ("fused", "8", "feature", True),
    "eeg8": ("eeg", "8", "feature", True),
    "enc8": ("fused", "8", "encoder", True),
    "noalign62": ("fused", "62", "feature", False),
}


class RunStore:
    """One directory per experiment: checkpoints/, predictions/ (target trials, no labels) and records/."""

    def __init__(self, name: str):
        self.root = config.results_dir(name)
        for d in ("checkpoints", "predictions", "records"):
            (self.root / d).mkdir(exist_ok=True)

    def done(self, run_id: str, expect: dict | None = None, artifacts=("checkpoint", "predictions")) -> bool:
        """True if the run finished: its record exists, was produced with the expected settings, and its files exist.
        A run with a record but missing files is trained again; a run produced with other settings stops the step."""
        path = self.root / "records" / f"{run_id}.json"
        if not path.exists():
            return False
        rec = json.loads(path.read_text())
        bad = {k: (rec.get(k), v) for k, v in (expect or {}).items() if str(rec.get(k)) != str(v)}
        if bad:
            raise SystemExit(f"{path} was produced with other settings {bad}; remove {self.root} to retrain")
        files = [self.checkpoint(run_id) if a == "checkpoint" else self.prediction_path(run_id if a == "predictions" else a)
                 for a in artifacts]
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"{run_id}: incomplete ({', '.join(f.name for f in missing)} missing), training again", flush=True)
            return False
        return True

    def prediction_path(self, name: str) -> Path:
        return self.root / "predictions" / f"{name}.csv"

    def checkpoint(self, run_id: str) -> Path:
        return self.root / "checkpoints" / f"{run_id}.pt"

    def save(self, run_id: str, record: dict, trials=None, probs=None, model=None, meta=None) -> dict:
        if model is not None:
            torch.save({"state_dict": model.state_dict(), **(meta or {})}, self.checkpoint(run_id))
        if probs is not None:
            write_predictions(self.root / "predictions" / f"{run_id}.csv", trials, probs)
        record = {"run_id": run_id, **record}
        path = self.root / "records" / f"{run_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, default=str))
        tmp.replace(path)
        return record

    def write(self, name: str, trials: list[dict], probs: np.ndarray) -> None:
        write_predictions(self.root / "predictions" / f"{name}.csv", trials, probs)

    def records(self) -> pd.DataFrame:
        return pd.DataFrame([json.loads(p.read_text()) for p in sorted((self.root / "records").glob("*.json"))])

    def predictions(self, run_id: str) -> pd.DataFrame:
        return pd.read_csv(self.prediction_path(run_id))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_predictions(path: Path, trials: list[dict], probs: np.ndarray) -> None:
    assert all("label" not in t for t in trials), "predictions are written without labels"
    pd.DataFrame({**{k: [t[k] for t in trials] for k in KEYS}, "pred": probs.argmax(axis=1),
                  **{c: probs[:, i] for i, c in enumerate(PROBS)},
                  "n_windows": [t["n_windows"] for t in trials]}).to_csv(path, index=False)


def picks() -> dict[str, tuple[int, int]]:
    sel = json.loads((config.RESULTS / "selection" / "selection.json").read_text())
    return {t: (p["d_model"], p["difference_order"]) for t, p in sel["picks"].items()}


def run_id(arm: str, source: str, target: str, seed: int, d: int | None = None, q: int | None = None) -> str:
    cell = "" if d is None else f"_d{d}_q{q}"
    return f"{arm}{cell}_{source}_to_{target}_seed{seed}"


def train_transductive(arm: str, source: str, target: str, seed: int, d_model: int, order: int,
                       source_trials: list[dict], target_trials: list[dict], steps_per_epoch=None, spec=None):
    """Trains one model on labelled source trials and unlabelled target trials; returns model, info, meta, inputs."""
    assert all("label" not in t for t in target_trials), "target trials must be label-free"
    modality, montage, arch, align = spec or ARMS[arm]
    seqs = [build_input(t, modality, order, montage) for t in source_trials]
    labels = np.array([t["label"] for t in source_trials])
    subjects = np.array([subject_code(source, t["subject"]) for t in source_trials])
    tseqs = [build_input(t, modality, order, montage) for t in target_trials]
    in_dim, eeg_dim = input_dims(modality, montage, order)
    assert seqs[0].shape[1] == in_dim
    factory, width = None, None
    if arch == "encoder":
        width = matched_branch_width(in_dim, eeg_dim, d_model)
        factory = lambda n, w=width, e=eeg_dim: EncoderFusionModel(n, d_branch=w, eeg_dim=e)
    model, info = train(seqs, labels, subjects, tseqs, seed=seed, d_model=d_model, align=align, model_factory=factory,
                        steps_per_epoch=steps_per_epoch)
    meta = {"arch": arch, "input_dim": in_dim, "d_model": d_model, "eeg_dim": eeg_dim, "d_branch": width,
            "modality": modality, "montage": montage, "difference_order": order}
    info = {"arm": arm, "source": source, "target": target, "seed": seed, "d_model": d_model,
            "difference_order": order, "modality": modality, "montage": montage, "architecture": arch,
            "align": align, "branch_width": width or "", **info}
    return model, info, meta, tseqs


def run_transductive(store: RunStore, rid: str, arm: str, source: str, target: str, seed: int, d_model: int,
                     order: int, source_trials: list[dict], target_trials: list[dict], spec=None) -> None:
    modality, montage, arch, align = spec or ARMS[arm]
    expect = {"arm": arm, "source": source, "target": target, "seed": seed, "d_model": d_model,
              "difference_order": order, "modality": modality, "montage": montage, "architecture": arch, "align": align}
    if store.done(rid, expect):
        return
    model, info, meta, tseqs = train_transductive(arm, source, target, seed, d_model, order, source_trials,
                                                  target_trials, spec=spec)
    store.save(rid, info, target_trials, predict_proba(model, tseqs), model, meta)
    print(f"{rid}: source val {info['source_val_acc']:.4f}, epoch {info['selected_epoch']}", flush=True)
