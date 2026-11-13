import copy
import random
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccsi import config
from ccsi.data import subject_code
from ccsi.features import montage_indices
from ccsi.training import validation_subjects


# ------------------------------------------------------------------ window-level data shared by both comparators
@dataclass
class Windows:
    eeg: np.ndarray
    eye: np.ndarray
    trial_id: np.ndarray
    subject: np.ndarray
    label: np.ndarray | None
    trials: list = field(default_factory=list)
    trial_label: np.ndarray | None = None


def to_windows(trials: list[dict], culture: str, labelled: bool, montage: str = "62") -> Windows:
    idx = montage_indices(montage)
    eeg, eye, tid, subj, lab = [], [], [], [], []
    for k, t in enumerate(trials, start=1):
        e = t["eeg"] if idx is None else t["eeg"][:, list(idx)]
        n = e.shape[0]
        eeg.append(e)
        eye.append(t["eye"])
        tid.append(np.full(n, k))
        subj.append(np.full(n, subject_code(culture, t["subject"])))
        if labelled:
            lab.append(np.full(n, t["label"]))
    return Windows(np.concatenate(eeg).astype(np.float32), np.concatenate(eye).astype(np.float32),
                   np.concatenate(tid).astype(np.int64), np.concatenate(subj).astype(np.int64),
                   np.concatenate(lab).astype(np.int64) if labelled else None, trials,
                   np.array([t["label"] for t in trials], dtype=np.int64) if labelled else None)


def source_split(trials: list[dict], source: str, seed: int, montage: str = "62"):
    val = set(validation_subjects([subject_code(source, t["subject"]) for t in trials], seed))
    tr = [t for t in trials if subject_code(source, t["subject"]) not in val]
    va = [t for t in trials if subject_code(source, t["subject"]) in val]
    return to_windows(tr, source, True, montage), to_windows(va, source, True, montage), sorted(val)


def trial_probs(window_probs: np.ndarray, trial_id: np.ndarray, n_trials: int) -> np.ndarray:
    out = np.zeros((n_trials, window_probs.shape[1]))
    np.add.at(out, trial_id - 1, window_probs)
    counts = np.bincount(trial_id - 1, minlength=n_trials)[:, None]
    assert (counts > 0).all()
    return out / counts


def trial_accuracy(window_probs: np.ndarray, w: Windows) -> float:
    return float((trial_probs(window_probs, w.trial_id, len(w.trials)).argmax(1) == w.trial_label).mean())


# ------------------------------------------------------------------ DCCA-AM
DCCA_WIDTHS = [400, 200, 150, 120, 60, 20]
DCCA_GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
DCCA_LR, DCCA_BATCH, DCCA_EPOCHS, CCA_R = 1e-3, 100, 40, 1e-8


class _Mapping(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        dims = [in_dim] + DCCA_WIDTHS
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.sigmoid(x)
        return x


class DCCAAM(nn.Module):
    def __init__(self, eeg_dim: int, eye_dim: int, n_classes: int = 3):
        super().__init__()
        self.f_eeg, self.f_eye = _Mapping(eeg_dim), _Mapping(eye_dim)
        self.w_attn = nn.Parameter(torch.zeros(DCCA_WIDTHS[-1]))
        self.classifier = nn.Linear(DCCA_WIDTHS[-1], n_classes)

    def forward(self, eeg, eye):
        o1, o2 = self.f_eeg(eeg), self.f_eye(eye)
        alpha = F.softmax(torch.stack([o1 @ self.w_attn, o2 @ self.w_attn], dim=1), dim=1)
        return self.classifier(alpha[:, :1] * o1 + alpha[:, 1:] * o2), o1, o2, alpha


def _inv_sqrt(s, r):
    w, v = torch.linalg.eigh(s)
    return v @ torch.diag(torch.clamp(w, min=r).rsqrt()) @ v.T


def cca_loss(h1, h2, r: float = CCA_R):
    h1, h2 = h1.double(), h2.double()
    b = h1.shape[0]
    assert b >= 2
    h1, h2 = h1 - h1.mean(0, keepdim=True), h2 - h2.mean(0, keepdim=True)
    s11 = h1.T @ h1 / (b - 1) + r * torch.eye(h1.shape[1], dtype=h1.dtype, device=h1.device)
    s22 = h2.T @ h2 / (b - 1) + r * torch.eye(h2.shape[1], dtype=h2.dtype, device=h2.device)
    s12 = h1.T @ h2 / (b - 1)
    return -torch.linalg.svdvals(_inv_sqrt(s11, r) @ s12 @ _inv_sqrt(s22, r)).sum()


def _dcca_batches(n: int, gen: torch.Generator) -> list:
    perm = torch.randperm(n, generator=gen)
    out = [perm[i:i + DCCA_BATCH] for i in range(0, n, DCCA_BATCH)]
    if len(out) > 1 and len(out[-1]) < 2:
        out[-2] = torch.cat([out[-2], out[-1]])
        out = out[:-1]
    return out


@torch.no_grad()
def dcca_window_probs(model, eeg: np.ndarray, eye: np.ndarray, dev, chunk: int = 4096) -> np.ndarray:
    model.eval()
    return np.concatenate([F.softmax(model(torch.from_numpy(eeg[i:i + chunk]).to(dev),
                                           torch.from_numpy(eye[i:i + chunk]).to(dev))[0], dim=1).cpu().numpy()
                           for i in range(0, len(eeg), chunk)])


def train_dcca(tr: Windows, va: Windows, gamma: float, seed: int, dev):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = DCCAAM(tr.eeg.shape[1], tr.eye.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=DCCA_LR, weight_decay=0.0)
    xe, xy, y = torch.from_numpy(tr.eeg).to(dev), torch.from_numpy(tr.eye).to(dev), torch.from_numpy(tr.label).to(dev)
    best, best_state, best_epoch = -1.0, None, 0
    for epoch in range(DCCA_EPOCHS):
        model.train()
        for idx in _dcca_batches(len(y), gen):
            idx = idx.to(dev)
            logits, o1, o2, _ = model(xe[idx], xy[idx])
            ce = F.cross_entropy(logits, y[idx])
            lc = cca_loss(o1, o2)
            loss = gamma * lc + 1.0 * ce
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc =trial_accuracy(dcca_window_probs(model, va.eeg, va.eye, dev), va)
        if acc > best:
            best, best_epoch, best_state = acc, epoch + 1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, {"source_val_acc": best, "selected_epoch": best_epoch}


# ------------------------------------------------------------------ CSMM (third_party/CSMM, fetched and patched)
CSMM_STAGES = ("eegsann", "eyesann", "fusion")


def csmm_modules():
    path = str(config.ROOT / "third_party" / "CSMM")
    if not (config.ROOT / "third_party" / "CSMM" / "main_modalFusion.py").exists():
        raise SystemExit("CSMM is missing: run 'bash third_party/fetch_csmm.sh' first")
    if path not in sys.path:
        sys.path.insert(0, path)
    import main_SANN_IV as sann_eeg
    import main_SANN_IV_eye as sann_eye
    import main_modalFusion as fusion
    return sann_eeg, sann_eye, fusion


def seed_all(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return torch.Generator().manual_seed(seed)


def _domains(w: Windows, modality: str):
    x = w.eeg if modality == "eeg" else w.eye
    return [(int(c), torch.from_numpy(x[w.subject == c]), torch.from_numpy(w.label[w.subject == c]))
            for c in sorted(np.unique(w.subject))]


def csmm_sann(stage: str, tr: Windows, va: Windows, tw: Windows, seed: int, dev):
    sann_eeg, sann_eye, _ = csmm_modules()
    mod, modality = (sann_eeg, "eeg") if stage == "eegsann" else (sann_eye, "eye")
    gen = seed_all(seed)
    model = mod.make_model()
    vx = torch.from_numpy(va.eeg if modality == "eeg" else va.eye)
    tx = torch.from_numpy(tw.eeg if modality == "eeg" else tw.eye)

    def val_eval(m):
        return float((trial_probs(np.exp(mod.predict_logprob(m, vx, dev)), va.trial_id, len(va.trials)).argmax(1)
                      == va.trial_label).mean())

    state, info = mod.train_and_evaluate(model, _domains(tr, modality), tx, val_eval, dev, gen, n_epoch=mod.N_EPOCH)
    model.load_state_dict(state)
    return model, {"source_val_acc": info["source_val_acc"], "selected_epoch": info["selected_epoch"]}


def csmm_fusion(tr: Windows, va: Windows, tw: Windows, net_eeg, net_eye, seed: int, dev):
    _, _, fusion = csmm_modules()
    net_eeg, net_eye = net_eeg.to(dev).eval(), net_eye.to(dev).eval()
    vx_e, vx_y = torch.from_numpy(va.eeg), torch.from_numpy(va.eye)

    def val_eval(m, sc_e, sc_y):
        c = fusion.Predict(m, vx_e, vx_y, dev, net_eeg, net_eye, sc_e, sc_y)
        return float((trial_probs(fusion.combined_window_probs(c), va.trial_id, len(va.trials)).argmax(1)
                      == va.trial_label).mean())

    gen = seed_all(seed)
    state, info, sc_e, sc_y = fusion.main_seed(fusion.CONFIG, net_eeg, net_eye, torch.from_numpy(tr.eeg),
                                               torch.from_numpy(tr.eye), tr.label, tr.trial_id,
                                               torch.from_numpy(tw.eeg), torch.from_numpy(tw.eye), val_eval, dev, gen,
                                               n_epochs=fusion.FUSION_EPOCHS)
    model = fusion.EFCL_seed_final(config=fusion.CONFIG).to(dev)
    model.load_state_dict(state)
    tc = fusion.Predict(model, torch.from_numpy(tw.eeg), torch.from_numpy(tw.eye), dev, net_eeg, net_eye, sc_e, sc_y)
    probs = trial_probs(fusion.combined_window_probs(tc), tw.trial_id, len(tw.trials))
    return model, {"source_val_acc": info["source_val_acc"], "selected_epoch": info["selected_epoch"]}, probs
