import os

import numpy as np
import torch
import torch.nn.functional as F

from ccsi.models import FeatureFusionModel, RandomFourierFeatures, median_sigma

N_EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 2.0
N_CLASSES = 3
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def enable_determinism(threads: int = 4) -> None:
    assert not torch.cuda.is_initialized(), "call enable_determinism() before any CUDA operation"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)


def validation_subjects(subjects, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    pool = np.unique(np.asarray(subjects))
    rng.shuffle(pool)
    return sorted(int(s) for s in pool[:max(1, len(pool) // 5)])


def pad_batch(seqs: list[np.ndarray], device: str):
    T, B, D = max(s.shape[0] for s in seqs), len(seqs), seqs[0].shape[1]
    x = torch.zeros(B, T, D, dtype=torch.float32)
    mask = torch.zeros(B, T, dtype=torch.float32)
    for i, s in enumerate(seqs):
        x[i, :s.shape[0]] = torch.tensor(s, dtype=torch.float32)
        mask[i, :s.shape[0]] = 1.0
    return x.to(device), mask.to(device)


def focal_loss(logits, y, gamma: float):
    ce = F.cross_entropy(logits, y, weight=None, reduction="none", label_smoothing=0.0)
    return ((1 - torch.exp(-ce)) ** gamma * ce).mean()


def lmmd_loss(rff, z_s, y_s, z_t, p_t, n_classes: int = N_CLASSES):
    phi_s, phi_t = rff(z_s), rff(z_t)
    total = 0.0
    for c in range(n_classes):
        w_s = (y_s == c).float()
        w_s = w_s / w_s.sum().clamp(min=1e-6)
        w_t = p_t[:, c]
        w_t = w_t / w_t.sum().clamp(min=1e-6)
        total = total + (((w_s.unsqueeze(1) * phi_s).sum(dim=0) - (w_t.unsqueeze(1) * phi_t).sum(dim=0)) ** 2).sum()
    return total / n_classes


def _cycling_batches(pool: np.ndarray, batch_size: int, rng: np.random.Generator):
    pool = np.asarray(pool)
    buffer = np.array([], dtype=pool.dtype)
    while True:
        if len(buffer) < batch_size:
            left = set(buffer.tolist())
            rest = np.array([x for x in pool if x not in left], dtype=pool.dtype)
            buffer = np.concatenate([buffer, np.concatenate([rng.permutation(rest), rng.permutation(buffer)])
                                     if len(buffer) else rng.permutation(pool)])
        yield buffer[:batch_size].copy()
        buffer = buffer[batch_size:]


def _val_accuracy(model, seqs, labels, idx, batch_size, dev) -> float:
    correct, total = 0, 0
    with torch.no_grad():
        for b in range(0, len(idx), batch_size):
            i = idx[b:b + batch_size]
            if len(i) == 0:
                continue
            x, m = pad_batch([seqs[j] for j in i], dev)
            correct += (model(x, m, lambda_=0.0)[0].argmax(dim=1) == torch.tensor(labels[i], dtype=torch.long).to(dev)).sum().item()
            total += len(i)
    return correct / max(total, 1)


def train(source_seqs: list, source_labels: np.ndarray, source_subjects: np.ndarray, target_seqs: list, seed: int,
          d_model: int, align: bool = True, model_factory=None, steps_per_epoch: int | None = None,
          n_epochs: int = N_EPOCHS, batch_size: int = BATCH_SIZE, dev: str | None = None):
    """Trains on labelled source trials and unlabelled target trials; selects the epoch by source-validation accuracy."""
    torch.manual_seed(seed)
    dev = dev or device()
    rng = np.random.default_rng(seed)
    pool = np.unique(source_subjects)
    rng.shuffle(pool)
    val_subjects = set(pool[:max(1, len(pool) // 5)].tolist())
    is_val = np.isin(source_subjects, list(val_subjects))
    train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]
    lmmd_lambda = 1.0 if align else 0.0

    in_dim = source_seqs[0].shape[1]
    model = (model_factory(in_dim) if model_factory else FeatureFusionModel(in_dim, d_model)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    with torch.no_grad():
        xb, mb = pad_batch([source_seqs[i] for i in train_idx[:32]], dev)
        z = model(xb, mb, lambda_=0.0)[2]
        sigma = median_sigma(z)
    rff = RandomFourierFeatures(d_in=z.shape[1], sigma=sigma).to(dev)

    if steps_per_epoch is None:
        per_epoch = min(max(1, len(train_idx) // batch_size), max(1, len(target_seqs) // batch_size))
    else:
        assert batch_size <= len(train_idx) and batch_size <= len(target_seqs)
        per_epoch = steps_per_epoch
        src_cycle = _cycling_batches(train_idx, batch_size, rng)
        tgt_cycle = _cycling_batches(np.arange(len(target_seqs)), batch_size, rng)
    total_steps, step = n_epochs * per_epoch, 0
    best_acc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(n_epochs):
        model.train()
        order = rng.permutation(train_idx)
        t_order = rng.permutation(len(target_seqs))
        if steps_per_epoch is None:
            batches = [(order[b * batch_size:(b + 1) * batch_size], t_order[b * batch_size:(b + 1) * batch_size])
                       for b in range(per_epoch)]
        else:
            batches = [(next(src_cycle), next(tgt_cycle)) for _ in range(per_epoch)]
        for s_idx, t_idx in batches:
            if len(s_idx) == 0 or len(t_idx) == 0:
                continue
            xs, ms = pad_batch([source_seqs[i] for i in s_idx], dev)
            ys = torch.tensor(source_labels[s_idx], dtype=torch.long).to(dev)
            xt, mt = pad_batch([target_seqs[i] for i in t_idx], dev)
            p = step / max(total_steps, 1)
            lam = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            logit_s, dom_s, z_s = model(xs, ms, lam)
            logit_t, dom_t, z_t = model(xt, mt, lam)
            loss = focal_loss(logit_s, ys, FOCAL_GAMMA)
            sub = lmmd_loss(rff, z_s, ys, z_t, F.softmax(logit_t, dim=1))
            if align:
                dl = torch.cat([torch.zeros(len(s_idx), dtype=torch.long), torch.ones(len(t_idx), dtype=torch.long)]).to(dev)
                loss = loss + F.cross_entropy(torch.cat([dom_s, dom_t], dim=0), dl) + lmmd_lambda * sub
            else:
                loss = loss + lmmd_lambda * sub
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        model.eval()
        acc = _val_accuracy(model, source_seqs, source_labels, val_idx, batch_size, dev)
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    info = {"source_val_acc": best_acc, "selected_epoch": best_epoch, "total_updates": step,
            "val_subjects": ";".join(str(int(x)) for x in sorted(val_subjects)),
            "n_params": sum(p.numel() for p in model.parameters())}
    return model, info


def predict_proba(model, seqs: list, batch_size: int = BATCH_SIZE, dev: str | None = None) -> np.ndarray:
    dev = dev or device()
    model = model.to(dev).eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(seqs), batch_size):
            x, m = pad_batch(seqs[b:b + batch_size], dev)
            out.append(F.softmax(model(x, m, lambda_=0.0)[0], dim=1).cpu().numpy())
    return np.concatenate(out)
