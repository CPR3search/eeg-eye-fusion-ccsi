import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccsi import config
from ccsi.data import load_trials
from ccsi.features import add_differences, montage_indices
from ccsi.models import load_model
from ccsi.runs import RunStore, picks, run_id

MONTAGES = ["62", "32", "16", "8"]
N_WARMUP = 50
CPU_THREADS = 4


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "cpu"


def flops(model, x, mask, T: int, d_model: int) -> float:
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x, mask, 0.0), verbose=False)
    except Exception:
        macs = float("nan")
    return 2 * macs + 2 * T * d_model


def timing(model, eeg, eye, order, dev):
    if dev == "cpu":
        torch.set_num_threads(CPU_THREADS)
    model = model.to(dev).eval()

    def prep(i):
        t0 = time.perf_counter()
        x = np.concatenate([add_differences(eeg[i], order), add_differences(eye[i], order)], axis=1)
        return x, (time.perf_counter() - t0) * 1000

    def tensors(x):
        xt = torch.tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)
        return xt, torch.ones(1, xt.shape[1], device=dev)

    prep_ms, model_ms = [], []
    with torch.no_grad():
        for i in range(min(N_WARMUP, len(eeg))):
            model(*tensors(prep(i)[0]), 0.0)
        if dev == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for i in range(len(eeg)):
            x, ms = prep(i)
            prep_ms.append(ms)
            xt, mask = tensors(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(xt, mask, 0.0)
            if dev == "cuda":
                torch.cuda.synchronize()
            model_ms.append((time.perf_counter() - t0) * 1000)
    peak = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
    return prep_ms, model_ms, peak


def main():
    """Model size, FLOPs and per-trial latency of feature-level fusion with 62, 32, 16 and 8 channels, one checkpoint
    per target (source: the first other cohort in alphabetical order)."""
    pk, rows = picks(), []
    stores = {"62": RunStore("selection"), "other": RunStore("montages")}
    trials = {c: load_trials(c) for c in config.CULTURES}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for name in MONTAGES:
        idx = montage_indices(name)
        for t in config.CULTURES:
            s = sorted(c for c in config.CULTURES if c != t)[0]
            d, q = pk[t]
            ck = (stores["62"].checkpoint(run_id("feat62", s, t, config.SINGLE_RUN_SEED, d, q)) if name == "62"
                  else stores["other"].checkpoint(run_id(name, s, t, config.SINGLE_RUN_SEED)))
            eeg = [x["eeg"] if idx is None else x["eeg"][:, list(idx)] for x in trials[t]]
            eye = [x["eye"] for x in trials[t]]
            for dev in devices:
                model = load_model(ck, dev)
                if dev == "cpu":
                    n_params = sum(p.numel() for p in model.parameters())
                    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                    x0 = np.concatenate([add_differences(eeg[0], q), add_differences(eye[0], q)], axis=1)
                    x0 = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
                    fl = flops(model, x0, torch.ones(1, x0.shape[1]), eeg[0].shape[0], d)
                prep_ms, model_ms, peak = timing(model, eeg, eye, q, dev)
                for part, arr in (("model", model_ms), ("feature_prep", prep_ms)):
                    rows.append({"montage": name, "target": t, "source": s, "device": dev,
                                 "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else cpu_name(),
                                 "threads": CPU_THREADS if dev == "cpu" else "", "part": part,
                                 "p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)),
                                 "n_trials": len(eeg), "peak_memory_mb": peak, "model_bytes": n_bytes,
                                 "n_params": n_params, "flops": fl})
                print(f"{name:>2} channels, {t:8s} {dev:4s}: model p50 {np.percentile(model_ms, 50):.3f} ms", flush=True)
    out = config.results_dir("profile") / "profile.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
