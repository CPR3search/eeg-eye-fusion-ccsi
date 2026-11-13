import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccsi import config
from ccsi.comparators import (CSMM_STAGES, DCCA_GAMMAS, DCCAAM, csmm_fusion, csmm_modules, csmm_sann, dcca_window_probs,
                              source_split, to_windows, train_dcca, trial_probs)
from ccsi.data import load_trials, unlabelled_trials
from ccsi.runs import RunStore, run_id
from ccsi.training import device, enable_determinism


def dcca(src, tgt, dev):
    """DCCA-AM: the correlation-loss weight is selected per target by source validation (first seed); the final
    model of the first seed is the selected selection model, the other seeds are trained with the selected weight."""
    store, seeds = RunStore("dcca"), config.seeds()
    sel_id = lambda g, s, t: f"dcca62_g{g}_{s}_to_{t}_seed{seeds[0]}_selection"

    def fit(s, t, seed, gamma):
        tr, va, val = source_split(src[s], s, seed)
        model, info = train_dcca(tr, va, gamma, seed, dev)
        meta = {"eeg_dim": tr.eeg.shape[1], "eye_dim": tr.eye.shape[1]}
        rec = {"arm": "dcca62", "source": s, "target": t, "seed": seed, "gamma": gamma,
               "val_subjects": ";".join(map(str, val)), "n_params": sum(p.numel() for p in model.parameters()), **info}
        return model, meta, rec

    for s, t in config.DIRECTIONS:
        for g in DCCA_GAMMAS:
            if not store.done(sel_id(g, s, t), {"source": s, "target": t, "seed": seeds[0], "gamma": g}, ["checkpoint"]):
                model, meta, rec = fit(s, t, seeds[0], g)
                store.save(sel_id(g, s, t), rec, model=model, meta=meta)
                print(f"{sel_id(g, s, t)}: source val {rec['source_val_acc']:.4f}", flush=True)
    rec = store.records().set_index("run_id")
    picks = {}
    for t in config.CULTURES:
        mean = {g: float(np.mean([rec.loc[sel_id(g, s, t), "source_val_acc"] for s in config.CULTURES if s != t]))
                for g in DCCA_GAMMAS}
        picks[t] = min(g for g in DCCA_GAMMAS if mean[g] == max(mean.values()))
    (store.root / "selection.json").write_text(json.dumps({"picks": picks}, indent=2))

    for seed in seeds:
        for s, t in config.DIRECTIONS:
            rid = run_id("dcca62", s, t, seed)
            if store.done(rid, {"source": s, "target": t, "seed": seed, "gamma": picks[t]},
                          ["predictions"] if seed == seeds[0] else ["checkpoint", "predictions"]):
                continue
            if seed == seeds[0]:
                ck = torch.load(store.checkpoint(sel_id(picks[t], s, t)), map_location=dev)
                model = DCCAAM(ck["eeg_dim"], ck["eye_dim"]).to(dev)
                model.load_state_dict(ck["state_dict"])
                r = {**rec.loc[sel_id(picks[t], s, t)].to_dict(), "reused_from": sel_id(picks[t], s, t)}
                meta = None
            else:
                model, meta, r = fit(s, t, seed, picks[t])
            tw = to_windows(tgt[t], t, False)
            probs = trial_probs(dcca_window_probs(model, tw.eeg, tw.eye, dev), tw.trial_id, len(tw.trials))
            store.save(rid, r, tgt[t], probs, model if meta else None, meta)
            print(f"{rid}: source val {r['source_val_acc']:.4f}", flush=True)


def csmm(src, tgt, dev):
    """CSMM: EEG and eye subject-aligned networks, then the contrastive fusion stage on their aligned features."""
    store = RunStore("csmm")
    sann_eeg, sann_eye, _ = csmm_modules()
    for seed in config.seeds():
        for s, t in config.DIRECTIONS:
            tr, va, val = source_split(src[s], s, seed)
            tw = to_windows(tgt[t], t, False)
            base = run_id("csmm62", s, t, seed)
            for stage in CSMM_STAGES:
                rid = base if stage == "fusion" else f"{base}_{stage}"
                if store.done(rid, {"source": s, "target": t, "seed": seed, "stage": stage},
                              ["checkpoint", "predictions"] if stage == "fusion" else ["checkpoint"]):
                    continue
                rec = {"arm": "csmm62", "stage": stage, "source": s, "target": t, "seed": seed,
                       "val_subjects": ";".join(map(str, val))}
                if stage != "fusion":
                    model, info = csmm_sann(stage, tr, va, tw, seed, dev)
                    store.save(rid, {**rec, **info}, model=model)
                else:
                    parents = []
                    for st, mod in (("eegsann", sann_eeg), ("eyesann", sann_eye)):
                        m = mod.make_model()
                        m.load_state_dict(torch.load(store.checkpoint(f"{base}_{st}"), weights_only=False)["state_dict"])
                        parents.append(m)
                    model, info, probs = csmm_fusion(tr, va, tw, *parents, seed, dev)
                    store.save(rid, {**rec, **info}, tgt[t], probs, model)
                print(f"{rid}: source val {info['source_val_acc']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcca", action="store_true")
    ap.add_argument("--csmm", action="store_true")
    a = ap.parse_args()
    enable_determinism()
    dev = device()
    src = {c: load_trials(c) for c in config.CULTURES}
    tgt = {c: unlabelled_trials(c) for c in config.CULTURES}
    if a.dcca or not a.csmm:
        dcca(src, tgt, dev)
    if a.csmm or not a.dcca:
        csmm(src, tgt, dev)


if __name__ == "__main__":
    main()
