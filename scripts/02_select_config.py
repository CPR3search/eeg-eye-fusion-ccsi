import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccsi import config
from ccsi.data import load_trials, unlabelled_trials
from ccsi.runs import GRID, RunStore, run_id, run_transductive
from ccsi.training import enable_determinism

SEED = config.SINGLE_RUN_SEED


def main():
    """Feature-level fusion at the six (width, difference order) configurations, one run per direction (seed 2).
    Per target, the configuration with the highest mean source-validation accuracy over the two incoming directions
    is selected; ties go to the smaller model."""
    enable_determinism()
    store = RunStore("selection")
    src = {c: load_trials(c) for c in config.CULTURES}
    tgt = {c: unlabelled_trials(c) for c in config.CULTURES}
    for s, t in config.DIRECTIONS:
        for d, q in GRID:
            run_transductive(store, run_id("feat62", s, t, SEED, d, q), "feat62", s, t, SEED, d, q, src[s], tgt[t])

    rec = store.records().set_index("run_id")
    picks, grid = {}, {}
    for t in config.CULTURES:
        sources = [s for s in config.CULTURES if s != t]
        points = []
        for d, q in GRID:
            acc = [float(rec.loc[run_id("feat62", s, t, SEED, d, q), "source_val_acc"]) for s in sources]
            points.append({"d_model": d, "difference_order": q, "mean_source_val_acc": float(np.mean(acc)),
                           "n_params": int(rec.loc[run_id("feat62", sources[0], t, SEED, d, q), "n_params"])})
        grid[t] = points
        best = sorted(points, key=lambda g: (-g["mean_source_val_acc"], g["n_params"]))[0]
        picks[t] = {k: best[k] for k in ("d_model", "difference_order", "mean_source_val_acc")}
    sel = {"seed": SEED, "rule": "highest mean source-validation accuracy over the two incoming directions; "
                                 "ties to fewer parameters", "picks": picks, "grid": grid}
    (store.root / "selection.json").write_text(json.dumps(sel, indent=2))
    for t, p in picks.items():
        print(f"{t}: width {p['d_model']}, difference order {p['difference_order']}")


if __name__ == "__main__":
    main()
