import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccsi import config
from ccsi.data import load_trials, unlabelled_trials
from ccsi.runs import GRID, RunStore, picks, run_id, run_transductive
from ccsi.training import enable_determinism


def main():
    """Feature- and encoder-level fusion (62 channels) at the five configurations not selected for each target,
    five runs per direction. The runs at the selected configurations come from 06_five_runs.py."""
    enable_determinism()
    store, pk = RunStore("config_grid"), picks()
    src = {c: load_trials(c) for c in config.CULTURES}
    tgt = {c: unlabelled_trials(c) for c in config.CULTURES}
    for arm in ("feat62", "enc62"):
        for seed in config.seeds():
            for s, t in config.DIRECTIONS:
                for d, q in GRID:
                    if (d, q) != pk[t]:
                        run_transductive(store, run_id(arm, s, t, seed, d, q), arm, s, t, seed, d, q, src[s], tgt[t])


if __name__ == "__main__":
    main()
