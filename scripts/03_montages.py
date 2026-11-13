import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccsi import config
from ccsi.data import load_trials, unlabelled_trials
from ccsi.runs import RunStore, picks, run_id, run_transductive
from ccsi.training import enable_determinism

SEED = config.SINGLE_RUN_SEED
NAMED = ["32", "16", "8", "frontal", "temporal", "frontal_temporal"]
RANDOM = [f"rand{n}_{i}" for n in (8, 20) for i in range(1, 16)]


def main():
    """Feature-level fusion with reduced EEG montages at the selected configurations, one run per direction (seed 2):
    the nested and regional montages and 15 random montages of 8 and of 20 channels."""
    enable_determinism()
    store, pk = RunStore("montages"), picks()
    src = {c: load_trials(c) for c in config.CULTURES}
    tgt = {c: unlabelled_trials(c) for c in config.CULTURES}
    for name in NAMED + RANDOM:
        for s, t in config.DIRECTIONS:
            d, q = pk[t]
            run_transductive(store, run_id(name, s, t, SEED), name, s, t, SEED, d, q, src[s], tgt[t],
                             spec=("fused", name, "feature", True))


if __name__ == "__main__":
    main()
