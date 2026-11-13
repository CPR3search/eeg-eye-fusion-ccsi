import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccsi import config
from ccsi.data import load_trials, unlabelled_trials
from ccsi.runs import ARMS, KEYS, PROBS, RunStore, picks, run_id, run_transductive
from ccsi.training import enable_determinism


def decision_fusion(store: RunStore):
    """Decision-level fusion of the EEG-only and eye-only models of the same run and direction: the mean of their
    trial probabilities, and the mean weighted by their source-validation accuracies."""
    rec = store.records().set_index("run_id")
    for seed in config.seeds():
        for s, t in config.DIRECTIONS:
            for montage in ("62", "8"):
                e_id, y_id = run_id(f"eeg{montage}", s, t, seed), run_id("eye", s, t, seed)
                ae, ay = float(rec.loc[e_id, "source_val_acc"]), float(rec.loc[y_id, "source_val_acc"])
                e, y = store.predictions(e_id), store.predictions(y_id)
                y = e[KEYS].merge(y, on=KEYS, how="left", validate="one_to_one")
                assert len(y) == len(e) and y.p_negative.notna().all()
                for kind, (we, wy) in (("mean", (0.5, 0.5)),
                                       ("weighted", (0.5, 0.5) if ae + ay == 0 else (ae / (ae + ay), ay / (ae + ay)))):
                    arm = f"dec{kind}{montage}"
                    p = we * e[PROBS].values + wy * y[PROBS].values
                    df = e[KEYS + ["n_windows"]].copy()
                    df["pred"] = p.argmax(1)
                    df[PROBS] = p
                    rid = run_id(arm, s, t, seed)
                    df[KEYS + ["pred"] + PROBS + ["n_windows"]].to_csv(store.root / "predictions" / f"{rid}.csv", index=False)
                    store.save(rid, {"arm": arm, "source": s, "target": t, "seed": seed, "montage": montage,
                                     "eeg_run_id": e_id, "eye_run_id": y_id, "w_eeg": we, "w_eye": wy})


def main():
    """The eight trained models of the five-run comparison (5 runs x 6 directions each), at the selected
    configurations, with the validation subjects drawn from each run's seed; then decision-level fusion."""
    enable_determinism()
    store, pk = RunStore("five_runs"), picks()
    src = {c: load_trials(c) for c in config.CULTURES}
    tgt = {c: unlabelled_trials(c) for c in config.CULTURES}
    for seed in config.seeds():
        for s, t in config.DIRECTIONS:
            d, q = pk[t]
            for arm in ARMS:
                run_transductive(store, run_id(arm, s, t, seed), arm, s, t, seed, d, q, src[s], tgt[t])
    decision_fusion(store)


if __name__ == "__main__":
    main()
