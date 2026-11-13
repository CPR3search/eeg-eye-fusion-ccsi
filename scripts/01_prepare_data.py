import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccsi import config
from ccsi.data import check_label_runs, label_runs, load_de, load_trials, sessions, subject_code, timing_epochs, trial_info
from ccsi.features import read_rows, channel_names
from ccsi.training import validation_subjects


def extract_french():
    out = config.DE_FEATURE_DIRS["french"] / "eeg_used_4s"
    if len(list(out.glob("*.npz"))) == 24:
        return
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(config.FRENCH_ZIP) as z:
        members = [m for m in z.namelist() if m.startswith("02-EEG-DE-features/eeg_used_4s/") and m.endswith(".npz")]
        assert len(members) == 24, members
        for m in members:
            (out / Path(m).name).write_bytes(z.read(m))
    print(f"extracted {len(members)} French EEG feature files to {out}")


def check_segmentation(out: Path):
    label_rows, qc, segments = [], [], []
    for c in config.CULTURES:
        for subject, session in sessions(c):
            lab = load_de(c, subject, session)["label"]
            runs = label_runs(lab)
            check = check_label_runs(c, runs)
            label_rows.append({"culture": c, "subject": subject, "session": session, "n_label_runs": len(runs),
                               "all_three_labels": {r["label"] for r in runs} == {0, 1, 2}, **check})
            n = len(lab)
            row = {"culture": c, "subject": subject, "session": session, "n_windows": n}
            for name, scaled in (("scaled", True), ("unscaled", False)):
                ep = timing_epochs(c, n, scale_to_windows=scaled)
                wrong = sum(int((lab[e["start_idx"]:e["end_idx"]] != e["label"]).sum()) for e in ep)
                row[f"{name}_wrong_windows"], row[f"{name}_wrong_pct"] = wrong, 100 * wrong / n
                if scaled:
                    row["n_trials"] = len(ep)
                    segments += [{"culture": c, "subject": subject, "session": session, "trial_index": k,
                                  "clip_number": e["clip_number"], "start_idx": e["start_idx"], "end_idx": e["end_idx"],
                                  "n_windows": e["end_idx"] - e["start_idx"]} for k, e in enumerate(ep)]
            qc.append(row)
    label_rows = pd.DataFrame(label_rows)
    label_rows.to_csv(out / "label_run_check.csv", index=False)
    if not (label_rows.passed & label_rows.all_three_labels).all():
        raise SystemExit("label-run check failed; see results/data/label_run_check.csv")
    qc = pd.DataFrame(qc)[["culture", "subject", "session", "n_windows", "scaled_wrong_windows", "scaled_wrong_pct",
                           "n_trials", "unscaled_wrong_windows", "unscaled_wrong_pct"]]
    qc.to_csv(out / "segmentation_qc.csv", index=False)
    pd.DataFrame(segments).to_csv(out / "trial_segments.csv", index=False)
    print(qc.groupby("culture")[["n_trials", "scaled_wrong_pct", "unscaled_wrong_pct"]].median().round(2).to_string())


def dataset_facts(out: Path):
    rows = []
    for c in config.CULTURES:
        ss = sessions(c)
        trials = load_trials(c)
        lens, labels, info = [t["n_windows"] for t in trials], [t["label"] for t in trials], trial_info(c)
        rows.append({"culture": c, "n_subjects": len({s for s, _ in ss}), "n_sessions": len(ss),
                     "clips_per_session": len(info), "n_trials": len(trials),
                     "trial_length_windows_min": min(lens), "trial_length_windows_max": max(lens),
                     "trial_length_windows_mean": round(float(np.mean(lens)), 2),
                     "trial_length_windows_median": float(np.median(lens)),
                     "clip_duration_s_min": info.duration_seconds.min(), "clip_duration_s_max": info.duration_seconds.max(),
                     "n_negative": labels.count(0), "n_neutral": labels.count(1), "n_positive": labels.count(2)})
    pd.DataFrame(rows).to_csv(out / "dataset_facts.csv", index=False)


def validation_splits(out: Path):
    rows = []
    for seed in [config.SINGLE_RUN_SEED] + config.seeds():
        for s, t in config.DIRECTIONS:
            codes = sorted({subject_code(s, subj) for subj, _ in sessions(s)})
            val = validation_subjects(codes, seed)
            rows.append({"seed": seed, "source": s, "target": t, "val_subjects": ";".join(map(str, val)),
                         "train_subjects": ";".join(str(x) for x in codes if x not in val)})
    pd.DataFrame(rows).to_csv(out / "validation_splits.csv", index=False)


def check_random_montages():
    """The 30 random montages are the first draws of SeedSequence(12345).spawn(3)[0]: 8 and 20 channels, 15 each."""
    rng, names = np.random.default_rng(np.random.SeedSequence(12345).spawn(3)[0]), channel_names()
    order = [(n, i) for n in (8, 20) for i in range(1, 6)] + [(n, i) for n in (8, 20) for i in range(6, 16)]
    for n, i in order:
        drawn = {names[j] for j in rng.choice(62, size=n, replace=False)}
        shipped = {r["channel"] for r in read_rows(config.CONFIGS / "channels" / f"rand{n}_{i}.txt")}
        assert drawn == shipped, f"rand{n}_{i}: configs/channels differs from the seeded draw"


def main():
    out = config.results_dir("data")
    extract_french()
    check_segmentation(out)
    dataset_facts(out)
    validation_splits(out)
    check_random_montages()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
