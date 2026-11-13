import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccsi import config
from ccsi.faults import fault_specs
from ccsi.runs import GRID, KEYS, PROBS, RunStore, file_sha256, picks, run_id
from ccsi.scoring import attach_labels, confusion, metrics, per_subject

STORES = ["selection", "montages", "heldout", "five_runs", "dcca", "csmm", "config_grid"]


def score_store(name: str):
    store = RunStore(name)
    rec = store.records().set_index("run_id")
    rows, subj, conf = [], [], []
    for p in sorted((store.root / "predictions").glob("*.csv")):
        pid = p.stem
        rid, _, norm = pid.partition("__")
        df = attach_labels(pd.read_csv(p))
        base = rec.loc[rid].to_dict() if rid in rec.index else {}
        m = metrics(df)
        if name in ("selection", "montages"):  # the arithmetic of the original single-run summaries
            m["accuracy_pct"] = 100 * (m["target_correct"] / m["target_n"])
        rows.append({"store": name, "prediction_id": pid, "run_id": rid, "normalization": norm, **base, **m})
        subj += per_subject(pid, df)
        conf += confusion(pid, df)
    return rows, subj, conf


def csv_round_trip(df: pd.DataFrame) -> pd.DataFrame:
    """Write and re-read a table with pandas defaults, as the original pipeline did between scoring and summaries."""
    return pd.read_csv(io.StringIO(df.to_csv(index=False)))


def score_faults(scores: pd.DataFrame) -> pd.DataFrame:
    five = RunStore("five_runs")
    rec = five.records().set_index("run_id")
    s5 = scores[scores.store == "five_runs"]
    clean = s5.set_index("prediction_id").accuracy_pct
    parsed = csv_round_trip(s5[["prediction_id", "accuracy_pct"]]).set_index("prediction_id").accuracy_pct
    rows = []
    for path in sorted(p for p in (config.RESULTS / "faults").glob("*.npz") if not p.name.endswith(".tmp.npz")):
        rid = path.stem
        r = rec.loc[rid]
        # the original summaries took the clean accuracy of the 62-channel encoder-level runs from a CSV
        base = parsed[rid] if r.arm == "enc62" else clean[rid]
        y = attach_labels(five.predictions(rid)).label.to_numpy()
        z = np.load(path)
        assert "checkpoint_sha256" in z.files and str(z["checkpoint_sha256"]) == file_sha256(five.checkpoint(rid)), \
            f"{path} is from another checkpoint; rerun step 9"
        pred = z["pred"]
        specs = fault_specs(str(r.montage))
        assert pred.shape == (len(specs), len(y))
        for i, (f, c) in enumerate(zip(specs, (pred == y).sum(1))):
            acc = 100 * int(c) / len(y)
            rows.append({"run_id": rid, "arm": r.arm, "seed": r.seed, "source": r.source, "target": r.target,
                         "montage": str(r.montage), "plan_index": i, **{k: v for k, v in f.items() if k != "make"},
                         "target_correct": int(c), "target_n": len(y), "accuracy_pct": acc,
                         "clean_accuracy_pct": base, "change_pp": acc - base})
    return pd.DataFrame(rows)


def score_grid_ensembles() -> pd.DataFrame:
    """Probability ensemble over the six configurations of the grid, per architecture, run and direction (the
    selected configuration first, then the others in grid order)."""
    pk, rows = picks(), []
    stores = {n: RunStore(n) for n in ("five_runs", "config_grid")}
    for arch in ("feat62", "enc62"):
        for seed in config.seeds():
            for s, t in config.DIRECTIONS:
                frames = []
                for d, q in [pk[t]] + [c for c in GRID if c != pk[t]]:
                    name, rid = (("five_runs", run_id(arch, s, t, seed)) if (d, q) == pk[t]
                                 else ("config_grid", run_id(arch, s, t, seed, d, q)))
                    frames.append(attach_labels(csv_round_trip(stores[name].predictions(rid))).set_index(KEYS).sort_index())
                assert all(f.index.equals(frames[0].index) for f in frames)
                df = frames[0][["label"]].copy()
                df["pred"] = np.mean([f[PROBS].to_numpy() for f in frames], axis=0).argmax(1)
                rows.append({"arch": arch, "seed": seed, "source": s, "target": t, **metrics(df.reset_index())})
    return pd.DataFrame(rows)


def main():
    """Attaches the target labels to the saved predictions and computes accuracy, macro-F1, per-subject accuracy and
    confusion counts. This is the only step that reads target labels for evaluation."""
    out = config.results_dir("scores")
    rows, subj, conf = [], [], []
    for name in STORES:
        if (config.RESULTS / name / "predictions").exists():
            r, s, c = score_store(name)
            rows += r
            subj += s
            conf += c
            print(f"{name}: {len(r)} prediction files scored")
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "scores.csv", index=False)
    pd.DataFrame(subj).to_csv(out / "per_subject.csv", index=False)
    pd.DataFrame(conf).to_csv(out / "confusion.csv", index=False)
    if (config.RESULTS / "faults").exists():
        f = score_faults(scores)
        f.to_csv(out / "fault_draws.csv", index=False)
        print(f"faults: {len(f)} rows")
    if (config.RESULTS / "config_grid" / "predictions").exists():
        score_grid_ensembles().to_csv(out / "grid_ensembles.csv", index=False)


if __name__ == "__main__":
    main()
