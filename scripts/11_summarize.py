import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccsi import config
from ccsi.features import input_dims, montage_size
from ccsi.models import encoder_params, feature_params, matched_branch_width
from ccsi.runs import GRID, RunStore, picks, run_id

SEEDS = config.seeds()
DIRS = [f"{s}_to_{t}" for s, t in config.DIRECTIONS]
TIE = 1e-12
METRICS = {"accuracy": "accuracy_pct", "macro_f1": "macro_f1_pct"}
FIVE_RUN_ARMS = ["feat62", "enc62", "decmean62", "decweighted62", "eeg62", "eye", "feat8", "eeg8", "decmean8",
                 "decweighted8", "noalign62", "dcca62", "csmm62", "enc8"]
NAMED = ["32", "16", "8", "frontal", "temporal", "frontal_temporal"]


def pair(a, b):
    return f"{a}-{b}", lambda x: x[a] - x[b]


def contrasts():
    F, T, LM, LW, E6, Y, F8, E8, LM8, LW8, FO, DC, CS, T8 = FIVE_RUN_ARMS
    out = [("fusion_level", *pair(a, b)) for a, b in [(F, T), (F, LM), (F, LW), (T, LM), (T, LW), (LM, LW)]]
    out += [("fusion_vs_single_sensor", *pair(a, b)) for a in (F, T, LM, LW) for b in (E6, Y)]
    out += [("montage", *pair(a, b)) for a, b in [(F8, F), (E8, E6), (F8, E8), (F8, Y)]]
    out += [("montage", f"I=({F8}-{F})-({E8}-{E6})", lambda x: ((x[F8] - x[F]) - x[E8]) + x[E6])]
    out += [("montage_supporting", *pair(F8, LM8)), ("montage_supporting", *pair(F8, LW8))]
    out += [("alignment", *pair(F, FO))]
    out += [("published_comparator", *pair(a, b)) for c in (DC, CS) for a, b in [(F, c), (T, c), (LM, c), (c, E6)]]
    out += [("encoder_8", *pair(T8, b)) for b in (F8, T, LM8, LW8, E8, Y)]
    out += [("encoder_8_interaction", f"J=({T8}-{T})-({F8}-{F})", lambda x: (x[T8] - x[T]) - (x[F8] - x[F]))]
    return out


def counts(v) -> tuple[int, int, int]:
    v = np.asarray(v)
    return int((v > TIE).sum()), int((np.abs(v) <= TIE).sum()), int((v < -TIE).sum())


def consistent(d: np.ndarray) -> str:
    if d.mean() > TIE and (d > TIE).sum() >= 4:
        return "A"
    if -d.mean() > TIE and (-d > TIE).sum() >= 4:
        return "B"
    return "none"


def contrast_row(cell: pd.Series, extra: dict, metric: str) -> dict:
    cell = cell.reindex(pd.MultiIndex.from_product([SEEDS, DIRS], names=["seed", "direction"]))
    assert cell.notna().all()
    per_seed = cell.groupby("seed").mean().reindex(SEEDS).to_numpy()
    pos, tie, neg = counts(per_seed)
    cpos, ctie, cneg = counts(cell.to_numpy())
    row = {**extra, "metric": metric, "unit": "percentage_points", "mean_D": per_seed.mean(), "sd_D": per_seed.std(ddof=1)}
    row.update({f"D_seed{s}": v for s, v in zip(SEEDS, per_seed)})
    row.update(seed_positive=pos, seed_tied=tie, seed_negative=neg, cell_positive=cpos, cell_tied=ctie, cell_negative=cneg,
               consistent=consistent(per_seed) if metric == "accuracy" else "")
    row.update({f"mean_D_{d}": cell.xs(d, level="direction").mean() for d in DIRS})
    return row


def with_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["direction"] = df.source + "_to_" + df.target
    df["seed"] = df.seed.astype(int)
    return df


# ------------------------------------------------------------------ single-run montages (seed 2)
def six_direction(rows: pd.DataFrame, pid) -> tuple[float, list[float]]:
    """Mean fraction correct over the six directions, and the fractions (the arithmetic of the original summaries)."""
    frac = [int(rows.loc[pid(s, t), "target_correct"]) / int(rows.loc[pid(s, t), "target_n"]) for s, t in config.DIRECTIONS]
    return float(np.mean(frac)), frac


def subject_sd(per_subj: pd.DataFrame, ids: list[str]) -> float:
    pooled = defaultdict(lambda: [0, 0])
    for r in per_subj[per_subj.run_id.isin(ids)].itertuples():
        pooled[(r.culture, r.subject)][0] += r.correct
        pooled[(r.culture, r.subject)][1] += r.n
    return st.stdev([100 * c / n for c, n in pooled.values()])


def montages(scores, per_subj, pk, out):
    sr = scores[scores.store.isin(["selection", "montages"])].set_index("prediction_id")
    seed = config.SINGLE_RUN_SEED
    pid = {"62": lambda s, t: run_id("feat62", s, t, seed, *pk[t])}
    for name in NAMED + [f"rand{n}_{i}" for n in (8, 20) for i in range(1, 16)]:
        pid[name] = lambda s, t, name=name: run_id(name, s, t, seed)
    mean = {}
    rows = []
    for name in ["62"] + NAMED:
        mean[name], frac = six_direction(sr, pid[name])
        ids = [pid[name](s, t) for s, t in config.DIRECTIONS]
        rows.append({"montage": name, "n_channels": 62 if name == "62" else montage_size(name),
                     "accuracy_pct": 100 * mean[name], "subject_sd_pct": subject_sd(per_subj, ids),
                     "difference_vs_62_pct": 100 * (mean[name] - mean["62"]) if name != "62" else "",
                     **{f"accuracy_{d}": 100 * f for d, f in zip(DIRS, frac)}})
    pd.DataFrame(rows).to_csv(out / "montages.csv", index=False)
    rows = []
    for n, named in ((8, "8"), (20, "frontal_temporal")):
        rand = [six_direction(sr, pid[f"rand{n}_{i}"])[0] for i in range(1, 16)]
        pool = sorted([("named", 100 * mean[named])] + [(f"rand{n}_{i}", 100 * v) for i, v in enumerate(rand, 1)],
                      key=lambda x: -x[1])
        rows.append({"size": n, "named_montage": named, "named_accuracy_pct": 100 * mean[named],
                     "random_mean_pct": 100 * float(np.mean(rand)),
                     "difference_pct": 100 * (mean[named] - np.mean(rand)),
                     "rank_of_named": [k for k, _ in pool].index("named") + 1, "n_ranked": len(pool),
                     "random_min_pct": 100 * min(rand), "random_max_pct": 100 * max(rand),
                     "random_accuracies_pct": ";".join(f"{100 * v:.4f}" for v in rand)})
    pd.DataFrame(rows).to_csv(out / "random_montages.csv", index=False)


# ------------------------------------------------------------------ held-out sessions
def heldout(scores, out):
    h = with_direction(scores[scores.store == "heldout"])
    h = h[["arm", "normalization", "source", "target", "direction", "target_correct", "target_n", "accuracy_pct", "macro_f1_pct"]]
    h.to_csv(out / "heldout.csv", index=False)
    h.groupby(["arm", "normalization"], sort=False)[["accuracy_pct", "macro_f1_pct"]].mean().reset_index().to_csv(
        out / "heldout_summary.csv", index=False)


# ------------------------------------------------------------------ five-run comparison
def five_run_table(scores) -> pd.DataFrame:
    a = with_direction(scores[scores.store.isin(["five_runs", "dcca", "csmm"]) & scores.arm.isin(FIVE_RUN_ARMS)])
    a = a.set_index(["arm", "seed", "direction"]).sort_index()
    assert len(a) == len(FIVE_RUN_ARMS) * 30 and a.index.is_unique, len(a)
    return a


def five_run(scores, conf, out):
    a = five_run_table(scores)
    cells = []
    for arm in FIVE_RUN_ARMS:
        for seed in SEEDS:
            q = [a.loc[(arm, seed, d)] for d in DIRS]
            cells.append({"arm": arm, "seed": seed, "scope": "seed_mean", "direction": "",
                          "accuracy_pct": pd.Series([r.accuracy_pct for r in q]).mean(),
                          "macro_f1_pct": pd.Series([r.macro_f1_pct for r in q]).mean(), "target_correct": "", "target_n": ""})
            cells += [{"arm": arm, "seed": seed, "scope": "direction", "direction": d, "accuracy_pct": r.accuracy_pct,
                       "macro_f1_pct": r.macro_f1_pct, "target_correct": r.target_correct, "target_n": r.target_n}
                      for d, r in zip(DIRS, q)]
    cells = pd.DataFrame(cells)
    cells.to_csv(out / "five_run_cells.csv", index=False)
    summ = [{"arm": arm, "scope": sc, "direction": d, "n_runs": len(q), "accuracy_mean_pct": q.accuracy_pct.mean(),
             "accuracy_sd_pct": q.accuracy_pct.std(ddof=1), "macro_f1_mean_pct": q.macro_f1_pct.mean(),
             "macro_f1_sd_pct": q.macro_f1_pct.std(ddof=1)}
            for (arm, sc, d), q in cells.groupby(["arm", "scope", "direction"], sort=False)]
    pd.DataFrame(summ).to_csv(out / "five_run_summary.csv", index=False)
    rows = []
    for family, name, f in contrasts():
        for metric, col in METRICS.items():
            x = {arm: a[col].xs(arm, level="arm") for arm in FIVE_RUN_ARMS}
            rows.append(contrast_row(f(x), {"family": family, "contrast": name}, metric))
    pd.DataFrame(rows).to_csv(out / "five_run_contrasts.csv", index=False)
    ids = set(a.prediction_id)
    conf[conf.run_id.isin(ids)].to_csv(out / "five_run_confusion.csv", index=False)


def faults(out):
    d = pd.read_csv(config.RESULTS / "scores" / "fault_draws.csv", dtype={"level": str})
    d["direction"] = d.source + "_to_" + d.target
    rows = []
    for (arm, fault, modality, level), q in d.groupby(["arm", "fault", "modality", "level"], sort=False):
        by_dir = q.groupby(["seed", "direction"])[["accuracy_pct", "change_pp", "clean_accuracy_pct"]].mean()
        assert len(by_dir) == 30
        by_seed = by_dir.groupby("seed").mean()
        draws = q.groupby("draw").change_pp.mean()
        rows.append({"arm": arm, "montage": q.montage.iloc[0], "fault": fault, "modality": modality, "level": level,
                     "n_draws_per_direction": q.draw.nunique(), "fault_accuracy_mean_pct": by_seed.accuracy_pct.mean(),
                     "fault_accuracy_sd_pct": by_seed.accuracy_pct.std(ddof=1),
                     "clean_accuracy_mean_pct": by_seed.clean_accuracy_pct.mean(),
                     "change_mean_pp": by_seed.change_pp.mean(), "change_sd_pp": by_seed.change_pp.std(ddof=1),
                     "draw_mean_change_min_pp": draws.min(), "draw_mean_change_max_pp": draws.max(),
                     "single_row_change_min_pp": q.change_pp.min(), "single_row_change_max_pp": q.change_pp.max()})
    pd.DataFrame(rows).to_csv(out / "faults.csv", index=False)
    d[d.fault == "whole_modality_loss"].drop(columns="direction").to_csv(out / "fault_sensor_masking.csv", index=False)


# ------------------------------------------------------------------ configuration grid (62 channels, five runs)
def grid(scores, pk, out):
    rec = {n: RunStore(n).records().set_index("run_id") for n in ("five_runs", "config_grid")}
    sc = scores[scores.store.isin(["five_runs", "config_grid"])].set_index("prediction_id")
    rows = []
    for arch in ("feat62", "enc62"):
        for seed in SEEDS:
            for s, t in config.DIRECTIONS:
                cells = [pk[t]] + [c for c in GRID if c != pk[t]]
                for d, q in cells:
                    store = "five_runs" if (d, q) == pk[t] else "config_grid"
                    rid = run_id(arch, s, t, seed) if store == "five_runs" else run_id(arch, s, t, seed, d, q)
                    assert (int(rec[store].loc[rid, "d_model"]), int(rec[store].loc[rid, "difference_order"])) == (d, q), rid
                    in_dim, eeg_dim = input_dims("fused", "62", q)
                    n = (feature_params(in_dim, d) if arch == "feat62"
                         else encoder_params(in_dim, matched_branch_width(in_dim, eeg_dim, d), eeg_dim))
                    rows.append({"arch": arch, "run_id": rid, "store": store, "seed": seed, "source": s, "target": t,
                                 "direction": f"{s}_to_{t}", "d": d, "q": q, "n_params": n,
                                 "source_val_acc": float(rec[store].loc[rid, "source_val_acc"]),
                                 **{k: sc.loc[rid, k] for k in ("accuracy_pct", "macro_f1_pct", "target_correct", "target_n")}})
    g = pd.DataFrame(rows)
    g.assign(source_val_acc_pct=100 * g.source_val_acc, target_acc_pct=g.accuracy_pct)[
        ["run_id", "arch", "seed", "source", "target", "d", "q", "source_val_acc_pct", "target_acc_pct"]].to_csv(
        out / "grid_source_validation.csv", index=False)

    ens = pd.read_csv(config.RESULTS / "scores" / "grid_ensembles.csv", float_precision="round_trip").set_index(
        ["arch", "seed", "source", "target"])
    out_rows = []
    for arch in ("feat62", "enc62"):
        for seed in SEEDS:
            ts = g[(g.arch == arch) & (g.seed == seed)]
            repick = {}
            for t in config.CULTURES:
                v = ts[ts.target == t].groupby(["d", "q"]).agg(v=("source_val_acc", "mean"), n=("n_params", "first")).reset_index()
                v["order"] = [GRID.index((d, q)) for d, q in zip(v.d, v.q)]
                c = v[(v.v.max() - v.v) <= TIE].sort_values(["n", "order"])
                repick[t] = (int(c.d.iloc[0]), int(c.q.iloc[0]))
            for s, t in config.DIRECTIONS:
                q = ts[ts.direction == f"{s}_to_{t}"]
                at = q.set_index(["d", "q"])

                def emit(endpoint, m, d="", qq=""):
                    out_rows.append({"arch": arch, "seed": seed, "scope": "direction", "direction": f"{s}_to_{t}",
                                     "endpoint": endpoint, "d": d, "q": qq, "accuracy_pct": m["accuracy_pct"],
                                     "macro_f1_pct": m["macro_f1_pct"]})
                emit("selected", at.loc[pk[t]])
                for c in GRID:
                    emit("cell", at.loc[c], *c)
                emit("uniform_average", {"accuracy_pct": q.accuracy_pct.mean(), "macro_f1_pct": q.macro_f1_pct.mean()})
                emit("ensemble", ens.loc[(arch, seed, s, t)])
                emit("source_reselection", at.loc[repick[t]], *repick[t])
    e = pd.DataFrame(out_rows)
    means = [{"arch": arch, "seed": seed, "scope": "seed_mean", "direction": "", "endpoint": ep,
              "d": d if ep == "cell" else "", "q": q if ep == "cell" else "",
              "accuracy_pct": x.accuracy_pct.mean(), "macro_f1_pct": x.macro_f1_pct.mean()}
             for (arch, seed, ep, d, q), x in e.assign(d=e.d.where(e.endpoint == "cell", ""), q=e.q.where(e.endpoint == "cell", ""))
             .groupby(["arch", "seed", "endpoint", "d", "q"], sort=False)]
    pd.concat([pd.DataFrame(means), e]).to_csv(out / "grid_cells.csv", index=False)
    cons = []
    dd = e.assign(d=e.d.where(e.endpoint == "cell", ""), q=e.q.where(e.endpoint == "cell", ""))
    for ep in ["selected", "cell", "uniform_average", "ensemble", "source_reselection"]:
        for c in (GRID if ep == "cell" else [("", "")]):
            sub = dd[(dd.endpoint == ep) & (dd.d.astype(str) == str(c[0])) & (dd.q.astype(str) == str(c[1]))]
            for metric, col in METRICS.items():
                x = sub.set_index(["arch", "seed", "direction"])[col]
                cons.append(contrast_row(x.xs("enc62", level="arch") - x.xs("feat62", level="arch"),
                                         {"contrast": "enc62-feat62", "endpoint": ep, "d": c[0], "q": c[1]}, metric))
    pd.DataFrame(cons).to_csv(out / "grid_contrasts.csv", index=False)


def encoder_widths(pk, out):
    """Branch width of encoder-level fusion matched to the parameter count of feature-level fusion."""
    rows = []
    cells = [("62", t, d, q) for t in config.CULTURES for d, q in GRID] + [("8", t, *pk[t]) for t in config.CULTURES]
    for mont, t, d, q in cells:
        in_dim, eeg_dim = input_dims("fused", mont, q)
        w = matched_branch_width(in_dim, eeg_dim, d)
        nf, ne = feature_params(in_dim, d), encoder_params(in_dim, w, eeg_dim)
        rows.append({"montage": mont, "target": t, "d": d, "q": q, "selected": (d, q) == pk[t], "input_dim": in_dim,
                     "eeg_dim": eeg_dim, "eye_dim": in_dim - eeg_dim, "feature_params": nf, "branch_width": w,
                     "encoder_params": ne, "rel_gap": f"{abs(ne - nf) / nf:.6f}"})
    pd.DataFrame(rows).to_csv(out / "encoder_widths.csv", index=False)


def main():
    """Tables and values of the paper, from results/scores/."""
    out = config.results_dir("summary")
    s = config.RESULTS / "scores"
    scores = pd.read_csv(s / "scores.csv", dtype={"montage": str}, keep_default_na=False, na_values=[""])
    per_subj, conf = pd.read_csv(s / "per_subject.csv"), pd.read_csv(s / "confusion.csv")
    pk = picks()
    montages(scores, per_subj, pk, out)
    heldout(pd.read_csv(s / "scores.csv", dtype={"montage": str}, keep_default_na=False, na_values=[""],
                        float_precision="round_trip"), out)
    five_run(scores, conf, out)
    faults(out)
    grid(scores, pk, out)
    encoder_widths(pk, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
