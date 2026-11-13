from functools import lru_cache

import numpy as np
import pandas as pd

from ccsi import config
from ccsi.data import load_trials
from ccsi.runs import KEYS, PROBS


@lru_cache(maxsize=None)
def labels() -> pd.DataFrame:
    d = pd.DataFrame([{**{k: t[k] for k in KEYS}, "label": t["label"]}
                      for c in config.CULTURES for t in load_trials(c)])
    assert not d.duplicated(KEYS).any()
    return d


def attach_labels(pred: pd.DataFrame) -> pd.DataFrame:
    assert "label" not in pred.columns
    out = pred.merge(labels(), on=KEYS, how="left", validate="one_to_one")
    assert len(out) == len(pred) and out.label.notna().all()
    out["label"] = out.label.astype(int)
    assert (out.pred == out[PROBS].values.argmax(1)).all()
    return out


def macro_f1(df: pd.DataFrame) -> float:
    f1s = []
    for c in (0, 1, 2):
        tp = ((df.pred == c) & (df.label == c)).sum()
        fp = ((df.pred == c) & (df.label != c)).sum()
        fn = ((df.pred != c) & (df.label == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        f1s.append(np.nan if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return 100 * np.nanmean(f1s)


def metrics(df: pd.DataFrame) -> dict:
    c = int((df.label == df.pred).sum())
    return {"target_correct": c, "target_n": len(df), "accuracy_pct": 100 * c / len(df), "macro_f1_pct": float(macro_f1(df))}


def per_subject(run_id: str, df: pd.DataFrame) -> list[dict]:
    g = df.assign(correct=(df.label == df.pred).astype(int)).groupby(["culture", "subject"], sort=True).correct.agg(["size", "sum"])
    return [{"run_id": run_id, "culture": c, "subject": s, "n": int(r["size"]), "correct": int(r["sum"]),
             "accuracy": 100 * r["sum"] / r["size"]} for (c, s), r in g.iterrows()]


def confusion(run_id: str, df: pd.DataFrame) -> list[dict]:
    return [{"run_id": run_id, "true": a, "pred": b, "count": int(((df.label == a) & (df.pred == b)).sum())}
            for a in range(3) for b in range(3)]
