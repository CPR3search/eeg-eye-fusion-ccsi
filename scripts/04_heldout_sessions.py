import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccsi import config
from ccsi.data import holdout_split, load_trials, strip_labels, subject_code
from ccsi.features import build_input
from ccsi.models import load_model
from ccsi.runs import RunStore, picks, run_id, train_transductive
from ccsi.training import BATCH_SIZE, device, enable_determinism, predict_proba, validation_subjects

SEED = config.SINGLE_RUN_SEED
NORMALIZATIONS = ("other_sessions", "own_session")


def main():
    """Held-out sessions: the last session of every target subject is excluded from adaptation. Feature-level fusion
    with and without alignment adapts to the other sessions; the held-out session is then classified once with the
    statistics of the other sessions and once with its own statistics. The transductive model of the configuration
    selection is applied to the same held-out sessions for reference."""
    enable_determinism()
    store, sel, pk, dev = RunStore("heldout"), RunStore("selection"), picks(), device()
    src = {c: load_trials(c) for c in config.CULTURES}
    split = {c: holdout_split(c) for c in config.CULTURES}
    for s, t in config.DIRECTIONS:
        d, q = pk[t]
        codes = [subject_code(s, x["subject"]) for x in src[s]]
        val = set(validation_subjects(codes, SEED))
        n_train = sum(c not in val for c in codes)
        steps = min(max(1, n_train // BATCH_SIZE), max(1, len(src[t]) // BATCH_SIZE))
        for arm, align in (("heldout_feat62", True), ("heldout_noalign62", False)):
            rid = run_id(arm, s, t, SEED)
            expect = {"arm": arm, "source": s, "target": t, "seed": SEED, "d_model": d, "difference_order": q,
                      "modality": "fused", "montage": "62", "architecture": "feature", "align": align}
            if store.done(rid, expect, artifacts=["checkpoint"] + [f"{rid}__{n}" for n in NORMALIZATIONS]):
                continue
            model, info, meta, _ = train_transductive(arm, s, t, SEED, d, q, src[s], strip_labels(split[t]["adaptation"]),
                                                      steps_per_epoch=steps, spec=("fused", "62", "feature", align))
            for norm in NORMALIZATIONS:
                trials = strip_labels(split[t][norm])
                store.write(f"{rid}__{norm}", trials, predict_proba(model, [build_input(x, "fused", q) for x in trials], dev=dev))
            store.save(rid, {**info, "steps_per_epoch": steps}, model=model, meta=meta)
            print(f"{rid}: source val {info['source_val_acc']:.4f}", flush=True)
        ref = run_id("feat62", s, t, SEED, d, q)
        model = load_model(sel.checkpoint(ref), dev)
        trials = strip_labels(split[t]["own_session"])
        tid = run_id("transductive_feat62", s, t, SEED)
        store.write(f"{tid}__own_session", trials, predict_proba(model, [build_input(x, "fused", q) for x in trials], dev=dev))
        store.save(tid, {"arm": "transductive_feat62", "source": s, "target": t, "seed": SEED, "reused_from": ref})


if __name__ == "__main__":
    main()
