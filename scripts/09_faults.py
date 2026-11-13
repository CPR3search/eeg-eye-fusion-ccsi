import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccsi import config
from ccsi.data import unlabelled_trials
from ccsi.faults import fault_specs
from ccsi.features import build_input
from ccsi.models import load_model
from ccsi.runs import PROBS, RunStore, file_sha256, run_id
from ccsi.training import device, predict_proba

ARMS = ["feat62", "feat8", "enc62", "enc8"]


def main():
    """Test-time sensor faults on the target trials of the five-run feature- and encoder-level models. Every
    checkpoint gets the same fault draws; the predicted classes per fault condition go to results/faults/."""
    torch.set_num_threads(3)
    dev, store, out = device(), RunStore("five_runs"), config.results_dir("faults")
    rec = store.records().set_index("run_id")
    trials = {c: unlabelled_trials(c) for c in config.CULTURES}
    for arm in ARMS:
        for seed in config.seeds():
            for s, t in config.DIRECTIONS:
                rid = run_id(arm, s, t, seed)
                path = out / f"{rid}.npz"
                sha = file_sha256(store.checkpoint(rid))
                if path.exists():
                    z = np.load(path)
                    if "checkpoint_sha256" in z.files and str(z["checkpoint_sha256"]) == sha:
                        continue
                r = rec.loc[rid]
                montage, order = str(r.montage), int(r.difference_order)
                model = load_model(store.checkpoint(rid), dev)

                def run(**tf):
                    return predict_proba(model, [build_input(x, "fused", order, montage, **tf) for x in trials[t]], dev=dev)

                saved = store.predictions(rid)[PROBS].to_numpy()
                intact = run()
                assert np.abs(intact - saved).max() < 1e-6 and (intact.argmax(1) == saved.argmax(1)).all(), \
                    f"{rid}: the intact path does not reproduce the saved predictions"
                specs = fault_specs(montage)
                pred = np.stack([run(**f["make"]()).argmax(1) for f in specs]).astype(np.int8)
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, pred=pred, checkpoint_sha256=np.array(sha))
                tmp.replace(path)
                print(f"{rid}: {len(specs)} fault conditions", flush=True)


if __name__ == "__main__":
    main()
