# Fusion Level and Montage Size in Cross-Cultural EEG and Eye-Movement Emotion Recognition

## Contents

```
ccsi/
  config.py        dataset paths, cohorts, transfer directions, seeds
  data.py          feature loading, trial segmentation from the clip timing, per-session normalization, held-out split
  features.py      difference features, EEG montages, model inputs
  models.py        feature-level fusion, encoder-level fusion, parameter counts and matched widths
  training.py      training with focal loss, DANN and LMMD alignment; source-validation epoch selection; prediction
  runs.py          the trained arms and the run store (checkpoints, label-free predictions, records)
  faults.py        test-time sensor faults
  comparators.py   DCCA-AM and the CSMM adapter
  scoring.py       label attachment and metrics
scripts/           11 numbered steps (below)
configs/           EEG montages, eye-movement feature groups, the five seeds
third_party/       patch and fetch script for the CSMM comparator
run_all.sh         runs the 11 steps in order
```

## Data

The datasets are available from the BCMI laboratory of Shanghai Jiao Tong University
(https://bcmi.sjtu.edu.cn/home/seed/) to academic researchers who sign its license agreement. 
Place them under one root and set `SEED_DATASET_ROOT` to it (default `../SEED-Dataset/Data`).

```
$SEED_DATASET_ROOT/
  SEED/channel_62_pos.locs
  SEED/SEED_Multimodel/02-EEG-DE-feature/eeg_used_4s/
  SEED/SEED_Multimodel/04-Eye-tracking-feature/eye_tracking_feature/
  SEED/SEED_Multimodel/China_information.xlsx
  SEED-GER/German_information.xlsx
  SEED-GER/German/02-EEG-DE-features/eeg_used_4s/
  SEED-GER/German/04-Eye-tracking-feature/eye_tracking_feature/
  SEED-FRA/SEED_FRA.zip
  SEED-FRA/French/French.zip
  SEED-FRA/French/04-Eye-tracking-features/eye_tracking_feature/
```

The released differential-entropy features on 4-s windows and the 33 eye-movement features are used as provided.
Step 1 extracts the French EEG features from `French.zip`.

## Environment

The experiments were run with Python 3.12 on Linux with one NVIDIA GeForce RTX 4060 Ti (CUDA 13.0).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash third_party/fetch_csmm.sh
```

## Re-running the experiments

```bash
export SEED_DATASET_ROOT=/path/to/SEED-Dataset/Data
./run_all.sh
```

## License

The code is released under the MIT License (`LICENSE`). The CSMM code fetched by `third_party/fetch_csmm.sh`, and
the upstream lines contained in `third_party/csmm.patch`, belong to their authors and are not covered by this
license.
