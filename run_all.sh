#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f third_party/CSMM/main_modalFusion.py ]; then
    echo "CSMM is missing: run 'bash third_party/fetch_csmm.sh' first" >&2
    exit 1
fi
python scripts/01_prepare_data.py
python scripts/02_select_config.py
python scripts/03_montages.py
python scripts/04_heldout_sessions.py
python scripts/05_profile.py
python scripts/06_five_runs.py
python scripts/07_comparators.py
python scripts/08_config_grid.py
python scripts/09_faults.py
python scripts/10_score.py
python scripts/11_summarize.py
