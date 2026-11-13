#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf third_party/CSMM
git clone https://github.com/xbrainnet/CSMM.git third_party/CSMM
git -C third_party/CSMM checkout ea99eed9c09709d6b4e4b6afcfd0e56d8316b7b5
rm -rf third_party/CSMM/.git
git apply third_party/csmm.patch
