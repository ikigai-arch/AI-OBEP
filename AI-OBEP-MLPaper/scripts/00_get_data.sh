#!/usr/bin/env bash
# Download the ROBOD dataset (Tekler et al., 2022) from its public GitHub repository
# into AI-OBEP-MLPaper/data/raw (5 files combined_Room<k>.csv, ~100 MB; not committed).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
git clone --depth 1 https://github.com/ideas-lab-nus/robod "$TMP/robod"
mkdir -p "$HERE/data/raw"
cp "$TMP"/robod/Data/combined_Room*.csv "$HERE/data/raw/"
rm -rf "$TMP"
ls -lh "$HERE/data/raw"
