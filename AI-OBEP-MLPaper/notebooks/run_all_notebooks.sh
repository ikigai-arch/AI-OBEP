#!/usr/bin/env bash
# Eksekusi semua notebook berurutan (in place). Jalankan dari Git Bash / terminal Linux / macOS: bash notebooks/run_all_notebooks.sh [mulai_dari]
# contoh: bash notebooks/run_all_notebooks.sh 05_tune_nsga2  -> lanjut dari notebook 05
set -uo pipefail
cd "$(dirname "$0")"
# Pakai Python dari virtual environment yang sedang aktif (lihat docs/INSTALL.md).
# Kalau Python lain yang mau dipakai: PYTHON=/path/ke/python bash notebooks/run_all_notebooks.sh
PY="${PYTHON:-python}"
START="${1:-00_get_data}"; go=0
for nb in 00_get_data 01_surrogate_baseline 02_tune_surrogate 02b_tune_paper_protocol 03_final_surrogate \
          04_feature_importance 05_tune_nsga2 06_run_nsga2 07_decomposition; do
  [ "$nb" = "$START" ] && go=1
  [ $go -eq 1 ] || continue
  echo "=== $(date +%H:%M:%S) START $nb"
  "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nb.ipynb" \
    || { echo "FAILED $nb"; exit 1; }
  echo "=== $(date +%H:%M:%S) DONE $nb"
done
echo ALL_DONE
