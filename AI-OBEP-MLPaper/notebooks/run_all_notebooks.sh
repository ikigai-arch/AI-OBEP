#!/usr/bin/env bash
# Eksekusi semua notebook berurutan (in place). Jalankan dari Git Bash: bash notebooks/run_all_notebooks.sh [mulai_dari]
# contoh: bash notebooks/run_all_notebooks.sh 05_tune_nsga2  -> lanjut dari notebook 05
set -uo pipefail
cd "$(dirname "$0")"
# kernel "python3" memanggil `python` dari PATH; pastikan Python 3.10 (punya semua paket) didahulukan
export PATH="/c/Users/yohan/AppData/Local/Programs/Python/Python310:/c/Users/yohan/AppData/Local/Programs/Python/Python310/Scripts:$PATH"
START="${1:-00_get_data}"; go=0
for nb in 00_get_data 01_surrogate_baseline 02_tune_surrogate 02b_tune_paper_protocol 03_final_surrogate \
          04_feature_importance 05_tune_nsga2 06_run_nsga2 07_decomposition; do
  [ "$nb" = "$START" ] && go=1
  [ $go -eq 1 ] || continue
  echo "=== $(date +%H:%M:%S) START $nb"
  python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nb.ipynb" \
    || { echo "FAILED $nb"; exit 1; }
  echo "=== $(date +%H:%M:%S) DONE $nb"
done
echo ALL_DONE
