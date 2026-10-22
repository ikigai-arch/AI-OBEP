#!/usr/bin/env bash
# Lanjutkan pipeline notebook dari 06 (00-05 sudah dieksekusi, output tersimpan di notebook & outputs/).
# Jalankan dari Git Bash: bash notebooks/resume_06_07.sh
set -uo pipefail
cd "$(dirname "$0")"
# kernel "python3" memanggil `python` dari PATH; pastikan Python 3.10 (punya semua paket) didahulukan
export PATH="/c/Users/yohan/AppData/Local/Programs/Python/Python310:/c/Users/yohan/AppData/Local/Programs/Python/Python310/Scripts:$PATH"
for nb in 06_run_nsga2 07_decomposition; do
  echo "=== $(date +%H:%M:%S) START $nb"
  python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nb.ipynb" \
    || { echo "FAILED $nb"; exit 1; }
  echo "=== $(date +%H:%M:%S) DONE $nb"
done
echo ALL_DONE
