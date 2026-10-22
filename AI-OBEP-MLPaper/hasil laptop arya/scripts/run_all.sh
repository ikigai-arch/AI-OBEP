set -euo pipefail
cd "$(dirname "$0")/.."
[ -f data/raw/combined_Room3.csv ] || bash scripts/00_get_data.sh
python scripts/01_surrogate_baseline.py        # ablation + default zoo + paper protocol
python scripts/02_tune_surrogate.py            # PSO / GA / HHO tune XGBoost (3 seeds x 200 evals)
python scripts/02b_tune_paper_protocol.py     # same three metaheuristics under the exact Lyu et al. protocol
python scripts/03_final_surrogate.py           # select + save deployed surrogate
python scripts/04_feature_importance.py        # SHAP + permutation importance
python scripts/05_tune_nsga2.py                # PSO / GA / HHO tune NSGA-II
python scripts/06_run_nsga2.py                 # 30 seeds x 25k evals, reference front, figures
python scripts/07_decomposition.py
python scripts/08_sensitivity.py