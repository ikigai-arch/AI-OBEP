# Panduan Instalasi AI-OBEP di Device Baru

Panduan ini membawa kamu dari laptop/PC kosong sampai notebook AI-OBEP bisa dijalankan.
Pilih **salah satu** jalur di bawah — semuanya menghasilkan hal yang sama.

| Jalur | Cocok untuk | Perlu install di laptop? |
|---|---|---|
| [A. Google Colab](#a-google-colab-paling-cepat-tanpa-install) | Cepat coba, laptop kentang, HP/tablet | Tidak |
| [B. VS Code](#b-vs-code-disarankan-untuk-kerja-harian) | Kerja harian, edit kode `src/` | Ya |
| [C. Anaconda / Jupyter Lab](#c-anaconda--jupyter-lab) | Sudah terbiasa Anaconda | Ya |
| [D. PyCharm / editor lain](#d-pycharm--editor-lain) | Sudah punya IDE favorit | Ya |

Setelah instalasi, lanjut ke [Urutan menjalankan](#urutan-menjalankan) dan
[docs/MEMAHAMI_ML.md](MEMAHAMI_ML.md) untuk memahami isi ML-nya.

---

## 0. Sekilas isi repo

Repo ini berisi **dua sub-proyek** yang berdiri sendiri:

```
AI-OBEP/
├── src/aiobep/              # v1: replikasi paper Lyu et al. (2025) — paket Python
├── notebooks/               # v1: 01 EDA → 02 surrogate → 03 optimasi (NSGA-II/III, SMS-EMOA)
├── scripts/                 # v1: versi skrip dari notebook 01
├── outputs/                 # v1: hasil (gambar, CSV)
├── SupplementaryData.zip    # data ROBOD untuk v1 (diekstrak ke data/raw/)
│
├── AI-OBEP-MLPaper/         # v2: pengembangan untuk paper (R² 0,94, PSO/GA/HHO, SHAP, setpoint ≥ 24 °C)
│   ├── src/obep_mlpaper/    #     kode inti
│   ├── notebooks/ 00 … 07   #     pipeline lengkap dalam notebook
│   ├── scripts/ 00 … 07     #     pipeline yang sama dalam skrip .py
│   └── outputs/             #     hasil + model surrogate tersimpan (surrogate.joblib)
│
├── requirements.txt         # paket v1
├── requirements-all.txt     # paket v1 + v2 + Jupyter  ← pakai ini
├── check_setup.py           # cek instalasi otomatis
├── colab_quickstart.ipynb   # setup sekali-klik untuk Google Colab
└── docs/                    # panduan ini
```

**Mana yang dipakai?** Untuk paper, fokus ke **`AI-OBEP-MLPaper/`** (v2) — hasilnya jauh lebih akurat.
v1 tetap berguna untuk memahami dataset (notebook 01) dan membandingkan algoritma optimasi.

## Tools yang digunakan

| Kategori | Tool / library | Dipakai untuk |
|---|---|---|
| Bahasa | **Python 3.10 – 3.12** (sudah dipakai juga di 3.13) | semua kode |
| Data | `numpy`, `pandas` | membaca CSV ROBOD, agregasi per jam, fitur waktu |
| Machine learning | `scikit-learn` | Ridge, RandomForest, ExtraTrees, KNN, MLP, cross-validation |
| | `xgboost`, `lightgbm` | gradient boosting — surrogate utama (XGBoost-PSO) |
| Metaheuristik | ditulis sendiri (`metaheuristics.py`) | PSO, GA, HHO untuk tuning hyperparameter |
| Optimasi multi-objektif | `pymoo` | NSGA-II (v1 juga NSGA-III, SMS-EMOA), hypervolume, IGD+ |
| Kenyamanan termal | `pythermalcomfort` (≥ 3.8) | PMV menurut ISO 7730 |
| Interpretasi model | `shap` | feature importance (SHAP + permutation) |
| Statistik | `scipy` | NNLS (bobot ensemble), uji Mann–Whitney |
| Visualisasi | `matplotlib` | semua gambar di `outputs/` |
| Lingkungan kerja | Jupyter / VS Code / Colab, `git` | menjalankan notebook, mengunduh data |

Data: **ROBOD** (Tekler et al., 2022) — 5 ruangan gedung SDE4 NUS Singapura, interval 5 menit.

---

## A. Google Colab (paling cepat, tanpa install)

1. Buka <https://colab.research.google.com> → `File → Open notebook → GitHub`.
2. Tempel `ikigai-arch/AI-OBEP`, pilih **`colab_quickstart.ipynb`**.
3. `Runtime → Run all`. Notebook itu akan:
   clone repo → `pip install -r requirements-all.txt` → `python check_setup.py --fix-data` →
   unduh data v2 → demo NSGA-II singkat dengan grafik Pareto.

Tidak perlu GPU (`Runtime → Change runtime type → CPU` sudah cukup).

**Membuka notebook lain secara interaktif di Colab:** buka notebooknya lewat `File → Open notebook → GitHub`,
lalu sisipkan sel ini **paling atas** dan jalankan duluan:

```python
import os
if not os.path.exists("/content/AI-OBEP"):
    !git clone --depth 1 https://github.com/ikigai-arch/AI-OBEP.git /content/AI-OBEP
    !pip install -q -r /content/AI-OBEP/requirements-all.txt
%cd /content/AI-OBEP/notebooks          # notebook v2: /content/AI-OBEP/AI-OBEP-MLPaper/notebooks
```

Untuk notebook v2 jalankan sekali juga `!bash /content/AI-OBEP/AI-OBEP-MLPaper/scripts/00_get_data.sh`.

> ⚠️ Semua file di Colab hilang saat runtime diputus. Simpan hasil ke Drive (sel terakhir `colab_quickstart.ipynb`)
> atau download folder `outputs/`.

---

## Persiapan untuk jalur B, C, D (sekali saja per device)

### 1. Install Python dan Git

| OS | Python | Git |
|---|---|---|
| **Windows** | <https://www.python.org/downloads/> → versi **3.12**. Saat install **centang "Add python.exe to PATH"**. | <https://git-scm.com/download/win> (sekaligus memberi **Git Bash**) |
| **macOS** | `brew install python@3.12` atau installer python.org | `xcode-select --install` |
| **Linux (Ubuntu)** | `sudo apt install python3 python3-venv python3-pip` | `sudo apt install git` |

Cek di terminal (Windows: PowerShell atau Command Prompt):

```bash
python --version      # Windows; harus 3.10 – 3.12
python3 --version     # macOS / Linux
git --version
```

### 2. Clone repo

```bash
git clone https://github.com/ikigai-arch/AI-OBEP.git
cd AI-OBEP
```

(Tanpa git: di GitHub klik **Code → Download ZIP**, lalu ekstrak. Tapi data v2 butuh git — lihat [Data](#data).)

### 3. Buat virtual environment + install paket

Virtual environment (`.venv`) = "kotak" paket khusus proyek ini, supaya tidak bentrok dengan proyek lain.
**Dibuat ulang di tiap device — jangan di-commit ke GitHub** (sudah di-ignore).

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # kalau ditolak: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-all.txt
```

**Windows (Command Prompt):** sama, tapi aktivasinya `.venv\Scripts\activate.bat`.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-all.txt
```

Tanda venv aktif: ada `(.venv)` di awal baris terminal. Setiap buka terminal baru, aktifkan lagi.

> ❗ Perintah yang **salah** (pernah terjadi di repo ini): `python -m venv venv\Scripts\activate` — itu justru
> *membuat* venv baru di dalam folder bernama `activate`. Membuat venv: `python -m venv .venv`;
> mengaktifkan: `.venv\Scripts\activate`.

### 4. Cek instalasi

```bash
python check_setup.py --fix-data
```

Semua paket harus `[ OK ]`. Opsi `--fix-data` sekalian mengekstrak `SupplementaryData.zip` ke `data/raw/`.

---

## B. VS Code (disarankan untuk kerja harian)

1. Install VS Code: <https://code.visualstudio.com>.
2. Install extension: **Python** dan **Jupyter** (keduanya dari Microsoft).
3. `File → Open Folder…` → pilih folder `AI-OBEP`.
4. Kerjakan [Persiapan](#persiapan-untuk-jalur-b-c-d-sekali-saja-per-device) langkah 3–4 di terminal VS Code (`` Ctrl+` ``).
5. `Ctrl+Shift+P` → **Python: Select Interpreter** → pilih yang ada `.venv`.
6. Buka notebook (mis. `notebooks/01_explore_dataset.ipynb`) → klik **Select Kernel** (kanan atas) →
   **Python Environments** → `.venv`.
7. **Run All**.

Folder kerja notebook tidak perlu diatur — semua notebook sudah otomatis mencari folder repo.

---

## C. Anaconda / Jupyter Lab

```bash
conda create -n aiobep python=3.12 -y
conda activate aiobep
cd path/ke/AI-OBEP
pip install -r requirements-all.txt      # pakai pip, bukan conda install, supaya versi sama
python check_setup.py --fix-data
jupyter lab
```

Di Jupyter Lab buka `notebooks/` atau `AI-OBEP-MLPaper/notebooks/`. Kalau kernel `aiobep` tidak muncul:
`python -m ipykernel install --user --name aiobep`.

---

## D. PyCharm / editor lain

- **PyCharm:** `Open` folder repo → `Settings → Project → Python Interpreter → Add Interpreter → Virtualenv`
  (atau pilih `.venv` yang sudah dibuat) → buka `requirements-all.txt`, klik **Install requirements**.
  Notebook bisa dibuka di PyCharm Professional, atau jalankan `jupyter lab` dari terminal PyCharm.
- **Spyder / editor teks biasa:** jalankan versi skrip, tanpa notebook:

```bash
python scripts/01_explore_dataset.py           # v1, EDA
cd AI-OBEP-MLPaper
python scripts/01_surrogate_baseline.py        # v2, dst. sampai 07 (lihat scripts/run_all.sh)
```

---

## Data

| Sub-proyek | Sumber | Cara menyiapkan | Lokasi |
|---|---|---|---|
| v1 (root) | `SupplementaryData.zip` di repo | otomatis oleh notebook 01, atau `python check_setup.py --fix-data` | `data/raw/` |
| v2 (`AI-OBEP-MLPaper`) | GitHub resmi ROBOD | notebook `00_get_data`, atau `bash AI-OBEP-MLPaper/scripts/00_get_data.sh` (butuh git) | `AI-OBEP-MLPaper/data/raw/` |

Kedua versi file CSV **sedikit berbeda** (versi GitHub punya satuan di nama kolom + kolom `rainfall_raw`),
jadi sengaja dipisah agar hasil masing-masing tetap bisa direproduksi persis.

Tanpa git (v2): download ZIP dari <https://github.com/ideas-lab-nus/robod> → salin
`Data/combined_Room1.csv … combined_Room5.csv` ke `AI-OBEP-MLPaper/data/raw/`.
Atau arahkan ke folder lain: set environment variable `ROBOD_DIR=/path/ke/folder/csv`.

Folder `data/raw/` tidak di-commit (sudah di `.gitignore`).

---

## Urutan menjalankan

### v1 — `notebooks/` (jalankan berurutan)

| Notebook | Isi | Waktu kira-kira |
|---|---|---|
| `01_explore_dataset` | memahami ROBOD | < 1 menit |
| `02_train_surrogate` | model zoo + weighted ensemble | beberapa menit |
| `03_run_optimisation` | NSGA-II / NSGA-III / SMS-EMOA | lama; set `QUICK = True` untuk uji cepat |

### v2 — `AI-OBEP-MLPaper/notebooks/`

| Notebook | Isi | Perlu dijalankan? |
|---|---|---|
| `00_get_data` | unduh ROBOD | ya, sekali |
| `01_surrogate_baseline` | ablasi data, model zoo, protokol paper | opsional |
| `02_tune_surrogate`, `02b_…` | PSO / GA / HHO men-tuning XGBoost | opsional (lama) |
| `03_final_surrogate` | pilih & simpan surrogate | hanya bila model tersimpan gagal dibuka |
| `04_feature_importance` | SHAP + permutation | opsional |
| `05_tune_nsga2` | metaheuristik men-tuning NSGA-II | opsional (lama) |
| `06_run_nsga2` | 30 seed × 25.000 evaluasi, solusi A/B/C | inti hasil (lama) |
| `07_decomposition` | kontribusi dinding vs setpoint | opsional |

Model surrogate hasil latih **sudah ada di repo** (`outputs/models/surrogate.joblib`) dan semua hasil
(CSV/JSON/gambar) juga sudah ada di `outputs/`. Jadi untuk sekadar melihat hasil atau melanjutkan dari
NSGA-II, kamu tidak perlu melatih ulang dari nol. Pipeline lengkap ±2–3 jam di 2 core CPU.

Menjalankan semua notebook v2 sekaligus (Git Bash / Linux / macOS, venv aktif):

```bash
cd AI-OBEP-MLPaper
bash notebooks/run_all_notebooks.sh                 # dari awal
bash notebooks/run_all_notebooks.sh 05_tune_nsga2   # lanjut dari notebook 05
```

---

## Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| `ModuleNotFoundError: No module named 'pymoo'` (atau paket lain) | Kernel/interpreter bukan `.venv`. Pilih ulang kernel di VS Code/Jupyter, atau `pip install -r requirements-all.txt` di venv yang aktif. |
| `ModuleNotFoundError: No module named 'aiobep'` / `'obep_mlpaper'` | Notebook dibuka dari luar folder repo. Pastikan folder repo yang dibuka di VS Code/Jupyter (di Colab: `%cd /content/AI-OBEP`). |
| `ImportError: cannot import name 'pmv_ppd_iso'` | pythermalcomfort terlalu lama → `pip install -U "pythermalcomfort>=3.8"`. |
| `FileNotFoundError … combined_Room3.csv` | Data belum disiapkan — lihat [Data](#data). |
| Gagal membuka `surrogate.joblib` | Versi xgboost/scikit-learn jauh berbeda dengan saat model dibuat. Jalankan ulang `03_final_surrogate` (perlu `best_xgb_params.json`, sudah ada di repo). |
| `pip install` gagal untuk `lightgbm`/`xgboost`/`shap` | Biasanya Python terlalu baru. Pakai Python 3.11 atau 3.12. |
| PowerShell menolak `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, lalu coba lagi. |
| `bash: command not found` di Windows | Jalankan skrip `.sh` dari **Git Bash**, atau pakai notebook (`00_get_data`) sebagai gantinya. |
| Notebook optimasi sangat lama | Kurangi seed/evaluasi: v1 `QUICK = True`; v2 ubah `args = SimpleNamespace(seeds=…, evals=…)` di sel pertama. |
| Angka sedikit berbeda dari README | Normal bila versi library beda. Hasil NSGA-II v2 identik per seed dengan pymoo ≥ 0.6.2. |
