"""Cek cepat apakah instalasi AI-OBEP di device ini sudah benar.

Jalankan dari folder repo:
    python check_setup.py            # cek paket + data
    python check_setup.py --fix-data # sekalian ekstrak SupplementaryData.zip ke data/raw

Tidak mengubah apa pun selain (opsional) mengekstrak data.
"""
import importlib
import importlib.util
import platform
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"

# (nama import, nama pip, dibutuhkan oleh)
PACKAGES = [
    ("numpy", "numpy", "semua"),
    ("pandas", "pandas", "semua"),
    ("scipy", "scipy", "semua"),
    ("sklearn", "scikit-learn", "semua"),
    ("matplotlib", "matplotlib", "semua"),
    ("joblib", "joblib", "semua"),
    ("pymoo", "pymoo", "optimasi (NSGA-II)"),
    ("pythermalcomfort", "pythermalcomfort", "komfort termal (PMV)"),
    ("xgboost", "xgboost", "AI-OBEP-MLPaper"),
    ("lightgbm", "lightgbm", "AI-OBEP-MLPaper"),
    ("shap", "shap", "AI-OBEP-MLPaper (feature importance)"),
    ("jupyter_core", "jupyter", "menjalankan notebook (opsional di VS Code/Colab)"),
]


def check_python() -> bool:
    v = sys.version_info
    msg = f"Python {platform.python_version()} ({sys.executable})"
    if v < (3, 10):
        print(BAD, msg, "-> butuh Python >= 3.10")
        return False
    if v >= (3, 14):
        print(WARN, msg, "-> belum diuji; kalau ada paket gagal install, pakai 3.10-3.12")
    else:
        print(OK, msg)
    return True


def check_packages() -> bool:
    ok = True
    missing = []
    for mod, pip_name, used_by in PACKAGES:
        try:
            m = importlib.import_module(mod)
            print(OK, f"{pip_name:18s} {getattr(m, '__version__', '?'):10s} ({used_by})")
        except Exception as e:  # noqa: BLE001
            optional = pip_name == "jupyter"
            print(WARN if optional else BAD, f"{pip_name:18s} {'-':10s} ({used_by}) -> {type(e).__name__}")
            if not optional:
                ok = False
                missing.append(pip_name)
    # API check: pythermalcomfort harus v3 (pmv_ppd_iso)
    try:
        from pythermalcomfort.models import pmv_ppd_iso  # noqa: F401
    except Exception:  # noqa: BLE001
        if "pythermalcomfort" not in missing:
            print(BAD, "pythermalcomfort terlalu lama: butuh >= 3.8 (fungsi pmv_ppd_iso)")
            ok = False
    if missing:
        print("\n   Perbaiki dengan:  pip install -r requirements-all.txt")
    return ok


def check_data(fix: bool) -> None:
    raw = ROOT / "data" / "raw"
    need = [raw / f"combined_Room{k}.csv" for k in range(1, 6)]
    zip_path = ROOT / "SupplementaryData.zip"
    if all(p.exists() for p in need):
        print(OK, "Data proyek root ada di data/raw/")
    elif fix and zip_path.exists():
        raw.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.startswith("combined_Room") and name.endswith(".csv"):
                    z.extract(name, raw)
        print(OK, "SupplementaryData.zip diekstrak ke data/raw/")
    else:
        print(WARN, "data/raw/ belum ada -> notebook 01 akan mengekstraknya otomatis, "
                    "atau jalankan: python check_setup.py --fix-data")

    raw2 = ROOT / "AI-OBEP-MLPaper" / "data" / "raw"
    if (raw2 / "combined_Room3.csv").exists():
        print(OK, "Data AI-OBEP-MLPaper ada di AI-OBEP-MLPaper/data/raw/")
    else:
        print(WARN, "AI-OBEP-MLPaper/data/raw/ belum ada -> jalankan notebook 00_get_data "
                    "atau: bash AI-OBEP-MLPaper/scripts/00_get_data.sh (butuh git & internet)")

    model = ROOT / "AI-OBEP-MLPaper" / "outputs" / "models" / "surrogate.joblib"
    if model.exists() and importlib.util.find_spec("xgboost") is None:
        print(WARN, "surrogate.joblib belum bisa dicek (xgboost belum terpasang)")
    elif model.exists():
        try:
            import joblib
            sys.path.insert(0, str(ROOT / "AI-OBEP-MLPaper" / "src"))
            joblib.load(model)
            print(OK, "Model surrogate tersimpan (surrogate.joblib) bisa dibuka")
        except Exception as e:  # noqa: BLE001
            print(WARN, f"surrogate.joblib gagal dibuka ({type(e).__name__}: {e}). "
                        "Biasanya beda versi xgboost/sklearn -> jalankan ulang notebook 03_final_surrogate")


def main() -> int:
    print(f"Folder repo: {ROOT}\n")
    ok = check_python()
    print()
    ok = check_packages() and ok
    print()
    check_data(fix="--fix-data" in sys.argv)
    print("\nSemua paket wajib terpasang. Siap dipakai!" if ok else "\nMasih ada yang perlu diperbaiki (lihat [FAIL]).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
