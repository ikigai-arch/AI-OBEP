# AI-OBEP — model ML untuk memahami ROBOD + replikasi paper "data-driven multi-objective optimisation … flexible building spaces"

Repo ini punya tiga tahap, masing-masing satu notebook di `notebooks/` (jalankan berurutan; logika inti ada di modul `src/aiobep/`):

| Notebook | Isi | Output |
|---|---|---|
| `notebooks/01_explore_dataset.ipynb` | Memahami ROBOD: profil harian, okupansi vs energi, data hilang, sensor proxy okupansi | `outputs/eda/` |
| `notebooks/02_train_surrogate.ipynb` | Surrogate energi HVAC per ruangan (7 fitur → energi), model zoo termasuk **MLP** + weighted ensemble | `outputs/surrogate_*.csv`, `fig_surrogate_performance.png` |
| `notebooks/03_run_optimisation.ipynb` | Optimasi multi-objektif posisi dinding (NSGA-II / NSGA-III / SMS-EMOA, pymoo) di dua studi kasus paper | `outputs/opt_ensemble/` |

```bash
pip install -r requirements.txt jupyter
unzip SupplementaryData.zip -d data/raw      # 5 file combined_Room*.csv (data/raw tidak di-commit)
jupyter lab notebooks/                        # jalankan 01 -> 02 -> 03
```
Di notebook 03 ubah `SEEDS` (paper: 30), `MODEL = "surrogate_mlp"` untuk memakai MLP, atau `QUICK = True` untuk uji cepat (1 seed, 30 generasi, output ke `outputs/_quick`).

## 1. Memahami dataset ROBOD
5 ruangan gedung SDE4 NUS (Singapura), 5 menit, 7 Sep–23 Des 2021, hari kerja saja.
Detail per ruangan: `outputs/eda/room_overview.csv`.

- **R1/R2** kuliah (FCU), **R3** kantor admin, **R4** kantor peneliti, **R5** perpustakaan (AHU/VAV). R1/R2 tidak punya kolom VAV/AHU (kolom itu kosong, bukan hilang).
- Data hilang sangat sedikit (<1%), kecuali `supply_air_flow`/`damper_position` R5 (19%).
- Okupansi mengikuti orang (puncak sore), tetapi **energi HVAC mengikuti jadwal** (08:30–18:40 di R3–R5): lihat `outputs/eda/daily_profiles.png`.
- Di dalam jam operasional, korelasi energi–okupansi lemah (Spearman 0,07–0,12 di kantor, bahkan −0,19 di perpustakaan). Artinya HVAC saat ini **tidak responsif terhadap okupansi** — inilah celah yang ingin diisi AI-OBEP (occupancy-centric).
- Untuk lapisan edge: Wi-Fi paling berkorelasi dengan jumlah orang asli (r 0,52–0,89); CO₂ lemah–sedang (−0,01–0,53); sound level tidak berguna (`occupancy_proxy_correlations.csv`).

## 2. Surrogate (Sec. 3 paper)
Input 7 fitur persis paper: suhu & RH dalam ruang, jumlah orang, suhu luar, RH luar, kecepatan angin, radiasi surya. Target: ceiling fan + chilled water + fan AHU (Wh/m² per 5 menit), kantor R3+R4 digabung, split per hari 70/15/15. AutoGluon diganti model sklearn (RF, ExtraTrees, HistGB, **MLP**, Ridge) + weighted ensemble NNLS.

| Model | R² semua jam | R² jam operasional |
|---|---|---|
| ExtraTrees | 0,49 | 0,11 |
| HistGB | 0,48 | 0,08 |
| WeightedEnsemble | 0,47 | 0,06 |
| MLP | 0,35 | −0,22 |
| Ridge | 0,22 | −0,27 |

**Perlu jujur:** R² paper 0,933 (12 hari satu kantor, outlier dihaluskan) **tidak tercapai** di sini (~0,47). Hampir seluruh R² berasal dari pembeda jam ON/OFF HVAC; saat HVAC menyala, 7 fitur nyaris tak menjelaskan variasi energi. Generalisasi antar-ruangan juga buruk (`surrogate_cross_room.csv`). Dugaan perbaikan (belum dicoba): fitur jam/jadwal, setpoint & suhu supply air, lag/time-series (LSTM/TFT seperti paper), per-ruangan. MLP di sini belum di-tuning.

## 3. Optimasi dinding bergerak (Sec. 2 & 4 paper)
`src/aiobep/layout.py` mengimplementasikan model matematika paper: variabel keputusan = posisi dinding dalam; aturan update ruangan (Alg. 3 luas, Alg. 4 penghuni, Alg. 5 suhu/RH dari zona pemanas); constraint luas minimum & semua penghuni tertampung. `optimise.py`: mutasi kustom (Alg. 6: Gaussian + sort), SBX, populasi awal = layout awal. Objektif: `f_cost` (energi total surrogate, kW) dan `f_tc` (rata-rata |PMV| ISO 7730 per penghuni, `pythermalcomfort`).
Verifikasi: f_tc awal 0,951 (paper 0,938) dan 0,587 (paper 0,586) — model komfort cocok.

Hasil (surrogate ensemble, 5 seed × 500 generasi; ukuran paper: 30 seed):

| Kasus | Solusi A (energi min) | Solusi B (komfort terbaik) |
|---|---|---|
| 1 (sederhana) | hemat 6,2% energi, f_tc 0,8% lebih buruk | f_tc 2,4% lebih baik, energi 0,2% lebih boros |
| 2 (Watson) | hemat 11,0%, f_tc 11,4% lebih buruk | f_tc 13,3% lebih baik, energi 1,7% lebih boros |

Angka paper (≈11,7–12% hemat, komfort +5–14%) cukup dekat, tetapi jangan baca sebagai klaim penghematan nyata: surrogate-nya lemah (R² ~0,47) sehingga optimasi hanya sebaik surrogate itu. Solusi A/B adalah titik ekstrem dari gabungan semua front (semua algoritma & seed). Hypervolume ternormalisasi: kasus 1 ≈0,0575–0,0577 (ketiganya setara); kasus 2 NSGA-II 0,094 > NSGA-III 0,091 > SMS-EMOA 0,089 (urutan NSGA-II terbaik sama dengan paper).

**Asumsi saya** (paper tidak merinci): surrogate dilatih pada kantor R3+R4 lalu dipakai untuk semua ruangan termasuk ruang rapat; bobot ruangan w_j = 1 (energi absolut kW); kedalaman ruang B dipilih sendiri (4 m / 3 m); kondisi luar kasus 1 & 2 sama dengan paper (27,5 °C, 85%, 650 W/m², 1 m/s); η mutasi dianggap simpangan baku dalam meter; hypervolume dinormalisasi dengan nilai layout awal (ref 1,2). R-NSGA-II tidak diimplementasikan.

## 4. Pertanyaan untuk Miss Sahidah (arah AI-OBEP)
Replikasi ini memakai variabel keputusan paper (posisi dinding), padahal AI-OBEP belum menetapkan: (1) variabel keputusan (setpoint? dinding? jadwal?), (2) definisi comfort (PMV atau proxy), (3) ruangan hardware. Karena setpoint ada di data (`temp_setpoint`) tetapi bukan input surrogate paper, varian "setpoint + okupansi" butuh keputusan itu dulu.
