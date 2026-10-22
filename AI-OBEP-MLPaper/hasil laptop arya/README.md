# Hasil laptop Arya (run penuh 22 Oktober 2026)

Hasil menjalankan seluruh pipeline AI-OBEP-MLPaper (skrip 01-08) pada satu laptop.
Angka di folder ini adalah angka final yang dipakai untuk laporan dan paper;
angka di laporan 21 Oktober dan di README lama sebagian berbeda (lihat bagian 11 laporan).

## Lingkungan

- Prosesor Intel Core i5-1235U (10 core, 12 thread), tanpa GPU, Python 3.12.5
- Skrip 05-08 memakai 6 proses paralel (`n_jobs=6`, joblib); skrip 01-04 dan 02b tanpa joblib
- Seed dikunci (split data 0; metaheuristik 0-2; tuning NSGA-II 101-103; evaluasi akhir 1-30)

## Waktu komputasi

| Skrip | Isi | Waktu |
| --- | --- | --- |
| 01 | Ablasi, model zoo, protokol paper | 20 detik |
| 02 | Tuning XGBoost (PSO, GA, HHO) | 15 menit 45 detik |
| 02b | Tuning pada protokol Lyu et al. | 2 menit 3 detik |
| 03 | Surrogate final | 33 detik |
| 04 | Feature importance | 5 detik |
| 05 | Tuning NSGA-II | 5 menit 33 detik |
| 06 | 30 seed x 4 konfigurasi x 2 kasus | 6 menit 4 detik |
| 07 | Dekomposisi dinding vs setpoint | 1 menit 17 detik |
| 08 | Uji sensitivitas beta dan T_ref | 4 menit 46 detik |
| Total | | 36 menit 26 detik |

Sumber: `outputs/waktu_komputasi.csv`.

## Isi folder

- `outputs/` : hasil run (surrogate, importance, nsga2, figures, models, waktu_komputasi.csv)
- `scripts/08_sensitivity.py` : skrip baru uji sensitivitas beta dan T_ref (tidak ada di folder scripts utama)
- `scripts/run_all.sh` : run_all dengan langkah 08 ditambahkan

Catatan: `outputs/models/` biasanya diabaikan oleh `.gitignore` repositori; di folder ini
ikut disertakan (sekitar 200 KB) supaya skrip 04-08 bisa dijalankan ulang tanpa melatih ulang surrogate.

## Ringkasan hasil kunci

- Surrogate (protokol ketat): R2 0,942; CV(RMSE) 27,0 %; NMBE -4,0 %; R2 jam operasional 0,466
- NSGA-II hasil tuning: HV ratio 98,8-99,3 % (kasus 1), 97,8-98,4 % (kasus 2); HHO-tuned terbaik di kedua kasus
- Solusi C (semua penghuni PMV mutlak <= 0,5): hemat 19,1 % (kasus 1) dan 26,5 % (kasus 2) terhadap layout awal 24 C
- Kontribusi dinding saja (setpoint tetap 24 C): 1,2 % dan 6,5 %
- Sensitivitas beta 5-12 %/C: hemat solusi C 11,0-20,5 % (kasus 1) dan 18,2-27,7 % (kasus 2); kontribusi dinding hampir tidak berubah
