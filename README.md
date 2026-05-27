# 📈 Prediksi Harga Saham IHSG dengan XGBoost: Perbandingan GridSearchCV vs Bayesian Optimization (Optuna)

---

## 1. Ringkasan Proyek

Proyek ini bertujuan memprediksi **harga penutupan harian (Closing Price)** Indeks Harga Saham Gabungan (**IHSG / ^JKSE**) menggunakan algoritma **XGBoost** dengan pendekatan regresi berbasis indikator teknikal.

Inti penelitian adalah **membandingkan dua metode hyperparameter tuning**:

| | **GridSearchCV** (Baseline) | **Optuna / Bayesian Optimization** (Proposed) |
|---|---|---|
| **Strategi** | Exhaustive discrete search | Probabilistic smart search |
| **Kelebihan** | Menjamin semua kombinasi dicoba | Lebih cepat & bisa menemukan parameter di ruang kontinu |
| **Kekurangan** | Lambat, terbatas pada grid diskrit | Tidak menjamin global optimum |

### Paper Referensi (Baseline)
> **"Stock Price Prediction Method Based on XGboost Algorithm"**  
> Yifan Zhang — Lanzhou University of Technology, 2023  
> DOI: [10.2991/978-94-6463-030-5_60](https://doi.org/10.2991/978-94-6463-030-5_60)

Kontribusi proyek ini terhadap paper di atas:
1. Mengganti metode tuning dari GridSearchCV ke **Bayesian Optimization (Optuna)** sebagai metode usulan
2. Memperluas ruang pencarian parameter ke **ruang kontinu** (bukan diskrit)
3. Menambahkan parameter anti-overfitting (`subsample`, `colsample_bytree`)
4. Menguji pada indeks pasar berkembang (**IHSG**) dengan volatilitas tinggi

---

## 2. Dataset

| Aspek | Detail |
|-------|--------|
| **Subjek** | IHSG (Jakarta Composite Index) — Ticker: `^JKSE` |
| **Sumber** | Yahoo Finance API (`yfinance`) |
| **Periode** | 1 Januari 2010 – 31 Desember 2025 (~15 tahun) |
| **Target** | Harga penutupan hari berikutnya (A+1) — `Target_Close = Close.shift(-1)` |

### Pembagian Data (Sequential / Kronologis)

```
|◀──────── 70% Train ─────────▶|◀── 15% Val ──▶|◀── 15% Test ──▶|
       2010 — ~2021                ~2021-2023         ~2023-2025
```

> ⚠️ **Catatan**: Pembagian dilakukan secara **kronologis (sequential split)**, bukan random split, agar sesuai dengan sifat data time-series dan mencegah data leakage.

---

## 3. Feature Engineering (Indikator Teknikal)

Model menggunakan **7 indikator teknikal** yang dihitung dari harga penutupan (`Close`) sebagai fitur input:

| # | Fitur | Deskripsi | Window |
|---|-------|-----------|--------|
| 1 | **EMA_9** | Exponential Moving Average | 9 hari |
| 2 | **SMA_5** | Simple Moving Average (jangka pendek) | 5 hari |
| 3 | **SMA_15** | Simple Moving Average (jangka menengah) | 15 hari |
| 4 | **SMA_30** | Simple Moving Average (jangka panjang) | 30 hari |
| 5 | **RSI** | Relative Strength Index — pengukur momentum | 14 hari |
| 6 | **MACD** | Moving Average Convergence Divergence (EMA12 − EMA26) | 12/26 |
| 7 | **MACD_signal** | Signal line MACD | span 9 |

Library yang digunakan: [`ta`](https://github.com/bukosabino/ta) (Technical Analysis Library in Python)

---

## 4. Hyperparameter Tuning

### A. GridSearchCV (Baseline — Mengikuti Paper Referensi)

Pencarian exhaustive pada grid diskrit berikut (sesuai Table 1 paper):

| Parameter | Kandidat Nilai |
|-----------|---------------|
| `n_estimators` | [100, 200, 300, 400] |
| `learning_rate` | [0.001, 0.005, 0.01, 0.05] |
| `max_depth` | [8, 10, 12, 15] |
| `gamma` | [0.001, 0.005, 0.01, 0.02] |
| `random_state` | 42 |

**Total kombinasi: 256** — setiap kombinasi dilatih & dievaluasi pada validation set.

### B. Optuna / Bayesian Optimization (Metode Usulan)

Pencarian cerdas berbasis Teorema Bayes pada ruang **kontinu** yang diperluas:

| Parameter | Range Pencarian | Distribusi |
|-----------|----------------|------------|
| `n_estimators` | 100 – 1000 | Uniform integer |
| `learning_rate` | 0.001 – 0.1 | **Log-uniform** |
| `max_depth` | 3 – 16 | Uniform integer |
| `gamma` | 0.0001 – 0.1 | **Log-uniform** |
| `subsample` | 0.6 – 1.0 | Uniform float |
| `colsample_bytree` | 0.6 – 1.0 | Uniform float |
| `random_state` | 42 (fixed) | — |

**Total iterasi: 50 trials** — setiap iterasi belajar dari hasil iterasi sebelumnya.

---

## 5. Hasil Eksperimen

### Parameter Terbaik

| Parameter | GridSearchCV | Optuna |
|-----------|-------------|--------|
| `n_estimators` | 400 | 930 |
| `learning_rate` | 0.05 | 0.0367 |
| `max_depth` | 12 | 3 |
| `gamma` | 0.01 | 0.0015 |
| `subsample` | — | 0.6013 |
| `colsample_bytree` | — | 0.7722 |

> 💡 **Insight**: Optuna menemukan bahwa model **dangkal** (`max_depth=3`) dengan **lebih banyak trees** (`n_estimators=930`) dan regularisasi tambahan (`subsample`, `colsample_bytree`) lebih optimal — berlawanan dengan Grid Search yang memilih pohon dalam (`max_depth=12`).

### Perbandingan Performa

| Metrik | GridSearchCV (Baseline) | Optuna (Proposed) | Pemenang |
|--------|------------------------|-------------------|----------|
| **Validation MSE** | 127,778.94 | 102,715.10 | ✅ Optuna |
| **Test MSE** | 853,577.93 | 817,097.75 | ✅ Optuna |
| **Test RMSE** | 923.89 | 903.93 | ✅ Optuna |
| **Waktu Eksekusi** | 122.40 detik | 21.85 detik | ✅ Optuna |
| **MAPE** | ~10.65% | ~10.14% | ✅ Optuna |

### Ringkasan Hasil

```
┌─────────────────────────────────────────────────────────────┐
│  Optuna menghasilkan model yang:                            │
│                                                             │
│  📉 MSE 4.3% lebih rendah dari GridSearch                   │
│  📉 RMSE 2.2% lebih rendah dari GridSearch                  │
│  ⚡ 5.6x lebih cepat dari GridSearch                        │
│  📊 MAPE ~10.14% (rata-rata meleset 10% dari harga asli)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Visualisasi Hasil

### Actual vs Predicted Price (Test Set)
Plot perbandingan harga asli IHSG vs prediksi model pada periode test (~2023–2025).  
📁 `outputs/plots/actual_vs_predicted.png`

### Feature Importance
Diagram batang menunjukkan skor kepentingan relatif setiap indikator teknikal setelah optimasi Bayesian.  
📁 `outputs/plots/feature_importance.png`

### Percentage Error Analysis
Grafik persentase error harian dan histogram distribusi error untuk masing-masing model.  
📁 `outputs/plots/percentage_error_xgboost__optuna.png`  
📁 `outputs/plots/percentage_error_xgboost__gridsearch.png`

---

## 7. Struktur Proyek

```
Project_ML/
│
├── README.md                  # Dokumentasi proyek (file ini)
├── Documentation.md           # Spesifikasi riset detail
├── config.py                  # Fungsi load data & split train/val/test
├── requirements.txt           # Dependensi Python
│
├── src/                       # Source code utama
│   ├── download_data.py       # Download data IHSG dari Yahoo Finance
│   └── features.py            # Feature engineering (indikator teknikal)
│
├── experiments/                # Script eksperimen
│   ├── run_gridsearch.py      # Baseline: XGBoost + GridSearchCV
│   ├── run_optuna.py          # Proposed: XGBoost + Optuna
│   ├── generate_plots.py      # Pembuatan visualisasi
│   └── check_percentage_error.py  # Analisis MAPE & persentase error
│
├── data/
│   ├── raw/                   # Data mentah dari Yahoo Finance
│   └── processed/             # Data setelah feature engineering
│
└── outputs/
    ├── models/                # Model XGBoost tersimpan (.json)
    ├── metrics/               # Hasil metrik performa (.json)
    └── plots/                 # Grafik visualisasi (.png)
```

---

## 8. Cara Menjalankan

### Prasyarat
```bash
pip install -r requirements.txt
```

### Langkah Eksekusi (Urutan)
```bash
# 1. Download data IHSG dari Yahoo Finance
python src/download_data.py

# 2. Feature engineering (hitung indikator teknikal)
python src/features.py

# 3. Jalankan baseline (GridSearchCV)
python experiments/run_gridsearch.py

# 4. Jalankan metode usulan (Optuna)
python experiments/run_optuna.py

# 5. Generate visualisasi
python experiments/generate_plots.py

# 6. (Opsional) Cek persentase error / MAPE
python experiments/check_percentage_error.py
```

---

## 9. Metrik Evaluasi

| Metrik | Rumus | Kegunaan |
|--------|-------|----------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Mengukur rata-rata kuadrat error — sensitif terhadap outlier |
| **RMSE** | $\sqrt{MSE}$ | Sama dengan MSE tapi dalam satuan asli (poin indeks) |
| **MAPE** | $\frac{1}{n}\sum\left\|\frac{y_i - \hat{y}_i}{y_i}\right\| \times 100\%$ | Rata-rata persentase error — intuitif dan mudah diinterpretasi |

---

## 10. Teknologi & Library

| Library | Versi | Fungsi |
|---------|-------|--------|
| `xgboost` | — | Algoritma model utama |
| `scikit-learn` | — | Metrik evaluasi & ParameterGrid |
| `optuna` | — | Bayesian hyperparameter optimization |
| `yfinance` | — | Download data saham |
| `ta` | — | Perhitungan indikator teknikal |
| `pandas` | — | Manipulasi data |
| `numpy` | — | Operasi numerik |
| `matplotlib` | — | Visualisasi grafik |
| `seaborn` | — | Styling visualisasi |

---

## 11. Kesimpulan

1. **Optuna (Bayesian Optimization) terbukti lebih unggul** dibandingkan GridSearchCV dalam semua metrik evaluasi: MSE lebih rendah, RMSE lebih rendah, dan waktu eksekusi **5.6x lebih cepat**.

2. Optuna menemukan konfigurasi parameter yang **berbeda secara signifikan** dari GridSearch — menghasilkan model yang lebih dangkal (`max_depth=3`) namun dengan lebih banyak pohon (`n_estimators=930`), mengindikasikan strategi ensemble yang lebih robust terhadap overfitting.

3. Kedua model memiliki **MAPE sekitar 10%**, yang menunjukkan adanya keterbatasan inherent dalam memprediksi harga saham menggunakan indikator teknikal saja, terutama untuk horizon waktu yang jauh dari data training.

4. Proyek ini sepenuhnya berada dalam domain **Machine Learning klasik** — menggunakan ensemble method (gradient boosting) berbasis decision tree, bukan deep learning / neural network.

---

## 12. Referensi

1. Zhang, Y. (2023). *Stock Price Prediction Method Based on XGboost Algorithm*. ICBBEM 2022, AHIS 5, pp. 595–603. DOI: [10.2991/978-94-6463-030-5_60](https://doi.org/10.2991/978-94-6463-030-5_60)
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD, pp. 785–794.
3. Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. Proceedings of the 25th ACM SIGKDD, pp. 2623–2631.
