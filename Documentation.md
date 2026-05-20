# Research Project Specification: Hyperparameter Optimization Comparison for Stock Price Prediction

## 1. Project Overview & Objective
Tujuan dari riset ini adalah menulis sebuah paper ilmiah komparatif di bidang kuantitatif finansial dan machine learning. Penelitian ini bertujuan untuk memprediksi nilai nominal harga saham harian menggunakan algoritma **XGBoost (eXtreme Gradient Boosting)** dan membandingkan dua metode optimisasi hiperparameter: **GridSearchCV** (sebagai metode *baseline*) dan **Bayesian Optimization menggunakan Optuna** (sebagai metode usulan).

Riset ini merupakan pengembangan langsung sekaligus pengisian *research gap* dari paper referensi: 
* **Paper Referensi:** *“Stock Price Prediction Method Based on XGboost Algorithm”* (Yifan Zhang, 2023).
* **Core Improvement:** Mengganti metode pencarian parameter konvensional (Grid Search) yang memakan waktu dan kaku, dengan pendekatan probabilistik cerdas (Bayesian/Optuna), memperluas ruang pencarian parameter ke ruang kontinu, serta mengujinya pada indeks pasar berkembang (*emerging market*) yang memiliki volatilitas tinggi.

---

## 2. Dataset & Target Variable
* **Subjek Data:** Indeks Harga Saham Gabungan / Jakarta Composite Index (IHSG) dengan kode ticker `^JKSE`.
* **Sumber Data:** Yahoo Finance API (`yfinance` di Python).
* **Rentang Waktu:** 1 Januari 2010 – 31 Desember 2025 (Periode 15 tahun untuk menangkap berbagai siklus pasar ekonomi, tren jangka panjang, dan guncangan fluktuasi ekstrem).
* **Tipe Tugas ML:** **Regresi Finansial (Financial Regression)**. Model akan memprediksi nilai nominal harga penutupan (*Closing Price*) harian secara spesifik, *bukan* klasifikasi biner (naik/turun).
* **Pembagian Data (Data Partitioning):** * 70% Training Set
  * 15% Validation Set
  * 15% Test Set

---

## 3. Preprocessing & Feature Engineering
Sebelum dilatih, data harga saham harian asli (Open, High, Low, Close, Volume) akan didekomposisi untuk dianalisis kelayakan deret waktunya (*price decomposition: trend, seasonality, residual*). Selanjutnya, serangkaian indikator teknikal berikut akan dihitung dan digunakan sebagai *Feature Factors* (Variabel Input):
1. **EMA 9:** Exponential Moving Average 9 hari.
2. **SMA 5, SMA 15, SMA 30:** Simple Moving Average untuk jangka pendek dan menengah.
3. **RSI (Relative Strength Index):** Pengukur momentum dan sentimen jenuh beli/jenuh jual pasar.
4. **MACD & MACD Signal:** Indikator tren dan momentum perpindahan harga.

---

## 4. Hyperparameter Tuning Framework (The Core Comparison)

### Metode A: GridSearchCV (Baseline - Mereplikasi Paper Referensi)
* **Karakteristik:** Pencarian menyeluruh secara diskrit (*exhaustive discrete search*).
* **Ruang Parameter (Dibatasi/Diskrit):**
  * `n_estimators`: [100, 200, 300, 400]
  * `learning_rate`: [0.001, 0.005, 0.01, 0.05]
  * `max_depth`: [8, 10, 12, 15]
  * `gamma`: [0.001, 0.005, 0.01, 0.02]

### Metode B: Optuna / Bayesian Optimization (Proposed Method)
* **Karakteristik:** Pencarian probabilistik berbasis Teorema Bayes yang belajar dari kesalahan iterasi sebelumnya secara dinamis.
* **Ruang Parameter (Diperluas & Kontinu):**
  * `n_estimators`: `suggest_int('n_estimators', 100, 1000)` (Skala diperluas)
  * `learning_rate`: `suggest_float('learning_rate', 0.001, 0.1, log=True)` (Distribusi kontinu/logaritmik)
  * `max_depth`: `suggest_int('max_depth', 3, 16)` (Membuka peluang pohon lebih dangkal untuk cegah overfitting)
  * `gamma`: `suggest_float('gamma', 1e-4, 0.1, log=True)`
  * *Tambahan parameter penjinak overfitting:* `subsample` dan `colsample_bytree`.

---

## 5. Evaluation Metrics & Paper Contributions
Paper komparatif ini akan mengevaluasi kedua pendekatan berdasarkan matriks performa utama:
1. **Akurasi Prediksi Regresi:** Dievaluasi menggunakan **Mean Squared Error (MSE)** dan **Root Mean Square Error (RMSE)** pada *test set*. Hipotesisnya adalah Optuna mampu menemukan kombinasi parameter yang menghasilkan MSE/RMSE lebih rendah (lebih akurat) daripada Grid Search.
2. **Efisiensi Komputasi:** Mengukur waktu eksekusi run-time (**Execution Time dalam satuan detik/menit**). Menunjukkan seberapa cepat Optuna mencapai konvergensi dibanding Grid Search.
3. **Feature Importance Analysis:** Mengeluarkan skor kepentingan fitur (F-score) pasca-optimasi untuk menganalisis indikator teknikal mana yang paling berpengaruh pada indeks IHSG selama 2010-2025.

---

## 6. What I Need From You (Instructions for AI Assistants)
Jika saya meminta Anda untuk melanjutkan pengerjaan proyek ini, Anda harus mampu membantu saya dalam:
1. Menulis kode Python bersih berbasis `yfinance`, `ta` (atau perhitungan manual fungsi finansial), `xgboost`, `scikit-learn`, dan `optuna`.
2. Menyusun visualisasi data (Plot visual dekomposisi waktu, grafik perbandingan *Actual vs Predicted Prices*, dan diagram batangan *Feature Importance*).
3. Membantu menulis draf naskah paper akademik bagian demi bagian (Introduction, Methodology, Results and Discussion, Conclusion) dalam gaya bahasa jurnal ilmiah terindeks (seperti IEEE atau Scopus).
