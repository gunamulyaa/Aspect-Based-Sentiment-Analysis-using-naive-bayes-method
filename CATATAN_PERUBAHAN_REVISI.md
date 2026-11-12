# Catatan Perubahan & Laporan Revisi (Acuan Bab I–III)
**Topik Penelitian**: Analisis Sentimen Berbasis Aspek (*Aspect-Based Sentiment Analysis* / ABSA) terhadap Kebijakan Pembatasan Penggunaan Media Sosial bagi Anak di Bawah 16 Tahun Menggunakan Metode *Multinomial Naive Bayes* dan Kamus *InSet*.

---

## 1. Ringkasan Pemenuhan 8 Poin Revisi (Bab I–III)

| No | Poin Revisi | Perubahan yang Dilakukan | Berkas Terkait |
| :-: | :--- | :--- | :--- |
| **1** | **Pemetaan 4 Kategori Aspek & Kata Kunci (Termasuk Bentuk Setelah Stemming)** | Mengubah pemetaan aspek dari 6 topik lama menjadi **4 kategori aspek sesuai Bab I–III**: `Perlindungan Anak`, `Peran Orang Tua`, `Peran Pemerintah/Komdigi`, dan `Efektivitas Kebijakan`. Kata kunci mencakup bentuk kata dasar, kata berimbuhan, dan bentuk akar hasil *stemming* Sastrawi (mis. `perlindungan`/`melindungi` $\rightarrow$ `lindung`, `pengawasan` $\rightarrow$ `awas`, `pendampingan` $\rightarrow$ `damping`, `pengasuhan` $\rightarrow$ `asuh`, `pendidikan` $\rightarrow$ `didik`, `pemerintah` $\rightarrow$ `perintah`, `kebijakan` $\rightarrow$ `bijak`, `peraturan`/`mengatur` $\rightarrow$ `atur`, `pembatasan` $\rightarrow$ `batas`, `penerapan` $\rightarrow$ `terap`, `setuju` $\rightarrow$ `tuju`). | `Label/Dictionary.py`, `Preprocessing/preprocessing.py`, `Label/Labeling.py` |
| **2** | **Pembentukan Pasangan Komentar–Aspek Multi-Label Kontekstual** | Satu komentar yang membahas lebih dari satu aspek dipecah menjadi beberapa baris pasangan `(komentar, aspek)`. Dari `2.367` komentar unik beraspek (di antaranya `1.002` komentar multi-aspek), terbentuk **`3.733` pasangan komentar–aspek**. Setiap pasangan aspek dievaluasi pada klausa/konteks aspeknya (memisahkan klausa pada konjungsi kontras seperti `tapi`, `tetapi`, `namun`, `padahal`, `sedangkan`, `meski`, `walaupun`) sehingga satu komentar dapat memiliki label sentimen berbeda pada aspek yang berbeda. | `Label/Labeling.py`, `Label/dataset_pasangan_komentar_aspek.xlsx` |
| **3** | **Penggunaan Kamus InSet, Pelestarian Negasi, & Pengeluaran Komentar Tanpa Aspek** | Menerapkan kamus **InSet (Indonesian Sentiment Lexicon)** berbobot $-5$ s.d. $+5$ (`Label/inset_positive.tsv` & `Label/inset_negative.tsv`). Memperbaiki bug pada `Preprocessing/preprocessing.py` yang sebelumnya menghapus kata negasi (`tidak`, `bukan`, `jangan`) pada `ASPECT_EXCLUSION` dan memasukkan `belum` ke `STOPWORDS`. Seluruh kata negasi (`tidak`, `bukan`, `jangan`, `belum`, `tanpa`, `kurang`, `tak`) dipertahankan pada teks pelabelan maupun klasifikasi. Sebanyak **`1.634` komentar unik yang tidak memuat aspek dikeluarkan** (`Label/komentar_tanpa_aspek_dikeluarkan.xlsx`) dan **tidak** otomatis diberi label netral. | `Label/Dictionary.py`, `Preprocessing/preprocessing.py`, `Label/Labeling.py` |
| **4** | **Perhitungan TF-IDF Sesuai Rumus Bab III (Hanya dari Data Latih)** | Menyesuaikan rumus TF-IDF dengan Bab III: $\text{TF}(t,d) = \frac{f_{t,d}}{|d|}$, $\text{IDF}(t) = \log\left(\frac{N}{df_t}\right) + 1$, dan $W_{t,d} = \text{TF}(t,d) \times \text{IDF}(t)$. Kosakata ($|V| = 1.058$ kata unik) dan nilai $\text{IDF}$ **hanya dibentuk dari data latih** ($N = 3.834$). Kata pada data uji yang berada di luar kosakata latih (*Out-of-Vocabulary*) diabaikan tanpa kebocoran statistik data uji. | `Klasifikasi/naive_bayes_model.py` |
| **5** | **Pembagian Data Latih & Uji 80:20 Sebelum Penyeimbangan (0 Overlap Komentar)** | Pemisahan 80:20 dilakukan **sebelum penyeimbangan** berdasarkan `Comment_ID` unik (`1.894` komentar unik latih $\rightarrow$ `2.967` pasangan aspek; `473` komentar unik uji $\rightarrow$ `766` pasangan aspek). Seluruh pasangan aspek dari komentar yang sama berada di kelompok yang sama. Dibuktikan **0 tumpang tindih (*zero overlap*)** pada `Comment_ID`, `Comment Text`, `normalized_text`, dan `stemmed_text`. Penyeimbangan (*random oversampling*) **hanya dilakukan pada data latih** (`2.967` $\rightarrow$ `3.834` pasangan latih seimbang: `1.278` Positif, `1.278` Negatif, `1.278` Netral). | `Klasifikasi/naive_bayes_model.py`, `Klasifikasi/data_latih_80.xlsx`, `Klasifikasi/data_latih_80_balanced.xlsx`, `Klasifikasi/data_uji_20.xlsx`, `Klasifikasi/bukti_tidak_tumpang_tindih.csv` |
| **6** | **Pelatihan Ulang Naive Bayes & Evaluasi Keseluruhan serta Per Aspek** | Melatih ulang *Multinomial Naive Bayes* dengan *Laplace smoothing* pada data latih seimbang dan mengevaluasi pada `766` pasangan data uji alami. Menghasilkan *Confusion Matrix*, *Accuracy* (**92,30%**), *Precision* (**91,04%**), *Recall* (**92,35%**), *F1-Score* (**91,47%**), serta rincian evaluasi lengkap untuk keempat kategori aspek. | `Klasifikasi/naive_bayes_model.py`, `Klasifikasi/evaluasi_naive_bayes_lengkap.xlsx`, `Visualisasi/evaluation_per_aspect.csv` |
| **7** | **Tabel Distribusi 4 Aspek (Sebelum Balancing) & Tabel Eksperimen Balancing Terpisah** | Menyusun ulang tabel distribusi 4 aspek dengan kolom `Aspek`, `Positif`, `Negatif`, `Netral`, dan `Total` menggunakan data **sebelum penyeimbangan** (`3.733` pasangan aspek). Tabel hasil penyeimbangan data latih disajikan secara terpisah untuk menjelaskan eksperimen. | `Klasifikasi/aspect_sentiment_summary.xlsx`, `Klasifikasi/tabel_eksperimen_balancing_latih.xlsx` |
| **8** | **Pembaruan Grafik, Word Cloud, Dashboard, & Contoh Komentar Hasil Prediksi** | Memperbarui seluruh grafik batang, *heatmap Confusion Matrix* keseluruhan & per aspek, *Word Cloud* per sentimen & per aspek, serta *dashboard* `streamlit_app.py`. Menyertakan tabel contoh komentar beserta aspek, kata kunci, konteks klausa, label acuan InSet, dan hasil prediksi Naive Bayes. | `Visualisasi/generate_visualisasi.py`, `streamlit_app.py`, `Visualisasi/contoh_prediksi_komentar_aspek.csv`, `Klasifikasi/hasil_prediksi_data_uji.xlsx` |

---

## 2. Catatan Jumlah Data pada Setiap Tahap Pipeline

| Tahap | Jumlah Data | Keterangan |
| :--- | ---: | :--- |
| **1. Data Mentah Gabungan (`combined_dataset.xlsx`)** | **5.802** baris | Penggabungan 7 berkas Excel komentar TikTok pada folder `Dataset/` |
| **2. Baris Kosong / Tanpa Teks / Hanya Simbol Dihapus** | **-323** baris | `26` baris tanpa `Comment Text` (NaN) + `297` baris hanya berisi angka/simbol/tanda baca/emoji netral |
| **3. Data Komentar Valid Setelah Pembersihan (*Cleansing*)** | **5.479** baris | Komentar yang memiliki teks valid setelah *cleansing*, normalisasi, *filtering*, dan *stemming* |
| **4. Duplikasi Komentar Identik Dihapus** | **-1.478** baris | Komentar berulang/identik dihapus agar tidak terjadi kebocoran teks identik antara data latih dan data uji |
| **5. Komentar Unik Bersih (`dataset_preprocessed.xlsx`)** | **4.001** komentar | Setiap komentar unik diberi penanda `Comment_ID` (`C0001` s.d. `C4001`) |
| **6. Komentar Tanpa Aspek Terdeteksi (Dikeluarkan)** | **-1.634** komentar | Dikeluarkan ke `komentar_tanpa_aspek_dikeluarkan.xlsx` (**tidak** otomatis diberi label netral) |
| **7. Komentar Unik Memiliki $\ge 1$ Kategori Aspek** | **2.367** komentar | Terdiri atas `1.365` komentar ber-aspek tunggal dan `1.002` komentar multi-aspek |
| **8. Total Pasangan Komentar–Aspek Terbentuk (`dataset_pasangan_komentar_aspek.xlsx`)** | **3.733** pasangan | Rata-rata $1,58$ aspek per komentar unik beraspek |
| **9. Pembagian Data Latih (80% Komentar Unik — Sebelum Balancing)** | **2.967** pasangan | Berasal dari **1.894 komentar unik** ($80,02\%$ dari $2.367$ komentar unik) |
| **10. Pembagian Data Uji (20% Komentar Unik — Alami / Tanpa Balancing)** | **766** pasangan | Berasal dari **473 komentar unik** ($19,98\%$ dari $2.367$ komentar unik) |
| **11. Penyeimbangan Data Latih (*Random Oversampling* Hanya pada Data Latih)** | **3.834** pasangan | `1.278` Positif + `1.278` Negatif + `1.278` Netral (seluruh `1.894` komentar unik latih tetap dipertahankan) |

---

## 3. Bukti Tidak Ada Komentar Tumpang Tindih Antara Data Latih dan Data Uji

Pemisahan 80:20 dilakukan pada tingkat `Comment_ID` unik **sebelum** pembentukan sampel penyeimbangan, sehingga seluruh pasangan `(komentar, aspek)` yang berasal dari satu komentar dijamin masuk ke kelompok yang sama. Hasil verifikasi himpunan irisan (*set intersection*) pada [`Klasifikasi/bukti_tidak_tumpang_tindih.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Klasifikasi/bukti_tidak_tumpang_tindih.csv):

- **Irisan `Comment_ID` ($\text{Train} \cap \text{Test}$)**: **`0`** komentar (`1.894` komentar unik latih + `473` komentar unik uji = `2.367` komentar unik beraspek)
- **Irisan `Comment Text` asli ($\text{Train} \cap \text{Test}$)**: **`0`** komentar
- **Irisan `normalized_text` ($\text{Train} \cap \text{Test}$)**: **`0`** komentar
- **Irisan `stemmed_text` ($\text{Train} \cap \text{Test}$)**: **`0`** komentar
- **Status Validasi**: **TERBUKTI 0 TUMPANG TINDIH (*ZERO OVERLAP*)**

---

## 4. Tabel Distribusi 4 Kategori Aspek (Data Sebelum Penyeimbangan)

Sumber berkas: [`Klasifikasi/aspect_sentiment_summary.xlsx`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Klasifikasi/aspect_sentiment_summary.xlsx) dan [`Visualisasi/distribusi_4_aspek_sebelum_balancing.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Visualisasi/distribusi_4_aspek_sebelum_balancing.csv).

| Kategori Aspek | Positif | Negatif | Netral | Total | Persentase Aspek |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Perlindungan Anak** | 269 | 294 | 432 | **995** | 26,65% |
| **Peran Orang Tua** | 83 | 97 | 114 | **294** | 7,88% |
| **Peran Pemerintah/Komdigi** | 183 | 531 | 464 | **1.178** | 31,56% |
| **Efektivitas Kebijakan** | 233 | 467 | 566 | **1.266** | 33,91% |
| **Total Keseluruhan** | **768** (20,57%) | **1.389** (37,21%) | **1.576** (42,22%) | **3.733** | **100,00%** |

---

## 5. Tabel Eksperimen Penyeimbangan Data Latih (Ditampilkan Terpisah)

Sumber berkas: [`Klasifikasi/tabel_eksperimen_balancing_latih.xlsx`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Klasifikasi/tabel_eksperimen_balancing_latih.xlsx).

### 5.1 Ringkasan Jumlah Data Sebelum dan Sesudah Penyeimbangan

| Kelompok Data | Komentar Unik | Positif | Negatif | Netral | Total Pasangan Aspek |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1. Seluruh Pasangan Komentar–Aspek (100% — Sebelum Split & Balancing)** | 2.367 | 768 | 1.389 | 1.576 | **3.733** |
| **2. Data Latih (80% — Sebelum Balancing)** | 1.894 | 604 | 1.085 | 1.278 | **2.967** |
| **3. Data Latih (80% — Sesudah Balancing / Oversampling)** | 1.894 | 1.278 | 1.278 | 1.278 | **3.834** |
| **4. Data Uji (20% — Alami / Tanpa Balancing)** | 473 | 164 | 304 | 298 | **766** |

### 5.2 Rincian Data Latih (Sebelum vs Sesudah Balancing) & Data Uji per Aspek

| Kategori Aspek | Latih Sebelum (Pos / Neg / Net / Total) | Latih Sesudah Balancing (Pos / Neg / Net / Total) | Uji Alami 20% (Pos / Neg / Net / Total) |
| :--- | :---: | :---: | :---: |
| **Perlindungan Anak** | 211 / 229 / 346 (**786**) | 450 / 272 / 346 (**1.068**) | 58 / 65 / 86 (**209**) |
| **Peran Orang Tua** | 64 / 73 / 91 (**228**) | 131 / 80 / 91 (**302**) | 19 / 24 / 23 (**66**) |
| **Peran Pemerintah/Komdigi** | 147 / 420 / 372 (**939**) | 320 / 494 / 372 (**1.186**) | 36 / 111 / 92 (**239**) |
| **Efektivitas Kebijakan** | 182 / 363 / 469 (**1.014**) | 377 / 432 / 469 (**1.278**) | 51 / 104 / 97 (**252**) |
| **Total** | **604 / 1.085 / 1.278 (2.967)** | **1.278 / 1.278 / 1.278 (3.834)** | **164 / 304 / 298 (766)** |

---

## 6. Hasil Evaluasi Multinomial Naive Bayes pada Data Uji (20%)

Sumber berkas: [`Klasifikasi/evaluasi_naive_bayes_lengkap.xlsx`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Klasifikasi/evaluasi_naive_bayes_lengkap.xlsx) dan [`Visualisasi/evaluation_per_aspect.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Visualisasi/evaluation_per_aspect.csv).

### 6.1 Tabel Evaluasi Keseluruhan dan Per Kategori Aspek

| Kategori Aspek | Jumlah Data Uji ($N_{\text{uji}}$) | Prediksi Benar | Accuracy (%) | Precision Macro (%) | Recall Macro (%) | F1-Score Macro (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Perlindungan Anak** | 209 | 193 | **92,34%** | 91,65% | 92,11% | 91,83% |
| **Peran Orang Tua** | 66 | 62 | **93,94%** | 93,85% | 93,96% | 93,76% |
| **Peran Pemerintah/Komdigi** | 239 | 220 | **92,05%** | 89,05% | 91,74% | 89,92% |
| **Efektivitas Kebijakan** | 252 | 232 | **92,06%** | 90,41% | 92,38% | 91,00% |
| **Keseluruhan (Total 4 Aspek)** | **766** | **707** | **92,30%** | **91,04%** | **92,35%** | **91,47%** |

### 6.2 Confusion Matrix Keseluruhan ($N_{\text{uji}} = 766$)

| Label Acuan (InSet) \ Prediksi (NB) | Prediksi Positif | Prediksi Negatif | Prediksi Netral | Total Aktual |
| :--- | ---: | ---: | ---: | ---: |
| **Aktual Positif** | **152** | 12 | 0 | 164 |
| **Aktual Negatif** | 20 | **284** | 0 | 304 |
| **Aktual Netral** | 18 | 9 | **271** | 298 |
| **Total Prediksi** | 190 | 305 | 271 | **766** |

### 6.3 Confusion Matrix Per Kategori Aspek

1. **Perlindungan Anak ($N_{\text{uji}} = 209$, Accuracy = 92,34%)**
   - Aktual Positif (58): Prediksi Positif = `53`, Negatif = `5`, Netral = `0`
   - Aktual Negatif (65): Prediksi Positif = `6`, Negatif = `59`, Netral = `0`
   - Aktual Netral (86): Prediksi Positif = `2`, Negatif = `3`, Netral = `81`
2. **Peran Orang Tua ($N_{\text{uji}} = 66$, Accuracy = 93,94%)**
   - Aktual Positif (19): Prediksi Positif = `18`, Negatif = `1`, Netral = `0`
   - Aktual Negatif (24): Prediksi Positif = `1`, Negatif = `23`, Netral = `0`
   - Aktual Netral (23): Prediksi Positif = `2`, Negatif = `0`, Netral = `21`
3. **Peran Pemerintah/Komdigi ($N_{\text{uji}} = 239$, Accuracy = 92,05%)**
   - Aktual Positif (36): Prediksi Positif = `33`, Negatif = `3`, Netral = `0`
   - Aktual Negatif (111): Prediksi Positif = `5`, Negatif = `106`, Netral = `0`
   - Aktual Netral (92): Prediksi Positif = `7`, Negatif = `4`, Netral = `81`
4. **Efektivitas Kebijakan ($N_{\text{uji}} = 252$, Accuracy = 92,06%)**
   - Aktual Positif (51): Prediksi Positif = `48`, Negatif = `3`, Netral = `0`
   - Aktual Negatif (104): Prediksi Positif = `8`, Negatif = `96`, Netral = `0`
   - Aktual Netral (97): Prediksi Positif = `7`, Negatif = `2`, Netral = `88`

---

## 7. Contoh Komentar Beserta Aspek, Label Acuan, dan Hasil Prediksi

Seluruh `766` hasil prediksi data uji dapat diperiksa di [`Klasifikasi/hasil_prediksi_data_uji.xlsx`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Klasifikasi/hasil_prediksi_data_uji.xlsx) dan sampel terkurasi di [`Visualisasi/contoh_prediksi_komentar_aspek.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Visualisasi/contoh_prediksi_komentar_aspek.csv). Berikut beberapa contoh pemeriksaan pasangan komentar–aspek pada data uji:

| Pair ID | Comment ID | Cuplikan Komentar Asli | Kategori Aspek | Kata Kunci Terdeteksi | Label Acuan (InSet) | Hasil Prediksi (NB) | Status |
| :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| `P1169` | `C1675` | *"Tolong please, jangan blokir YouTube, aku padahal cuman nonton pokémon... cuman ada pengaturan orang tua, sama lagi, sensor, layar kalau ada konten seksual langsung di banned akunnya please"* | **Perlindungan Anak** | `seksual, konten` | `netral` | `netral` | Sesuai |
| `P1170` | `C1675` | *(komentar multi-aspek yang sama dengan C1675 di atas)* | **Peran Orang Tua** | `orang tua` | `netral` | `netral` | Sesuai |
| `P1171` | `C1675` | *(komentar multi-aspek yang sama dengan C1675 di atas — klausa menolak pemblokiran)* | **Peran Pemerintah/Komdigi** | `banned, blokir, atur` | `negatif` | `negatif` | Sesuai |
| `P1172` | `C1675` | *(komentar multi-aspek yang sama dengan C1675 di atas — klausa penolakan blokir YouTube)* | **Efektivitas Kebijakan** | `youtube, akun` | `negatif` | `negatif` | Sesuai |
| `P1241` | `C1723` | *"Negara apaan coba ini makin lama makin hancur, Masa game ama medsos di urusan, giliran yang lain kayak, korupsi, judol dll nya di biarin... daripada ngurusin hidup orang lain, mending ngurusin hidup ibu nya sendiri deh, lagian juga gak semua anak anak memainkan game, dan medsos dengan asal asalan."* | **Perlindungan Anak** | `anak anak, anak` | `netral` | `netral` | Sesuai |
| `P1242` | `C1723` | *(komentar multi-aspek yang sama dengan C1723 di atas)* | **Peran Orang Tua** | `biarin, ibu` | `netral` | `netral` | Sesuai |
| `P1243` | `C1723` | *(komentar multi-aspek yang sama dengan C1723 di atas — klausa kritik pemerintah & judol)* | **Peran Pemerintah/Komdigi** | `pemerintah, ngurusin, perintah, korupsi, negara` | `negatif` | `negatif` | Sesuai |
| `P1244` | `C1723` | *(komentar multi-aspek yang sama dengan C1723 di atas)* | **Efektivitas Kebijakan** | `medsos, game` | `negatif` | `negatif` | Sesuai |
| `P0220` | `C0372` | *"Buat yang nggak setuju atau kesel, coba dipikir dulu... Medsos sendiri sebenarnya punya banyak manfaat, misalnya buat belajar... solusi utamanya bukan langsung membatasi, tapi mengajarkan dari awal cara menggunakan medsos dengan benar, ditambah pengawasan yang cukup..."* | **Perlindungan Anak** | `belajar, ajar, umur` | `positif` | `positif` | Sesuai |
| `P0221` | `C0372` | *(komentar multi-aspek yang sama dengan C0372 di atas)* | **Peran Orang Tua** | `pengawasan, kontrol` | `positif` | `positif` | Sesuai |
| `P0068` | `C0104` | *"percuma aku nabung buat beli hp...."* | **Perlindungan Anak** | `hp` | `negatif` | `negatif` | Sesuai |
| `P0069` | `C0104` | *"percuma aku nabung buat beli hp...."* | **Peran Orang Tua** | `beli hp` | `negatif` | `negatif` | Sesuai |
| `P0165` | `C0296` | *"bertahap bukan hoaks"* (contoh pembalikan negasi `bukan hoaks` $\rightarrow$ positif) | **Efektivitas Kebijakan** | `tahap` | `positif` | `positif` | Sesuai |
