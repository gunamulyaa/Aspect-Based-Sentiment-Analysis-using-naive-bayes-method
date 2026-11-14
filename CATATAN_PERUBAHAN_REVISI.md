# Catatan Perubahan & Laporan Revisi Kedua (Acuan Bab I–III)
**Topik Penelitian**: Analisis Sentimen Berbasis Aspek (*Aspect-Based Sentiment Analysis* / ABSA) terhadap Kebijakan Pembatasan Penggunaan Media Sosial bagi Anak di Bawah 16 Tahun Menggunakan Metode *Multinomial Naive Bayes* dan Kamus *InSet*.

---

## 1. Ringkasan Pemenuhan 9 Poin Revisi (Bab I–III)

| No | Poin Revisi | Perubahan yang Dilakukan | Berkas Terkait |
| :-: | :--- | :--- | :--- |
| **1** | **Gunakan Kamus InSet dari Sumber Aslinya & Dokumentasikan Modifikasi Terpisah** | Mengganti kamus buatan dalam kode (68 positif / 116 negatif) dengan berkas **InSet (Indonesian Sentiment Lexicon) asli** karya **Fajri Koto & Gemala Y. Rahmaningtyas (2017)** dari repositori resmi `https://github.com/fajri91/InSet` (`Label/inset_original_positive.tsv`: `3.609` baris / `3.607` kata unik; `Label/inset_original_negative.tsv`: `6.609` baris / `6.606` kata unik). Seluruh penetralan kata berderau/netral domain (`709` kata), resolusi kata tumpang-tindih positif–negatif (`54` kata), serta penambahan kata/frasa domain & token emoji (`56` entri) dicatat secara terpisah dari kamus asli pada `Label/inset_modifikasi_dan_tambahan.csv` dan `.xlsx` (total `819` catatan perubahan). Kamus efektif aktif mencakup `2.657` entri positif dan `5.827` entri negatif. | `Label/Dictionary.py`, `Label/inset_original_positive.tsv`, `Label/inset_original_negative.tsv`, `Label/inset_sumber_asli_metadata.csv`, `Label/inset_modifikasi_dan_tambahan.csv`, `Label/inset_modifikasi_dan_tambahan.xlsx` |
| **2** | **Pertahankan Negasi Sampai Menjadi Masukan Naive Bayes** | Memperbaiki ekstraksi fitur agar kata negasi (`tidak`, `bukan`, `jangan`, `belum`, `tanpa`, `kurang`) **dan** token majemuk negasi (`tidak_<kata_berikutnya>`, mis. `tidak_blokir`, `jangan_batas`, `bukan_konten`) selalu dibentuk dan dipertahankan hingga kolom `aspect_text` yang menjadi masukan TF-IDF dan *Multinomial Naive Bayes*, baik ketika kata yang dinegasikan ada di dalam kamus opini maupun kata konteks non-opini. Konteks `"komdigi tidak blokir fb"` kini menghasilkan fitur `"komdigi tidak tidak_blokir blokir fb"` (bukan `"blokir fb"`). | `Label/Labeling.py` (`score_clause_with_inset()`, `build_aspect_classification_text()`) |
| **3** | **Perbaiki `build_aspect_classification_text()` (Pertahankan Konteks Aspek)** | Menghapus pemangkasan yang sebelumnya hanya mengambil kata beropini atau membatasi teks netral menjadi 3 kata pertama. Fungsi `build_aspect_classification_text()` kini mempertahankan seluruh kata konteks klausa aspek yang relevan beserta kata negasi dan token majemuk negasi (`tidak_<kata>`), sehingga kosakata latih meningkat menjadi `2.975` fitur unik yang merepresentasikan konteks aspek secara utuh. | `Label/Labeling.py` |
| **4** | **Perketat Aturan Pemetaan 4 Aspek (Kontekstual)** | Memisahkan kata kunci menjadi **Kata Kunci Inti (*Core Keywords*)** dan **Kata Kunci Kontekstual (*Contextual Rules*)** pada `detect_aspects_strict()`. Kata umum seperti `"hp"`, `"beli hp"`, `"main hp"`, `"pegang hp"`, `"youtube"`, `"tiktok"`, `"roblox"`, `"game"`, `"sosmed"`, `"akun"`, `"sekolah"`, `"ibu"`, `"bapak"`, `"umur"`, atau `"usia"` **tidak lagi langsung memicu kategori aspek** kecuali disertai kata konteks kebijakan/anak/orang tua yang relevan dalam komentar tersebut. Tetap menggunakan 4 kategori aspek sesuai Bab I–III (`Perlindungan Anak`, `Peran Orang Tua`, `Peran Pemerintah/Komdigi`, `Efektivitas Kebijakan`). | `Label/Dictionary.py` (`ASPECT_CORE_KEYWORDS`, `ASPECT_CONTEXTUAL_RULES`, `detect_aspects_strict()`) |
| **5** | **Perbaiki Pemisahan Konteks Sentimen Setiap Aspek (Termasuk Tanpa `tetapi`/`namun`)** | Memperbarui `Preprocessing/preprocessing.py` (`build_clause_preserved_text()`) agar batas kalimat/tanda baca (`.`, `,`, `;`, `!`, `?`, baris baru) dipertahankan sebagai pemisah klausa `\|`, serta menambahkan `segment_clause_by_aspect_anchors()` di `Label/Labeling.py` untuk memisahkan segmen klausa berbasis posisi *anchor* kata kunci aspek ketika satu klausa memuat >1 aspek tanpa kata penghubung kontras (`tetapi`/`namun`). Diuji dan diverifikasi pada 5 skenario uji (`UJI-01` s.d. `UJI-05` di `Label/uji_pemisahan_konteks_multiaspek.csv`). Pada dataset penuh, dari `733` komentar multi-aspek, `338` komentar memiliki label sentimen yang berbeda antar-aspeknya. | `Preprocessing/preprocessing.py`, `Label/Labeling.py`, `Label/uji_pemisahan_konteks_multiaspek.csv` |
| **6** | **Pemeriksaan Manusia (*Human Inspection Audit*) pada Sampel Pasangan Komentar–Aspek** | Melakukan pemeriksaan manusia secara independen pada **100 sampel pasangan komentar–aspek** dari data uji (`64` komentar unik, mencakup 4 aspek, 3 sentimen, `52` pasangan bernegasi, dan `71` pasangan multi-aspek). Menyediakan kolom `Label_Acuan_Manusia` yang diperiksa secara independen, mencatat kesalahan pemetaan aspek (`2/100` atau ketepatan `98,00%`) dan kesalahan pelabelan kamus InSet (`34/100`), serta mengevaluasi kamus InSet (**Accuracy 66,00%**, **Macro F1 64,70%**) dan Naive Bayes (**Accuracy 60,00%**, **Macro F1 55,38%**) terhadap penilaian manusia. | `Label/human_inspection_audit.py`, `Label/sampel_pemeriksaan_manusia.xlsx`, `Visualisasi/evaluasi_terhadap_label_manusia.csv`, `Visualisasi/ringkasan_kesalahan_pelabelan_manusia.csv` |
| **7** | **Cocokkan Rumus TF-IDF dengan Bab III ($\log_{10}(N/df_t)$ Tanpa $+1$)** | Mencocokkan rumus kode dengan **Bab II Persamaan (2.2) dan Bab III Tabel 3.12** pada `Laporan Seminar Aris.docx`: $\text{TF}(t,d) = \frac{f_{t,d}}{|d|}$, $\text{IDF}(t) = \log_{10}\left(\frac{N}{df_t}\right)$ (**logaritma basis 10 tanpa penambahan konstanta $+1$**), dan $W_{d,t} = \text{TF}(t,d) \times \text{IDF}(t)$. Diverifikasi langsung terhadap contoh perhitungan manual Tabel 3.12 Bab III ($N=3, |d|=6$: untuk $df=1 \rightarrow \text{IDF}=0,477, \text{TF-IDF}=0,080$; $df=2 \rightarrow \text{IDF}=0,176, \text{TF-IDF}=0,029$; $df=3 \rightarrow \text{IDF}=0,000, \text{TF-IDF}=0,000$). | `Klasifikasi/naive_bayes_model.py`, `Klasifikasi/verifikasi_rumus_tfidf_bab3.csv`, `Visualisasi/verifikasi_rumus_tfidf_bab3.csv` |
| **8** | **Lengkapi Validasi Relevansi & Periode Komentar Sesuai Batasan Penelitian** | Menambahkan validasi periode komentar (`parse_and_validate_comment_date()`, memperbaiki korup *autofill* tahun berurutan `8-3-2027..3868` pada `Comments_1787985625.091926.xlsx` menjadi `08-03-2026`, serta mengeluarkan komentar di luar periode penelitian $> 31\text{ Mei }2026$) dan validasi relevansi topik (`check_relevance()`, mengeluarkan komentar *spam* / promosi / topik asing di luar kebijakan pembatasan medsos anak). Seluruh baris yang dikeluarkan dicatat beserta alasan spesifiknya di `Preprocessing/data_dikeluarkan_validasi.xlsx`. | `Preprocessing/preprocessing.py`, `Preprocessing/data_dikeluarkan_validasi.xlsx`, `Preprocessing/preprocessing_stage_summary.csv` |
| **9** | **Jalankan Ulang Seluruh Pipeline & Perbarui Model, Evaluasi, Tabel, Grafik, serta Dashboard** | Menjalankan ulang seluruh tahapan dari penggabungan data, *preprocessing*, pelabelan, pemodelan *Multinomial Naive Bayes*, pemeriksaan manusia, hingga pembuatan grafik dan *dashboard* Streamlit. Melaporkan hasil evaluasi terbaru secara jujur dan apa adanya (**Accuracy keseluruhan = 66,79%**, **Macro Precision = 66,20%**, **Macro Recall = 68,22%**, **Macro F1-Score = 65,97%** pada $N_{\text{uji}} = 548$ pasangan aspek). | Seluruh berkas pada `Preprocessing/`, `Label/`, `Klasifikasi/`, `Visualisasi/`, `streamlit_app.py` |

---

## 2. Catatan Jumlah Data & Validasi pada Setiap Tahap Pipeline

Sumber berkas: [`Preprocessing/preprocessing_stage_summary.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Preprocessing/preprocessing_stage_summary.csv) dan [`Preprocessing/data_dikeluarkan_validasi.xlsx`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Preprocessing/data_dikeluarkan_validasi.xlsx).

| Tahap | Jumlah Data | Keterangan & Alasan Pengeluaran Data |
| :--- | ---: | :--- |
| **1. Data Mentah Gabungan (`combined_dataset.xlsx`)** | **5.802** baris | Penggabungan 7 berkas Excel komentar TikTok pada folder `Dataset/` (`1.842` baris pada `091926.xlsx` yang mengalami *Excel autofill year corruption* `8-3-2027..3868` dipulihkan ke tanggal pengumpulan `08-03-2026`) |
| **2. Dikeluarkan: Baris Tanpa Teks Komentar (`NaN` / Kosong)** | **-26** baris | Kolom `Comment Text` kosong (`NaN`) pada berkas sumber |
| **3. Dikeluarkan: Di Luar Periode Penelitian ($> 31\text{ Mei }2026$)** | **-61** baris | Komentar bertanggal Juni–Agustus 2026 di luar periode pengumpulan Maret–Mei 2026 (Bab I & Bab III Tabel 3.6) |
| **4. Dikeluarkan: Tidak Relevan / Spam / Bahasa Asing** | **-36** baris | Komentar di luar topik kebijakan medsos anak (mis. *spam follow-back*, promosi produk, komentar warganet Malaysia tentang aturan negara lain, tokoh sepak bola/selebritas) |
| **5. Dikeluarkan: Hanya Simbol / Angka / Emoji Tanpa Kata Bermakna** | **-368** baris | Komentar hanya berisi tanda baca, angka, tawa (`wkwk`), atau karakter non-huruf yang menjadi kosong setelah *cleansing* & *filtering* |
| **6. Komentar Valid Sebelum Deduplikasi** | **5.311** baris | Komentar berbahasa Indonesia yang relevan dan berada dalam periode penelitian |
| **7. Dikeluarkan: Duplikasi Komentar Identik (*Deduplication*)** | **-1.419** baris | Menghapus duplikasi komentar identik (termasuk `424` baris duplikat antar-berkas *scraping* `091926` dan `166978`) agar tidak terjadi kebocoran teks identik antara data latih dan uji |
| **8. Komentar Unik Bersih (`dataset_preprocessed.xlsx`)** | **3.892** komentar | Setiap komentar unik diberi penanda `Comment_ID` (`C0001` s.d. `C3892`) |
| **9. Dikeluarkan: Komentar Tanpa Aspek Terdeteksi (Aturan Ketat)** | **-2.124** komentar | Dikeluarkan ke `Label/komentar_tanpa_aspek_dikeluarkan.xlsx` (**tidak** otomatis diberi label netral; termasuk komentar yang hanya menyebut `"hp"`/`"youtube"` tanpa konteks kebijakan/anak/orang tua) |
| **10. Komentar Unik Memiliki $\ge 1$ Kategori Aspek** | **1.768** komentar | Terdiri atas `1.035` komentar ber-aspek tunggal dan `733` komentar multi-aspek (`338` di antaranya memiliki sentimen berbeda antar-aspek) |
| **11. Total Pasangan Komentar–Aspek (`dataset_pasangan_komentar_aspek.xlsx`)** | **2.744** pasangan | Rata-rata $1,55$ aspek per komentar unik beraspek |
| **12. Pembagian Data Latih (80% Komentar Unik — Sebelum Balancing)** | **2.196** pasangan | Berasal dari **1.414 komentar unik** ($79,98\%$ dari $1.768$ komentar unik) |
| **13. Pembagian Data Uji (20% Komentar Unik — Alami / Tanpa Balancing)** | **548** pasangan | Berasal dari **354 komentar unik** ($20,02\%$ dari $1.768$ komentar unik) |
| **14. Penyeimbangan Data Latih (*Random Oversampling* Hanya Data Latih)** | **2.805** pasangan | `935` Positif + `935` Negatif + `935` Netral (seluruh `1.414` komentar unik latih tetap dipertahankan) |

---

## 3. Dokumentasi Sumber Asli Kamus InSet & Catatan Modifikasi Terpisah (Revisi Poin 1)

1. **Sumber Resmi Kamus InSet**:
   - **Penulis & Publikasi**: Fajri Koto & Gemala Y. Rahmaningtyas (2017), *"InSet Lexicon: Evaluation of a Word List for Indonesian Sentiment Analysis in Microblogs"*, IEEE International Conference on Asian Language Processing (IALP 2017).
   - **Repositori Resmi**: `https://github.com/fajri91/InSet`
   - **Berkas Sumber Asli di Proyek**:
     - [`Label/inset_original_positive.tsv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Label/inset_original_positive.tsv): **3.609 baris** (`3.607` kata unik, rentang bobot $+1$ s.d. $+5$).
     - [`Label/inset_original_negative.tsv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Label/inset_original_negative.tsv): **6.609 baris** (`6.606` kata unik, rentang bobot $-1$ s.d. $-5$).
2. **Catatan Modifikasi dan Tambahan Terpisah**:
   - Disimpan secara transparan dan terpisah dari berkas asli pada [`Label/inset_modifikasi_dan_tambahan.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Label/inset_modifikasi_dan_tambahan.csv) dan [`Label/inset_modifikasi_dan_tambahan.xlsx`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Label/inset_modifikasi_dan_tambahan.xlsx) (total **819 entri**):
     - **Penetralan Kata Netral Domain / Kata Kunci Aspek / Stopword Berderau (`709` kata)**: Kata dalam InSet asli yang secara leksikal merupakan kata fungsi atau kata kunci topik/aspek (seperti `anak`, `orang`, `tua`, `perintah`, `atur`, `batas`, `tonton`, `main`, `ajar`, `sekolah`, `akun`, `sosial`, `media`, `blokir`, `tahu`, `bisa`, `ada`, `buat`) dinetralkan agar penyebutan topik tidak otomatis dihitung sebagai sentimen positif/negatif.
     - **Resolusi Kata Ambigu Positif–Negatif (`54` kata)**: Penetapan polaritas tunggal bagi kata yang muncul sekaligus di kedua berkas `positive.tsv` dan `negative.tsv` pada InSet asli (mis. `dukung`, `setuju`, `tuju`, `manfaat`, `aman`, `lindungi` ditetapkan positif; `rugikan`, `bahaya`, `ancaman`, `kecanduan`, `salah`, `tolak` ditetapkan negatif).
     - **Tambahan Kata/Frasa Domain TikTok & Token Emoji (`56` entri)**: Slang/singkatan media sosial dan ekspresi yang belum tercakup di InSet 2017 (mis. `mantap`, `setuju banget`, `lepas tangan`, `tutup mata`, `akal akalan`, `emoji_suka`, `emoji_marah`, `emoji_sedih`).

---

## 4. Verifikasi Pelestarian Negasi, Fitur Konteks `aspect_text`, & Pemisahan Klausa Multi-Aspek (Revisi Poin 2, 3, & 5)

Sumber berkas pengujian: [`Label/uji_pemisahan_konteks_multiaspek.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Label/uji_pemisahan_konteks_multiaspek.csv).

| ID Uji | Komentar Uji | Aspek Terdeteksi | Konteks Klausa Aspek (`aspect_context`) | Fitur Masukan Naive Bayes (`aspect_text`) | Skor Bersih | Label Sentimen |
| :--- | :--- | :--- | :--- | :--- | ---: | :---: |
| **UJI-01** *(Negasi pada kata non-opini)* | *"komdigi tidak blokir fb"* | `Peran Pemerintah/Komdigi` & `Efektivitas Kebijakan` | `komdigi tidak blokir fb` | `komdigi tidak tidak_blokir blokir fb` | `0.0` | `netral` |
| **UJI-02** *(Multi-aspek dengan `tetapi`)* | *"Saya sangat setuju kebijakan pemerintah melindungi mental anak, tetapi implementasi sistem verifikasi umur masih mudah dibobol pakai vpn"* | `Perlindungan Anak`<br>`Peran Pemerintah/Komdigi`<br>`Efektivitas Kebijakan` | `bijak perintah lindung mental anak...`<br>`sangat tuju bijak perintah lindung`<br>`implementasi sistem verifikasi umur mudah bobol pakai vpn` | `bijak perintah lindung mental anak...`<br>`sangat tuju bijak perintah lindung`<br>`implementasi sistem verifikasi umur mudah bobol pakai vpn` | `+7.0`<br>`+11.5`<br>`-1.0` | `positif`<br>`positif`<br>`negatif` |
| **UJI-03** *(Multi-aspek TANPA `tetapi`/`namun` — tanda baca titik)* | *"Aturan pemerintah bagus sekali melindungi mental anak kecil. Sayangnya orang tua di rumah malah lalai dan lepas tangan."* | `Perlindungan Anak`<br>`Peran Pemerintah/Komdigi`<br>`Peran Orang Tua` | `bagus sekali lindung mental anak kecil`<br>`atur perintah bagus`<br>`sayang orang tua rumah malah lalai lepas tangan` | `bagus sekali lindung mental anak kecil`<br>`atur perintah bagus`<br>`sayang orang tua rumah malah lalai lepas tangan lepas_tangan` | `+12.0`<br>`+4.0`<br>`-8.0` | `positif`<br>`positif`<br>`negatif` |
| **UJI-04** *(Multi-aspek TANPA `tetapi`/`namun` & tanpa tanda baca)* | *"langkah komdigi mantap menjaga kesehatan mental anak orang tua justru lalai lepas tangan"* | `Perlindungan Anak`<br>`Peran Pemerintah/Komdigi`<br>`Peran Orang Tua` | `jaga sehat mental anak orang`<br>`langkah komdigi mantap`<br>`anak orang tua justru lalai lepas tangan` | `jaga sehat mental anak orang`<br>`langkah komdigi mantap`<br>`anak orang tua justru lalai lepas tangan lepas_tangan` | `+4.0`<br>`+5.0`<br>`-8.0` | `positif`<br>`positif`<br>`negatif` |
| **UJI-05** *(Multi-aspek TANPA `tetapi`/`namun` — opini berbeda)* | *"Pengawasan orang tua sangat bagus mendampingi anak belajar. Pemblokiran akun sosmed percuma karena tanggal lahir bisa dipalsukan."* | `Perlindungan Anak`<br>`Peran Orang Tua`<br>`Efektivitas Kebijakan` | `sangat bagus damping anak ajar`<br>`awas orang tua sangat bagus damping anak`<br>`blokir akun sosmed percuma tanggal lahir palsu` | `sangat bagus damping anak ajar`<br>`awas orang tua sangat bagus damping anak`<br>`blokir akun sosmed percuma tanggal lahir palsu` | `+6.0`<br>`+6.0`<br>`-8.0` | `positif`<br>`positif`<br>`negatif` |

---

## 5. Verifikasi Kesesuaian Rumus TF-IDF dengan Tabel 3.12 Bab III (Revisi Poin 7)

Sesuai **Bab II Persamaan (2.2)** dan **Bab III Tabel 3.12** pada laporan:
$$\text{TF}(t, d) = \frac{f_{t,d}}{|d|}, \qquad \text{IDF}(t) = \log_{10}\left(\frac{N}{df_t}\right), \qquad W_{d,t} = \text{TF}(t, d) \times \text{IDF}(t)$$

Hasil pengujian fungsi perhitungan TF-IDF pada [`Klasifikasi/verifikasi_rumus_tfidf_bab3.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Klasifikasi/verifikasi_rumus_tfidf_bab3.csv) terhadap contoh perhitungan manual pada Tabel 3.12 Bab III ($N = 3$ dokumen, panjang dokumen $|d| = 6$ kata sehingga $\text{TF} = 1/6 \approx 0,167$):

| Jumlah Dokumen ($N$) | Document Frequency ($df_t$) | $\text{TF}(t,d) = 1/6$ | Rumus $\text{IDF}(t)$ | $\text{IDF}$ Kode | $\text{IDF}$ Tabel 3.12 Bab III | $\text{TF-IDF}$ Kode | $\text{TF-IDF}$ Tabel 3.12 Bab III | Status Verifikasi |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :---: |
| `3` | `1` | `0.167` | $\log_{10}(3/1)$ | **0.477** | **0.477** | **0.080** | **0.080** | **SESUAI PERSIS** |
| `3` | `2` | `0.167` | $\log_{10}(3/2)$ | **0.176** | **0.176** | **0.029** | **0.029** | **SESUAI PERSIS** |
| `3` | `3` | `0.167` | $\log_{10}(3/3)$ | **0.000** | **0.000** | **0.000** | **0.000** | **SESUAI PERSIS** |

---

## 6. Bukti Tidak Ada Komentar Tumpang Tindih Antara Data Latih dan Data Uji

Hasil verifikasi himpunan irisan (*set intersection*) pada [`Klasifikasi/bukti_tidak_tumpang_tindih.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Klasifikasi/bukti_tidak_tumpang_tindih.csv):
- **Irisan `Comment_ID` ($\text{Train} \cap \text{Test}$)**: **`0`** komentar (`1.414` komentar unik latih + `354` komentar unik uji = `1.768` komentar unik beraspek)
- **Irisan `Comment Text` asli ($\text{Train} \cap \text{Test}$)**: **`0`** komentar
- **Irisan `normalized_text` ($\text{Train} \cap \text{Test}$)**: **`0`** komentar
- **Irisan `stemmed_text` ($\text{Train} \cap \text{Test}$)**: **`0`** komentar
- **Status Validasi**: **TERBUKTI 0 TUMPANG TINDIH (*ZERO OVERLAP*)**

---

## 7. Tabel Distribusi 4 Kategori Aspek (Data Sebelum Penyeimbangan)

Sumber berkas: [`Klasifikasi/aspect_sentiment_summary.xlsx`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Klasifikasi/aspect_sentiment_summary.xlsx) dan [`Visualisasi/distribusi_4_aspek_sebelum_balancing.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Visualisasi/distribusi_4_aspek_sebelum_balancing.csv).

| Kategori Aspek | Positif | Negatif | Netral | Total | Persentase Aspek |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Perlindungan Anak** | 200 | 220 | 269 | **689** | 25,11% |
| **Peran Orang Tua** | 38 | 30 | 87 | **155** | 5,65% |
| **Peran Pemerintah/Komdigi** | 142 | 496 | 374 | **1.012** | 36,88% |
| **Efektivitas Kebijakan** | 163 | 292 | 433 | **888** | 32,36% |
| **Total Keseluruhan** | **543** (19,79%) | **1.038** (37,83%) | **1.163** (42,38%) | **2.744** | **100,00%** |

---

## 8. Tabel Eksperimen Penyeimbangan Data Latih (Ditampilkan Terpisah)

Sumber berkas: [`Klasifikasi/tabel_eksperimen_balancing_latih.xlsx`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Klasifikasi/tabel_eksperimen_balancing_latih.xlsx).

### 8.1 Ringkasan Jumlah Data Sebelum dan Sesudah Penyeimbangan

| Kelompok Data | Komentar Unik | Positif | Negatif | Netral | Total Pasangan Aspek |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1. Seluruh Pasangan Komentar–Aspek (100% — Sebelum Split & Balancing)** | 1.768 | 543 | 1.038 | 1.163 | **2.744** |
| **2. Data Latih (80% — Sebelum Balancing)** | 1.414 | 433 | 828 | 935 | **2.196** |
| **3. Data Latih (80% — Sesudah Balancing / Oversampling)** | 1.414 | 935 | 935 | 935 | **2.805** |
| **4. Data Uji (20% — Alami / Tanpa Balancing)** | 354 | 110 | 210 | 228 | **548** |

### 8.2 Rincian Data Latih (Sebelum vs Sesudah Balancing) & Data Uji per Aspek

| Kategori Aspek | Latih Sebelum (Pos / Neg / Net / Total) | Latih Sesudah Balancing (Pos / Neg / Net / Total) | Uji Alami 20% (Pos / Neg / Net / Total) |
| :--- | :---: | :---: | :---: |
| **Perlindungan Anak** | 162 / 174 / 209 (**545**) | 344 / 198 / 209 (**751**) | 38 / 46 / 60 (**144**) |
| **Peran Orang Tua** | 26 / 27 / 72 (**125**) | 62 / 31 / 72 (**165**) | 12 / 3 / 15 (**30**) |
| **Peran Pemerintah/Komdigi** | 117 / 392 / 296 (**805**) | 236 / 444 / 296 (**976**) | 25 / 104 / 78 (**207**) |
| **Efektivitas Kebijakan** | 128 / 235 / 358 (**721**) | 293 / 262 / 358 (**913**) | 35 / 57 / 75 (**167**) |
| **Total** | **433 / 828 / 935 (2.196)** | **935 / 935 / 935 (2.805)** | **110 / 210 / 228 (548)** |

---

## 9. Hasil Evaluasi Multinomial Naive Bayes pada Data Uji Alami (20%)

Sumber berkas: [`Klasifikasi/evaluasi_naive_bayes_lengkap.xlsx`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Klasifikasi/evaluasi_naive_bayes_lengkap.xlsx) dan [`Visualisasi/evaluation_per_aspect.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Visualisasi/evaluation_per_aspect.csv).

### 9.1 Tabel Evaluasi Keseluruhan dan Per Kategori Aspek (Terhadap Label Acuan InSet)

| Kategori Aspek | Jumlah Data Uji ($N_{\text{uji}}$) | Prediksi Benar | Accuracy (%) | Precision Macro (%) | Recall Macro (%) | F1-Score Macro (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Perlindungan Anak** | 144 | 83 | **57,64%** | 61,76% | 59,13% | 56,81% |
| **Peran Orang Tua** | 30 | 19 | **63,33%** | 48,11% | 47,22% | 47,59% |
| **Peran Pemerintah/Komdigi** | 207 | 149 | **71,98%** | 67,40% | 70,42% | 67,04% |
| **Efektivitas Kebijakan** | 167 | 115 | **68,86%** | 67,89% | 69,37% | 68,32% |
| **Keseluruhan (Total 4 Aspek)** | **548** | **366** | **66,79%** | **66,20%** | **68,22%** | **65,97%** |

### 9.2 Confusion Matrix Keseluruhan ($N_{\text{uji}} = 548$)

| Label Acuan (InSet) \ Prediksi (NB) | Prediksi Positif | Prediksi Negatif | Prediksi Netral | Total Aktual |
| :--- | ---: | ---: | ---: | ---: |
| **Aktual Positif** | **83** | 10 | 17 | 110 |
| **Aktual Negatif** | 46 | **135** | 29 | 210 |
| **Aktual Netral** | 42 | 38 | **148** | 228 |
| **Total Prediksi** | 171 | 183 | 194 | **548** |

---

## 10. Hasil Pemeriksaan Manusia (*Human Inspection Audit*) pada 100 Sampel Pasangan Komentar–Aspek (Revisi Poin 6)

Sumber berkas: [`Label/sampel_pemeriksaan_manusia.xlsx`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Label/sampel_pemeriksaan_manusia.xlsx), [`Visualisasi/evaluasi_terhadap_label_manusia.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Visualisasi/evaluasi_terhadap_label_manusia.csv), dan [`Visualisasi/ringkasan_kesalahan_pelabelan_manusia.csv`](file:///c:/Users/gunam/Downloads/TESTING/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Aspect-Based-Sentiment-Analysis-using-naive-bayes-method-main/Visualisasi/ringkasan_kesalahan_pelabelan_manusia.csv).

### 10.1 Cakupan Sampel Audit Manusia ($N = 100$ Pasangan Komentar–Aspek)
- Berasal dari **64 komentar unik** pada data uji, mencakup **71 pasangan multi-aspek** dan **52 pasangan yang memuat kata negasi**.
- Distribusi per aspek: `Perlindungan Anak` (`45` pasangan), `Peran Orang Tua` (`12` pasangan), `Peran Pemerintah/Komdigi` (`25` pasangan), `Efektivitas Kebijakan` (`18` pasangan).
- **Ketepatan Pemetaan Aspek terhadap Penilaian Manusia**: **98/100 (98,00%)** tepat; hanya ditemukan **2 kasus ambiguitas pemetaan aspek** (`P0025` pada frasa *"anak 16 tahun disuruh main hp"* yang memicu `Peran Orang Tua`, serta `P1601` pada frasa *"pengawasan dilapangan"* yang merujuk pada pengawasan pemerintah tetapi terpetakan ke `Peran Orang Tua`).

### 10.2 Tabel Evaluasi Kamus InSet & Naive Bayes terhadap Label Acuan Manusia ($N = 100$)

| Metode yang Dievaluasi terhadap Label Acuan Manusia | Cakupan | Jumlah Sampel ($N$) | Sesuai Manusia | Accuracy (%) | Precision Macro (%) | Recall Macro (%) | F1-Score Macro (%) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pelabelan Kamus InSet vs Label Acuan Manusia** | **Keseluruhan (4 Aspek)** | **100** | **66** | **66,00%** | **64,42%** | **67,61%** | **64,70%** |
| **Prediksi Naive Bayes vs Label Acuan Manusia** | **Keseluruhan (4 Aspek)** | **100** | **60** | **60,00%** | **56,70%** | **57,41%** | **55,38%** |
| Pelabelan Kamus InSet vs Label Acuan Manusia | Perlindungan Anak | 45 | 32 | 71,11% | 46,67% | 46,55% | 46,21% |
| Prediksi Naive Bayes vs Label Acuan Manusia | Perlindungan Anak | 45 | 31 | 68,89% | 44,95% | 51,67% | 45,31% |
| Pelabelan Kamus InSet vs Label Acuan Manusia | Peran Orang Tua | 12 | 4 | 33,33% | 33,33% | 44,44% | 37,78% |
| Prediksi Naive Bayes vs Label Acuan Manusia | Peran Orang Tua | 12 | 2 | 16,67% | 11,11% | 11,11% | 11,11% |
| Pelabelan Kamus InSet vs Label Acuan Manusia | Peran Pemerintah/Komdigi | 25 | 19 | 76,00% | 74,89% | 76,21% | 72,68% |
| Prediksi Naive Bayes vs Label Acuan Manusia | Peran Pemerintah/Komdigi | 25 | 19 | 76,00% | 72,96% | 75,91% | 73,47% |
| Pelabelan Kamus InSet vs Label Acuan Manusia | Efektivitas Kebijakan | 18 | 11 | 61,11% | 66,67% | 73,33% | 61,78% |
| Prediksi Naive Bayes vs Label Acuan Manusia | Efektivitas Kebijakan | 18 | 8 | 44,44% | 48,81% | 47,78% | 43,17% |

### 10.3 Temuan Utama Penyebab Kesalahan Pelabelan Kamus InSet pada Pemeriksaan Manusia
Dari 34 ketidakcocokan antara label kamus InSet dan penilaian manusia pada sampel audit, penyebab utamanya adalah keterbatasan alami pendekatan berbasis kamus (*lexicon-based*):
1. **Dominasi Kata Bahaya Topik pada Kalimat Dukungan Perlindungan Anak** (mis. `P1190`, `P0266`, `P1480`): Komentar yang mendukung pembatasan medsos demi melindungi anak dari *bullying*, kecanduan, atau konten negatif memuat banyak kata berkonotasi negatif (`bullying`, `bahaya`, `takut`, `candu`, `goblok`) sehingga jumlah skor InSet menjadi negatif meskipun maksud penulis mendukung kebijakan (+). Menariknya, **Multinomial Naive Bayes berhasil memprediksi `positif` dengan tepat** pada `P1190`, `P0266`, dan `P1480` karena mempelajari pola konteks kalimat dukungan.
2. **Sarkasme dan Ironi** (mis. `P1833` / `C2558`: *"Ah elah judol gk di ban? Wah sangat bagus!"*): Kamus InSet memberi skor positif karena kata `"sangat bagus"`, padahal maknanya sindiran negatif terhadap pemerintah.
3. **Elipsis Kata Kerja pada Kalimat Permohonan/Negasi** (mis. `P1555` / `C2202`: *"YouTube jangan lah"*): Tanpa kata `"blokir"` eksplisit setelah `"jangan"`, kamus InSet memberi skor `0` (`netral`), sedangkan manusia dan Naive Bayes mengenali konteks penolakan (`negatif`).
4. **Umpatan Tersensor & Variasi Ejaan Tidak Baku** (mis. `P0948` / `C1716`: *"ortu tol\*l... bod\*h"*, `P1516` / `C2165`: *"Setujuh"* yang ter-stem menjadi angka `"tujuh"` bukan `"tuju"`).
