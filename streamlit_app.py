# -*- coding: utf-8 -*-
"""
Dashboard Streamlit Analisis Sentimen Berbasis Aspek (ABSA)
Kebijakan Pembatasan Media Sosial untuk Anak di Bawah 16 Tahun (Komdigi)
Menggunakan Kamus InSet, Pembobotan TF-IDF, dan Multinomial Naive Bayes.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
VIS_DIR = BASE_DIR / "Visualisasi"
PREP_DIR = BASE_DIR / "Preprocessing"
KLAS_DIR = BASE_DIR / "Klasifikasi"
LABEL_DIR = BASE_DIR / "Label"

st.set_page_config(
    page_title="Dashboard ABSA - Naive Bayes & Kamus InSet",
    layout="wide",
)

st.title("Aspect-Based Sentiment Analysis (ABSA) — Metode Naive Bayes")
st.caption(
    "Analisis Sentimen Publik terhadap Kebijakan Pembatasan Media Sosial untuk Anak di Bawah 16 Tahun "
    "(4 Kategori Aspek: Perlindungan Anak, Peran Orang Tua, Peran Pemerintah/Komdigi, dan Efektivitas Kebijakan)"
)

summary_path = VIS_DIR / "summary_visualisasi.csv"
if not summary_path.exists():
    st.warning("Berkas hasil visualisasi belum tersedia. Jalankan `python run_pipeline.py` terlebih dahulu.")
    st.stop()

summary = pd.read_csv(summary_path)
row = summary.iloc[0]

# =========================================================
# RINGKASAN METRIK UTAMA
# =========================================================
st.subheader("1. Ringkasan Data & Pembagian Latih/Uji (80:20)")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Komentar Unik Beraspek", f"{int(row.get('total_unique_comments', 2370)):,}")
c2.metric("Total Pasangan Komentar–Aspek", f"{int(row['total_dataset']):,}")
c3.metric("Positif (Sebelum Balancing)", f"{int(row['total_positif']):,}")
c4.metric("Negatif (Sebelum Balancing)", f"{int(row['total_negatif']):,}")
c5.metric("Netral (Sebelum Balancing)", f"{int(row['total_netral']):,}")
c6.metric("Overlap Komentar Latih–Uji", f"{int(row.get('overlap_komentar', 0))}")

c7, c8, c9 = st.columns(3)
c7.metric(
    "Data Latih 80% (Sebelum Balancing)",
    f"{int(row['train_set']):,} pasangan ({int(row.get('train_comments', 1896)):,} komentar)",
)
c8.metric(
    "Data Latih 80% (Sesudah Balancing)",
    f"{int(row.get('train_set_balanced', 3735)):,} pasangan",
)
c9.metric(
    "Data Uji 20% (Alami / Tanpa Balancing)",
    f"{int(row['test_set']):,} pasangan ({int(row.get('test_comments', 474)):,} komentar)",
)

# =========================================================
# TABS DASHBOARD
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Distribusi 4 Aspek & Balancing",
    "🎯 Evaluasi Naive Bayes & Confusion Matrix",
    "☁️ Word Cloud Sentimen & 4 Aspek",
    "🔍 Pemeriksaan Contoh Komentar & Prediksi",
    "📋 Rekapitulasi Tahap & Bukti 0 Overlap",
])

with tab1:
    st.markdown("### Tabel Distribusi 4 Kategori Aspek (Data Sebelum Penyeimbangan)")
    dist_4_path = VIS_DIR / "distribusi_4_aspek_sebelum_balancing.csv"
    if dist_4_path.exists():
        dist_4_df = pd.read_csv(dist_4_path)
        st.dataframe(dist_4_df, use_container_width=True, hide_index=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        p1 = VIS_DIR / "bar_chart_distribusi_4_aspek.png"
        if p1.exists():
            st.image(str(p1), use_container_width=True)
    with col_g2:
        p2 = VIS_DIR / "bar_chart_sentimen.png"
        if p2.exists():
            st.image(str(p2), use_container_width=True)

    st.markdown("### Tabel Eksperimen Penyeimbangan Data Latih (Ditampilkan Terpisah)")
    st.info(
        "Sesuai rancangan metodologi Bab III, penyeimbangan kelas (oversampling) **hanya dilakukan pada data latih (80%)** "
        "setelah pemisahan data latih dan data uji. Data uji (20%) dipertahankan sesuai distribusi alaminya."
    )
    bal_csv = VIS_DIR / "tabel_eksperimen_balancing.csv"
    if bal_csv.exists():
        bal_df = pd.read_csv(bal_csv)
        st.dataframe(bal_df, use_container_width=True, hide_index=True)

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        p3 = VIS_DIR / "bar_chart_balancing.png"
        if p3.exists():
            st.image(str(p3), use_container_width=True)
    with col_g4:
        p4 = VIS_DIR / "bar_chart_split_80_20.png"
        if p4.exists():
            st.image(str(p4), use_container_width=True)

with tab2:
    st.markdown("### Hasil Evaluasi Multinomial Naive Bayes (Keseluruhan & Per Aspek)")
    eval_asp_csv = VIS_DIR / "evaluation_per_aspect.csv"
    if eval_asp_csv.exists():
        eval_asp_df = pd.read_csv(eval_asp_csv)
        st.dataframe(eval_asp_df, use_container_width=True, hide_index=True)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        pe1 = VIS_DIR / "bar_chart_evaluasi.png"
        if pe1.exists():
            st.image(str(pe1), use_container_width=True)
    with col_e2:
        pe2 = VIS_DIR / "bar_chart_evaluasi_per_aspek.png"
        if pe2.exists():
            st.image(str(pe2), use_container_width=True)

    st.markdown("### Confusion Matrix (Keseluruhan & Per Kategori Aspek)")
    col_cm1, col_cm2 = st.columns([1, 1.35])
    with col_cm1:
        pcm1 = VIS_DIR / "confusion_matrix_keseluruhan.png"
        if pcm1.exists():
            st.image(str(pcm1), use_container_width=True)
        cm_ov_csv = VIS_DIR / "confusion_matrix_overall.csv"
        if cm_ov_csv.exists():
            st.dataframe(pd.read_csv(cm_ov_csv), use_container_width=True, hide_index=True)
    with col_cm2:
        pcm2 = VIS_DIR / "confusion_matrix_per_aspek.png"
        if pcm2.exists():
            st.image(str(pcm2), use_container_width=True)

with tab3:
    st.markdown("### Word Cloud Berdasarkan Kelas Sentimen")
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.image(str(VIS_DIR / "wordcloud_positif.png"), caption="Sentimen Positif", use_container_width=True)
    with wc2:
        st.image(str(VIS_DIR / "wordcloud_negatif.png"), caption="Sentimen Negatif", use_container_width=True)
    with wc3:
        st.image(str(VIS_DIR / "wordcloud_netral.png"), caption="Sentimen Netral", use_container_width=True)

    st.markdown("### Word Cloud Berdasarkan 4 Kategori Aspek")
    wc_asp_img = VIS_DIR / "wordcloud_4_aspek.png"
    if wc_asp_img.exists():
        st.image(str(wc_asp_img), use_container_width=True)

with tab4:
    st.markdown("### Pemeriksaan Contoh Komentar, Aspek, Label Acuan (InSet), dan Hasil Prediksi (Naive Bayes)")
    sample_csv = VIS_DIR / "contoh_prediksi_komentar_aspek.csv"
    full_pred_xlsx = KLAS_DIR / "hasil_prediksi_data_uji.xlsx"

    if sample_csv.exists():
        st.markdown("#### Sampel Representatif (Termasuk Komentar Multi-Aspek)")
        sample_df = pd.read_csv(sample_csv)
        st.dataframe(sample_df, use_container_width=True, hide_index=True)

    if full_pred_xlsx.exists():
        st.markdown("#### Eksplorasi Seluruh Data Uji (791 Pasangan Komentar–Aspek)")
        full_pred_df = pd.read_excel(full_pred_xlsx)
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            sel_aspect = st.selectbox("Filter Kategori Aspek:", ["Semua Aspek"] + sorted(full_pred_df["aspect"].unique().tolist()))
        with f_col2:
            sel_actual = st.selectbox("Filter Label Acuan (InSet):", ["Semua Label", "positif", "negatif", "netral"])
        with f_col3:
            sel_status = st.selectbox("Filter Status Prediksi:", ["Semua Status", "Sesuai", "Tidak Sesuai"])

        filtered = full_pred_df.copy()
        if sel_aspect != "Semua Aspek":
            filtered = filtered[filtered["aspect"] == sel_aspect]
        if sel_actual != "Semua Label":
            filtered = filtered[filtered["sentiment"] == sel_actual]
        if sel_status != "Semua Status":
            filtered = filtered[filtered["Status_Prediksi"] == sel_status]

        display_cols = [
            "Pair_ID", "Comment_ID", "Comment Text", "aspect",
            "matched_keywords", "aspect_context", "sentiment",
            "predicted_sentiment", "Status_Prediksi",
        ]
        st.caption(f"Menampilkan {len(filtered):,} dari {len(full_pred_df):,} pasangan komentar–aspek pada data uji.")
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

with tab5:
    st.markdown("### Catatan Jumlah Data pada Setiap Tahap Pipeline")
    stage_csv = PREP_DIR / "preprocessing_stage_summary.csv"
    if stage_csv.exists():
        st.dataframe(pd.read_csv(stage_csv), use_container_width=True, hide_index=True)

    st.markdown("### Bukti Tidak Ada Komentar Tumpang Tindih Antara Data Latih dan Data Uji")
    proof_csv = VIS_DIR / "bukti_tidak_tumpang_tindih.csv"
    if proof_csv.exists():
        st.dataframe(pd.read_csv(proof_csv), use_container_width=True, hide_index=True)
