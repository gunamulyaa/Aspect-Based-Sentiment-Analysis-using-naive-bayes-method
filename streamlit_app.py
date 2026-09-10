import os
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
VIS_DIR = BASE_DIR / "Visualisasi"

st.set_page_config(page_title="Aspect-Based Sentiment Analysis", layout="wide")

st.title("Aspect-Based Sentiment Analysis")
st.caption("Pembatasan Media Sosial untuk Anak di Bawah 16 Tahun")

summary_path = VIS_DIR / "summary_visualisasi.csv"
if summary_path.exists():
    summary = pd.read_csv(summary_path)
    row = summary.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Dataset", int(row["total_dataset"]))
    col2.metric("Positif", int(row["total_positif"]))
    col3.metric("Negatif", int(row["total_negatif"]))
    col4.metric("Netral", int(row["total_netral"]))

    col5, col6 = st.columns(2)
    col5.metric("Train (80%)", int(row["train_set"]))
    col6.metric("Test (20%)", int(row["test_set"]))

    st.subheader("Distribusi Sentimen")
    st.image(str(VIS_DIR / "bar_chart_sentimen.png"), use_container_width=True)

    st.subheader("Pembagian Dataset 80:20")
    st.image(str(VIS_DIR / "bar_chart_split_80_20.png"), use_container_width=True)

    eval_path = VIS_DIR / "bar_chart_evaluasi.png"
    if eval_path.exists():
        st.subheader("Evaluasi Model")
        st.image(str(eval_path), use_container_width=True)

    st.subheader("Word Cloud Sentimen")
    col7, col8, col9 = st.columns(3)
    with col7:
        st.image(str(VIS_DIR / "wordcloud_positif.png"), caption="Positif", use_container_width=True)
    with col8:
        st.image(str(VIS_DIR / "wordcloud_negatif.png"), caption="Negatif", use_container_width=True)
    with col9:
        st.image(str(VIS_DIR / "wordcloud_netral.png"), caption="Netral", use_container_width=True)
else:
    st.warning("File visualisasi belum dibuat. Jalankan script di folder Visualisasi terlebih dahulu.")
