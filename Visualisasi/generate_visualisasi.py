# -*- coding: utf-8 -*-
"""
Pembuatan Grafik, Confusion Matrix, dan Word Cloud Mengikuti Hasil Revisi Bab I-III:
1. Distribusi Sentimen Pasangan Komentar-Aspek (Data Sebelum Penyeimbangan)
2. Tabel & Grafik Distribusi 4 Kategori Aspek (Positif, Negatif, Netral, Total - Sebelum Penyeimbangan)
3. Perbandingan Data Latih Sebelum vs Sesudah Penyeimbangan (Eksperimen Balancing Terpisah)
4. Pembagian Data Latih (80%) dan Data Uji (20%) Tanpa Tumpang Tindih Komentar
5. Evaluasi Model Naive Bayes Keseluruhan & Per Aspek (Accuracy, Precision, Recall, F1-Score)
6. Heatmap Confusion Matrix Keseluruhan & Per Aspek
7. Word Cloud Per Sentimen (Positif, Negatif, Netral) & Per Kategori Aspek
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
LABEL_DIR = BASE_DIR / "Label"
KLASIFIKASI_DIR = BASE_DIR / "Klasifikasi"
OUTPUT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(LABEL_DIR))
from Dictionary import ASPECT_CATEGORIES  # noqa: E402

PAIRS_FILE = LABEL_DIR / "dataset_pasangan_komentar_aspek.xlsx"
if not PAIRS_FILE.exists():
    PAIRS_FILE = LABEL_DIR / "dataset_labeled.xlsx"

TRAIN_RAW_FILE = KLASIFIKASI_DIR / "data_latih_80.xlsx"
TRAIN_BAL_FILE = KLASIFIKASI_DIR / "data_latih_80_balanced.xlsx"
TEST_FILE = KLASIFIKASI_DIR / "data_uji_20.xlsx"


def get_label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["sentiment"].fillna("netral").astype(str).str.strip().str.lower().value_counts()
    return {
        "positif": int(counts.get("positif", 0)),
        "negatif": int(counts.get("negatif", 0)),
        "netral": int(counts.get("netral", 0)),
    }


def save_summary(full_df: pd.DataFrame, train_raw_df: pd.DataFrame, train_bal_df: pd.DataFrame, test_df: pd.DataFrame):
    full_counts = get_label_counts(full_df)
    tr_raw_counts = get_label_counts(train_raw_df)
    tr_bal_counts = get_label_counts(train_bal_df)
    te_counts = get_label_counts(test_df)

    summary = {
        "total_dataset": [len(full_df)],
        "total_unique_comments": [int(full_df["Comment_ID"].nunique())],
        "total_positif": [full_counts["positif"]],
        "total_negatif": [full_counts["negatif"]],
        "total_netral": [full_counts["netral"]],
        "train_comments": [int(train_raw_df["Comment_ID"].nunique())],
        "test_comments": [int(test_df["Comment_ID"].nunique())],
        "train_set": [len(train_raw_df)],
        "train_set_balanced": [len(train_bal_df)],
        "test_set": [len(test_df)],
        "train_raw_positif": [tr_raw_counts["positif"]],
        "train_raw_negatif": [tr_raw_counts["negatif"]],
        "train_raw_netral": [tr_raw_counts["netral"]],
        "train_bal_positif": [tr_bal_counts["positif"]],
        "train_bal_negatif": [tr_bal_counts["negatif"]],
        "train_bal_netral": [tr_bal_counts["netral"]],
        "test_positif": [te_counts["positif"]],
        "test_negatif": [te_counts["negatif"]],
        "test_netral": [te_counts["netral"]],
        "rasio_split": ["80:20"],
        "overlap_komentar": [0],
    }
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "summary_visualisasi.csv", index=False, encoding="utf-8-sig")
    return full_counts, tr_raw_counts, tr_bal_counts, te_counts


def make_bar_chart_sentiment(label_counts: dict[str, int], filename: str = "bar_chart_sentimen.png"):
    plt.style.use("seaborn-v0_8-whitegrid")
    labels = ["Positif", "Negatif", "Netral"]
    values = [label_counts["positif"], label_counts["negatif"], label_counts["netral"]]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars = ax.bar(labels, values, color=colors, width=0.58, edgecolor="black", linewidth=0.9)
    ax.set_title(
        f"Distribusi Sentimen Pasangan Komentar–Aspek Sebelum Balancing (N = {total:,})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Jumlah Pasangan Komentar–Aspek", fontsize=11)
    ax.set_xlabel("Kelas Sentimen (Kamus InSet)", fontsize=11)
    ax.set_ylim(0, max(values) * 1.22 if max(values) > 0 else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        pct = (value / total * 100) if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.02,
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_four_aspect_distribution_chart(full_df: pd.DataFrame, filename: str = "bar_chart_distribusi_4_aspek.png"):
    plt.style.use("seaborn-v0_8-whitegrid")
    pos_vals, neg_vals, neu_vals, tot_vals = [], [], [], []

    for asp in ASPECT_CATEGORIES:
        sub = full_df[full_df["aspect"] == asp]
        c = get_label_counts(sub)
        pos_vals.append(c["positif"])
        neg_vals.append(c["negatif"])
        neu_vals.append(c["netral"])
        tot_vals.append(len(sub))

    x = np.arange(len(ASPECT_CATEGORIES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    b_pos = ax.bar(x - width, pos_vals, width, label="Positif", color="#2ecc71", edgecolor="black", linewidth=0.8)
    b_neg = ax.bar(x, neg_vals, width, label="Negatif", color="#e74c3c", edgecolor="black", linewidth=0.8)
    b_neu = ax.bar(x + width, neu_vals, width, label="Netral", color="#3498db", edgecolor="black", linewidth=0.8)

    ax.set_title(
        "Distribusi Sentimen pada 4 Kategori Aspek (Data Sebelum Penyeimbangan)",
        fontsize=13.5,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Jumlah Pasangan Komentar–Aspek", fontsize=11)
    ax.set_xlabel("Kategori Aspek", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{asp}\n(Total: {tot})" for asp, tot in zip(ASPECT_CATEGORIES, tot_vals)], fontsize=10, fontweight="bold")
    ax.legend(frameon=True, fontsize=10.5)
    max_v = max(max(pos_vals), max(neg_vals), max(neu_vals), 1)
    ax.set_ylim(0, max_v * 1.24)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bars in (b_pos, b_neg, b_neu):
        for bar in bars:
            val = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max_v * 0.015,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_balance_comparison_chart(
    tr_raw_counts: dict[str, int],
    tr_bal_counts: dict[str, int],
    te_counts: dict[str, int],
    filename: str = "bar_chart_balancing.png",
):
    plt.style.use("seaborn-v0_8-whitegrid")
    labels = ["Positif", "Negatif", "Netral"]
    raw_vals = [tr_raw_counts["positif"], tr_raw_counts["negatif"], tr_raw_counts["netral"]]
    bal_vals = [tr_bal_counts["positif"], tr_bal_counts["negatif"], tr_bal_counts["netral"]]
    te_vals = [te_counts["positif"], te_counts["negatif"], te_counts["netral"]]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10.5, 6))
    b1 = ax.bar(
        x - width,
        raw_vals,
        width=width,
        label=f"Data Latih 80% Sebelum Balancing (N={sum(raw_vals):,})",
        color="#95a5a6",
        edgecolor="black",
        linewidth=0.8,
    )
    b2 = ax.bar(
        x,
        bal_vals,
        width=width,
        label=f"Data Latih 80% Sesudah Balancing (N={sum(bal_vals):,})",
        color="#2980b9",
        edgecolor="black",
        linewidth=0.8,
    )
    b3 = ax.bar(
        x + width,
        te_vals,
        width=width,
        label=f"Data Uji 20% Alami / Tanpa Balancing (N={sum(te_vals):,})",
        color="#f39c12",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title(
        "Eksperimen Penyeimbangan Data Latih (Balancing Hanya pada Data Latih 80%)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Jumlah Pasangan Komentar–Aspek", fontsize=11)
    ax.set_xlabel("Kelas Sentimen", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    max_value = max(max(raw_vals), max(bal_vals), max(te_vals), 1)
    ax.set_ylim(0, max_value * 1.25)

    for bars in (b1, b2, b3):
        for bar in bars:
            val = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max_value * 0.015,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_split_chart(
    train_pairs: int,
    test_pairs: int,
    train_comments: int,
    test_comments: int,
    filename: str = "bar_chart_split_80_20.png",
):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.6))

    categories = ["Data Latih (80%)", "Data Uji (20%)"]
    pair_values = [train_pairs, test_pairs]
    comment_values = [train_comments, test_comments]

    x = np.arange(len(categories))
    width = 0.34

    b_pairs = ax.bar(x - width / 2, pair_values, width, label="Pasangan Komentar–Aspek", color="#8e44ad", edgecolor="black", linewidth=0.8)
    b_comms = ax.bar(x + width / 2, comment_values, width, label="Komentar Unik (Comment_ID)", color="#16a085", edgecolor="black", linewidth=0.8)

    ax.set_title(
        "Pembagian Data Latih & Uji (80:20) Sebelum Penyeimbangan (0 Overlap Komentar)",
        fontsize=12.8,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Jumlah Data", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    max_v = max(pair_values + comment_values + [1])
    ax.set_ylim(0, max_v * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bars in (b_pairs, b_comms):
        for bar in bars:
            val = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max_v * 0.018,
                f"{val:,}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_evaluation_chart(filename: str = "bar_chart_evaluasi.png"):
    metrics_path = OUTPUT_DIR / "evaluation_metrics.csv"
    if not metrics_path.exists():
        return

    eval_df = pd.read_csv(metrics_path)
    n_test = int(eval_df["n_test"].iloc[0]) if "n_test" in eval_df.columns and not eval_df.empty else 791

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    scores = [
        float(eval_df.loc[eval_df["metric"] == m, "score"].iloc[0])
        if m in eval_df["metric"].values
        else 0.0
        for m in metrics
    ]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax.bar(metrics, [s * 100 for s in scores], color=colors, width=0.56, edgecolor="black", linewidth=0.9)
    ax.set_title(
        f"Hasil Evaluasi Keseluruhan Multinomial Naive Bayes (Data Uji N = {n_test})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Skor (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, scores):
        pct = value * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            pct + 1.8,
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_per_aspect_evaluation_chart(filename: str = "bar_chart_evaluasi_per_aspek.png"):
    per_asp_path = OUTPUT_DIR / "evaluation_per_aspect.csv"
    if not per_asp_path.exists():
        return

    df_asp = pd.read_csv(per_asp_path)
    df_asp = df_asp[df_asp["Aspek"].isin(ASPECT_CATEGORIES)].copy()
    if df_asp.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.arange(len(df_asp))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 6.2))
    m_cols = [
        ("Accuracy_Pct", "Accuracy", "#1f77b4", -1.5 * width),
        ("Precision_Pct", "Precision", "#2ca02c", -0.5 * width),
        ("Recall_Pct", "Recall", "#ff7f0e", 0.5 * width),
        ("F1_Score_Pct", "F1-Score", "#d62728", 1.5 * width),
    ]

    for col_name, label_name, color, offset in m_cols:
        vals = df_asp[col_name].tolist()
        bars = ax.bar(x + offset, vals, width, label=label_name, color=color, edgecolor="black", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 1.2,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
            )

    ax.set_title(
        "Perbandingan Evaluasi Naive Bayes Per Kategori Aspek pada Data Uji (20%)",
        fontsize=13.5,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Skor Evaluasi (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['Aspek']}\n(N Uji = {int(r['Jumlah_Data_Uji'])})" for _, r in df_asp.iterrows()],
        fontsize=10,
        fontweight="bold",
    )
    ax.set_ylim(0, 112)
    ax.legend(frameon=True, fontsize=10, ncol=4, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_confusion_matrix_overall_chart(filename: str = "confusion_matrix_keseluruhan.png"):
    cm_path = OUTPUT_DIR / "confusion_matrix_overall.csv"
    if not cm_path.exists():
        return

    cm_df = pd.read_csv(cm_path)
    matrix = cm_df[["Prediksi_Positif", "Prediksi_Negatif", "Prediksi_Netral"]].values
    labels = ["Positif", "Negatif", "Netral"]
    total = int(matrix.sum())

    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Label Prediksi (Naive Bayes)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Label Acuan (Kamus InSet)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Confusion Matrix Keseluruhan pada Data Uji (N = {total})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix[i, j])
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=13,
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_confusion_matrix_per_aspect_chart(filename: str = "confusion_matrix_per_aspek.png"):
    cm_asp_path = OUTPUT_DIR / "confusion_matrix_per_aspect.csv"
    if not cm_asp_path.exists():
        return

    cm_df = pd.read_csv(cm_asp_path)
    labels = ["Positif", "Negatif", "Netral"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for idx, aspect in enumerate(ASPECT_CATEGORIES):
        ax = axes_flat[idx]
        sub = cm_df[cm_df["Aspek"] == aspect]
        matrix = sub[["Prediksi_Positif", "Prediksi_Negatif", "Prediksi_Netral"]].values
        n_asp = int(matrix.sum())
        tp_asp = int(np.trace(matrix))
        acc_asp = (tp_asp / n_asp * 100) if n_asp else 0.0

        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_xlabel("Prediksi", fontsize=10)
        ax.set_ylabel("Acuan", fontsize=10)
        ax.set_title(
            f"{aspect}\n(N Uji = {n_asp} | Accuracy = {acc_asp:.2f}%)",
            fontsize=11.5,
            fontweight="bold",
        )

        thresh = matrix.max() / 2.0 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = int(matrix[i, j])
                ax.text(
                    j,
                    i,
                    f"{val}",
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=11.5,
                    fontweight="bold",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Confusion Matrix Per Kategori Aspek pada Data Uji (20%)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_wordcloud_for_sentiment(df: pd.DataFrame, sentiment: str, filename: str):
    sub = df[df["sentiment"] == sentiment]
    col_to_use = "aspect_context" if "aspect_context" in sub.columns else "aspect_text"
    texts = sub[col_to_use].dropna().astype(str).tolist()
    text = " ".join(item.lower() for item in texts if item.strip())

    if not text.strip():
        text = sentiment

    cmap_map = {
        "positif": "summer",
        "negatif": "autumn",
        "netral": "winter",
    }

    wc = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        colormap=cmap_map.get(sentiment, "viridis"),
        max_words=100,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"Word Cloud Sentimen {sentiment.capitalize()} (N = {len(sub):,} Pasangan Aspek)",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_wordcloud_four_aspects(df: pd.DataFrame, filename: str = "wordcloud_4_aspek.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    axes_flat = axes.flatten()
    cmaps = ["viridis", "plasma", "magma", "cividis"]

    for idx, aspect in enumerate(ASPECT_CATEGORIES):
        ax = axes_flat[idx]
        sub = df[df["aspect"] == aspect]
        col_to_use = "aspect_context" if "aspect_context" in sub.columns else "aspect_text"
        text = " ".join(sub[col_to_use].dropna().astype(str).tolist()).lower()
        if not text.strip():
            text = aspect

        wc = WordCloud(
            width=800,
            height=480,
            background_color="white",
            colormap=cmaps[idx % len(cmaps)],
            max_words=80,
            collocations=False,
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Aspek: {aspect} (N = {len(sub):,})", fontsize=12.5, fontweight="bold")

    fig.suptitle("Word Cloud Per Kategori Aspek (4 Aspek Utama)", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_human_audit_chart(filename: str = "bar_chart_evaluasi_manusia.png"):
    audit_path = OUTPUT_DIR / "evaluasi_terhadap_label_manusia.csv"
    if not audit_path.exists():
        return

    df_aud = pd.read_csv(audit_path)
    if "Cakupan" in df_aud.columns:
        df_aud = df_aud[df_aud["Cakupan"] == "Keseluruhan (4 Aspek)"].copy()
    if df_aud.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    metrics = ["Accuracy_Pct", "Precision_Pct", "Recall_Pct", "F1_Score_Pct"]
    metric_labels = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = ["#2980b9", "#e67e22"]

    for idx, (_, row) in enumerate(df_aud.iterrows()):
        vals = [float(row[m]) for m in metrics]
        offset = -width / 2 if idx == 0 else width / 2
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=row.get("Metode_Dievaluasi", row.get("Metode_yang_Dievaluasi", f"Metode {idx+1}")),
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.85,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 1.5,
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    ax.set_title(
        "Evaluasi Kamus InSet & Naive Bayes terhadap Label Acuan Manusia (N = 100 Sampel Audit)",
        fontsize=12.8,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Skor Evaluasi (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_visualizations():
    print("========================================")
    print("TAHAP 5: PEMBUATAN GRAFIK, CONFUSION MATRIX, & WORD CLOUD")
    print("========================================")

    if not PAIRS_FILE.exists():
        raise FileNotFoundError(f"Berkas pasangan komentar-aspek tidak ditemukan: {PAIRS_FILE}")

    full_df = pd.read_excel(PAIRS_FILE)
    train_raw_df = pd.read_excel(TRAIN_RAW_FILE) if TRAIN_RAW_FILE.exists() else full_df
    train_bal_df = pd.read_excel(TRAIN_BAL_FILE) if TRAIN_BAL_FILE.exists() else full_df
    test_df = pd.read_excel(TEST_FILE) if TEST_FILE.exists() else full_df

    full_counts, tr_raw_counts, tr_bal_counts, te_counts = save_summary(
        full_df, train_raw_df, train_bal_df, test_df
    )

    make_bar_chart_sentiment(full_counts, "bar_chart_sentimen.png")
    make_four_aspect_distribution_chart(full_df, "bar_chart_distribusi_4_aspek.png")
    make_balance_comparison_chart(tr_raw_counts, tr_bal_counts, te_counts, "bar_chart_balancing.png")
    make_split_chart(
        train_pairs=len(train_raw_df),
        test_pairs=len(test_df),
        train_comments=int(train_raw_df["Comment_ID"].nunique()),
        test_comments=int(test_df["Comment_ID"].nunique()),
        filename="bar_chart_split_80_20.png",
    )
    make_evaluation_chart("bar_chart_evaluasi.png")
    make_per_aspect_evaluation_chart("bar_chart_evaluasi_per_aspek.png")
    make_confusion_matrix_overall_chart("confusion_matrix_keseluruhan.png")
    make_confusion_matrix_per_aspect_chart("confusion_matrix_per_aspek.png")
    make_human_audit_chart("bar_chart_evaluasi_manusia.png")

    for sentiment in ["positif", "negatif", "netral"]:
        make_wordcloud_for_sentiment(full_df, sentiment, f"wordcloud_{sentiment}.png")

    make_wordcloud_four_aspects(full_df, "wordcloud_4_aspek.png")

    print("Seluruh grafik dan word cloud berhasil diperbarui di folder Visualisasi:")
    print(f" - Total pasangan komentar-aspek (sebelum balancing) : {len(full_df)} (dari {full_df['Comment_ID'].nunique()} komentar unik)")
    print(f" - Distribusi sebelum balancing                      : Positif={full_counts['positif']}, Negatif={full_counts['negatif']}, Netral={full_counts['netral']}")
    print(f" - Data Latih 80% (sebelum -> sesudah balancing)     : {len(train_raw_df)} -> {len(train_bal_df)}")
    print(f" - Data Uji 20% (alami / tanpa balancing)            : {len(test_df)}")
    print("========================================")


if __name__ == "__main__":
    generate_all_visualizations()
