from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


BASE_DIR = Path(__file__).resolve().parents[1]
LABEL_DIR = BASE_DIR / "Label"
DATASET_FILE = LABEL_DIR / "dataset_labeled.xlsx"
if not DATASET_FILE.exists():
    DATASET_FILE = LABEL_DIR / "dataset_labeled_balanced.xlsx"

OUTPUT_DIR = Path(__file__).resolve().parent


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_FILE}")

    df = pd.read_excel(DATASET_FILE)
    required = ["aspect_text", "sentiment"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom yang dibutuhkan tidak ada: {missing}")

    df = df[required].copy()
    df["aspect_text"] = df["aspect_text"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].fillna("netral").astype(str).str.strip().str.lower()
    df = df[df["aspect_text"].str.strip() != ""].reset_index(drop=True)
    return df


def save_summary(df):
    total_dataset = len(df)
    label_counts = (
        df["sentiment"].value_counts().reindex(["positif", "negatif", "netral"], fill_value=0)
    )

    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(shuffled) * 0.8)
    train_df = shuffled.iloc[:split_idx].copy()
    test_df = shuffled.iloc[split_idx:].copy()

    summary = {
        "total_dataset": [total_dataset],
        "total_positif": [int(label_counts.get("positif", 0))],
        "total_negatif": [int(label_counts.get("negatif", 0))],
        "total_netral": [int(label_counts.get("netral", 0))],
        "train_set": [len(train_df)],
        "test_set": [len(test_df)],
        "rasio_split": ["80:20"],
    }

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "summary_visualisasi.csv", index=False)
    return train_df, test_df, label_counts


def make_bar_chart(label_counts, title, filename):
    plt.style.use("seaborn-v0_8-whitegrid")
    labels = ["Positif", "Negatif", "Netral"]
    values = [int(label_counts.get("positif", 0)), int(label_counts.get("negatif", 0)), int(label_counts.get("netral", 0))]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars = ax.bar(labels, values, color=colors, width=0.7, edgecolor="black", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Jumlah", fontsize=11)
    ax.set_xlabel("Sentimen", fontsize=11)
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02, str(value), ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_split_chart(train_count, test_count, filename):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = ["Train (80%)", "Test (20%)"]
    values = [train_count, test_count]
    colors = ["#8e44ad", "#f39c12"]
    bars = ax.bar(labels, values, color=colors, width=0.7, edgecolor="black", linewidth=0.8)
    ax.set_title("Pembagian Dataset 80:20", fontsize=14, fontweight="bold")
    ax.set_ylabel("Jumlah Data", fontsize=11)
    ax.set_xlabel("Kelompok", fontsize=11)
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02, str(value), ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_evaluation_chart(filename="bar_chart_evaluasi.png"):
    metrics_path = OUTPUT_DIR / "evaluation_metrics.csv"
    if not metrics_path.exists():
        pd.DataFrame([
            {"metric": "Accuracy", "score": 0.0},
            {"metric": "Precision", "score": 0.0},
            {"metric": "Recall", "score": 0.0},
            {"metric": "F1-Score", "score": 0.0},
        ]).to_csv(metrics_path, index=False)

    eval_df = pd.read_csv(metrics_path)
    eval_df = eval_df[eval_df["metric"].isin(["Accuracy", "Precision", "Recall", "F1-Score"])].copy()
    eval_df["score"] = pd.to_numeric(eval_df["score"], errors="coerce").fillna(0.0)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    scores = [float(eval_df.loc[eval_df["metric"] == m, "score"].iloc[0]) if m in eval_df["metric"].values else 0.0 for m in metrics]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax.bar(metrics, [s * 100 for s in scores], color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("Hasil Evaluasi Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Skor (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, scores):
        pct = value * 100
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 2, f"{pct:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_wordcloud_for_sentiment(df, sentiment, filename):
    texts = df[df["sentiment"] == sentiment]["aspect_text"].tolist()
    text = " ".join(str(item).lower() for item in texts if str(item).strip())

    if not text.strip():
        text = f"{sentiment}"

    wc = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=100
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud Sentimen {sentiment.title()}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def generate_all_visualizations():
    df = load_dataset()
    train_df, test_df, label_counts = save_summary(df)

    make_bar_chart(label_counts, "Distribusi Sentimen", "bar_chart_sentimen.png")
    make_split_chart(len(train_df), len(test_df), "bar_chart_split_80_20.png")
    make_evaluation_chart("bar_chart_evaluasi.png")

    for sentiment in ["positif", "negatif", "netral"]:
        make_wordcloud_for_sentiment(df, sentiment, f"wordcloud_{sentiment}.png")

    print("Visualisasi berhasil dibuat pada folder Visualisasi.")
    print(f"Jumlah dataset total: {len(df)}")
    print(f"Positif: {int(label_counts.get('positif', 0))}")
    print(f"Negatif: {int(label_counts.get('negatif', 0))}")
    print(f"Netral: {int(label_counts.get('netral', 0))}")
    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")
    print("Evaluasi visualisasi: bar_chart_evaluasi.png")


if __name__ == "__main__":
    generate_all_visualizations()
