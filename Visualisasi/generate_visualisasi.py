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
    labels = ["Positif", "Negatif", "Netral"]
    values = [int(label_counts.get("positif", 0)), int(label_counts.get("negatif", 0)), int(label_counts.get("netral", 0))]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Jumlah")
    ax.set_xlabel("Sentimen")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def make_split_chart(train_count, test_count, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Train (80%)", "Test (20%)"]
    values = [train_count, test_count]
    colors = ["#8e44ad", "#f39c12"]
    ax.bar(labels, values, color=colors)
    ax.set_title("Pembagian Dataset 80:20")
    ax.set_ylabel("Jumlah Data")
    for i, value in enumerate(values):
        ax.text(i, value + 5, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
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

    for sentiment in ["positif", "negatif", "netral"]:
        make_wordcloud_for_sentiment(df, sentiment, f"wordcloud_{sentiment}.png")

    print("Visualisasi berhasil dibuat pada folder Visualisasi.")
    print(f"Jumlah dataset total: {len(df)}")
    print(f"Positif: {int(label_counts.get('positif', 0))}")
    print(f"Negatif: {int(label_counts.get('negatif', 0))}")
    print(f"Netral: {int(label_counts.get('netral', 0))}")
    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")


if __name__ == "__main__":
    generate_all_visualizations()
