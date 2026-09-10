import os
import pandas as pd
import re

# ============================================================
# IMPORT KAMUS
# ============================================================

from Dictionary import (
    positive_words,
    negative_words,
    intensifiers,
    negation_words
)


# ============================================================
# KONFIGURASI FILE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "Preprocessing", "dataset_preprocessed.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "Label")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset_labeled.xlsx")
BALANCED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset_labeled_balanced.xlsx")

# Kolom yang digunakan untuk pelabelan
TEXT_COLUMN = "aspect_text"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Dataset preprocessing tidak ditemukan: {INPUT_FILE}\n"
        "Pastikan file telah dibuat dengan script preprocessing terlebih dahulu."
    )


# ============================================================
# MEMBACA DATASET
# ============================================================

print("Membaca dataset...")

df = pd.read_excel(INPUT_FILE)

# Jaga dataset agar tetap utuh; netral tetap valid dan tidak perlu dihapus.
mask = df[TEXT_COLUMN].notna() & df[TEXT_COLUMN].astype(str).str.strip().ne("")
df = df.loc[mask].copy()

print(f"Jumlah data yang diproses: {len(df)}")


# ============================================================
# PEMBERSIHAN TEKS
# ============================================================

def clean_text(text):
    """
    Membersihkan teks sebelum proses pelabelan.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Menghapus karakter selain huruf dan spasi
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Menghapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# FUNGSI MENGAMBIL BOBOT KAMUS
# ============================================================

def get_weight(dictionary, word):
    """
    Mengambil bobot sebuah kata dari kamus.

    Mendukung beberapa kemungkinan format:
    - {"bagus": 3}
    - {"bagus": 1.5}
    """

    value = dictionary.get(word, 0)

    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return 0.0


# ============================================================
# FUNGSI PELABELAN
# ============================================================

def calculate_sentiment(text):

    text = clean_text(text)

    if not text:
        return 0, 0, 0, "netral"

    # Tokenisasi sederhana
    tokens = text.split()

    positive_score = 0.0
    negative_score = 0.0
    positive_hits = 0
    negative_hits = 0

    # Menyimpan apakah kata sebelumnya adalah intensifier
    intensifier_multiplier = 1.0

    # Menyimpan apakah ada negasi sebelum kata sentimen
    negation = False


    # ========================================================
    # MEMERIKSA SETIAP TOKEN
    # ========================================================

    for word in tokens:

        # ----------------------------------------------------
        # 1. CEK INTENSIFIER
        # ----------------------------------------------------

        if word in intensifiers:

            intensifier_weight = get_weight(
                intensifiers,
                word
            )

            # Jika bobot tidak tersedia atau 0,
            # gunakan pengali default 2
            if intensifier_weight <= 0:
                intensifier_weight = 2.0

            intensifier_multiplier = intensifier_weight

            continue


        # ----------------------------------------------------
        # 2. CEK NEGASI
        # ----------------------------------------------------

        if word in negation_words:

            negation = True

            continue


        # ----------------------------------------------------
        # 3. CEK KATA POSITIF
        # ----------------------------------------------------

        if word in positive_words:

            weight = get_weight(
                positive_words,
                word
            )

            weight *= intensifier_multiplier

            positive_hits += 1

            if negation:

                negative_score += weight

            else:

                positive_score += weight


            # Reset setelah kata sentimen
            intensifier_multiplier = 1.0
            negation = False

            continue


        # ----------------------------------------------------
        # 4. CEK KATA NEGATIF
        # ----------------------------------------------------

        if word in negative_words:

            weight = get_weight(
                negative_words,
                word
            )

            weight *= intensifier_multiplier

            negative_hits += 1

            if negation:

                positive_score += weight

            else:

                negative_score += weight


            # Reset setelah kata sentimen
            intensifier_multiplier = 1.0
            negation = False

            continue


        # ----------------------------------------------------
        # 5. RESET INTENSIFIER / NEGASI
        # ----------------------------------------------------

        # Jika terlalu jauh dari kata sentimen,
        # modifier tidak diterapkan.
        #
        # Contoh:
        # "sangat bagus" -> modifier diterapkan
        # "sangat kebijakan ini bagus" -> modifier tidak
        # diterapkan ke "bagus"

        # Tidak langsung reset di sini agar negasi
        # tetap dapat bekerja pada kata berikutnya.


    # ========================================================
    # MENENTUKAN LABEL
    # ========================================================

    if positive_hits == 0 and negative_hits == 0:
        sentiment = "netral"

    elif positive_hits + positive_score > negative_hits + negative_score:
        sentiment = "positif"

    elif negative_hits + negative_score > positive_hits + positive_score:
        sentiment = "negatif"

    else:
        sentiment = "netral"


    # ========================================================
    # RETURN
    # ========================================================

    return (
        positive_score,
        negative_score,
        positive_score - negative_score,
        sentiment
    )


# ============================================================
# PROSES PELABELAN DATASET
# ============================================================

print("\nMemulai pelabelan otomatis...")

results = df[TEXT_COLUMN].apply(calculate_sentiment)


# ============================================================
# MENAMBAHKAN HASIL KE DATASET
# ============================================================

df["positive_score"] = results.apply(
    lambda x: x[0]
)

df["negative_score"] = results.apply(
    lambda x: x[1]
)

df["sentiment_score"] = results.apply(
    lambda x: x[2]
)

df["sentiment"] = results.apply(
    lambda x: x[3]
)


# ============================================================
# BALANCING DATASET
# ============================================================

# Tetap simpan dataset utama utuh, dan buat dataset seimbang sebagai file tambahan.
positive_count = (df["sentiment"] == "positif").sum()
negative_count = (df["sentiment"] == "negatif").sum()
neutral_count = (df["sentiment"] == "netral").sum()

max_minor_class = max(positive_count, negative_count)
target_neutral = min(neutral_count, max_minor_class)

neutral_sample = df[df["sentiment"] == "netral"].sample(
    n=target_neutral,
    random_state=42
)

balanced_df = pd.concat(
    [
        df[df["sentiment"] == "positif"],
        df[df["sentiment"] == "negatif"],
        neutral_sample,
    ],
    ignore_index=True
)

# ============================================================
# MENYIMPAN DATASET
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

if os.path.exists(BALANCED_OUTPUT_FILE):
    os.remove(BALANCED_OUTPUT_FILE)

df.to_excel(
    OUTPUT_FILE,
    index=False
)

balanced_df.to_excel(
    BALANCED_OUTPUT_FILE,
    index=False
)


# ============================================================
# HASIL
# ============================================================

print("\n========================================")
print("PELABELAN SELESAI")
print("========================================")

print(f"File output : {OUTPUT_FILE}")
print(f"Jumlah data : {len(df)}")

print("\nDistribusi sentimen sebelum balancing:")

print(
    df["sentiment"]
    .value_counts()
)

print("\nDistribusi sentimen setelah balancing:")

print(
    balanced_df["sentiment"]
    .value_counts()
)

print(f"\nFile lengkap: {OUTPUT_FILE}")
print(f"File seimbang: {BALANCED_OUTPUT_FILE}")

print("\nContoh hasil:")

print(
    df[
        [
            TEXT_COLUMN,
            "positive_score",
            "negative_score",
            "sentiment_score",
            "sentiment"
        ]
    ].head(10)
)