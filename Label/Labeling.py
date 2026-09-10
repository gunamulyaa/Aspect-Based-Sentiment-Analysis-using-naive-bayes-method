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
    # Netral hanya dipakai jika benar-benar tidak ada sinyal sentimen
    # atau selisihnya sangat kecil. Jangan jadikan netral sebagai default
    # ketika ada kata positif/negatif yang jelas.
    # ========================================================

    sentiment_score = positive_score - negative_score
    total_hits = positive_hits + negative_hits
    total_weight = positive_score + negative_score

    if total_hits == 0 and total_weight == 0:
        sentiment = "netral"
    elif total_hits > 0 and abs(sentiment_score) <= 0.1:
        sentiment = "netral"
    elif sentiment_score > 0:
        sentiment = "positif"
    elif sentiment_score < 0:
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

# Over-sampling kelas minoritas agar positif dan negatif mendekati jumlah netral,
# tanpa menghilangkan kelas netral sebagai kelas dominan yang sudah lebih banyak.
positive_count = (df["sentiment"] == "positif").sum()
negative_count = (df["sentiment"] == "negatif").sum()
neutral_count = (df["sentiment"] == "netral").sum()

target_count = neutral_count

balanced_frames = []

neutral_df = df[df["sentiment"] == "netral"]
if len(neutral_df) > target_count:
    neutral_df = neutral_df.sample(n=target_count, random_state=42)
balanced_frames.append(neutral_df)

for label in ["positif", "negatif"]:
    class_df = df[df["sentiment"] == label]
    if len(class_df) < target_count:
        class_df = class_df.sample(n=target_count, replace=True, random_state=42)
    else:
        class_df = class_df.sample(n=target_count, random_state=42)
    balanced_frames.append(class_df)

balanced_df = pd.concat(balanced_frames, ignore_index=True)

# Acak ulang agar urutan data tidak terstruktur berdasarkan kelas.
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

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