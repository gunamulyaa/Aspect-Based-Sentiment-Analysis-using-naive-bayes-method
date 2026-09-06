import pandas as pd
import re
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "Dataset", "combined_dataset.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "Preprocessing")
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "dataset_preprocessed.xlsx"
)

# Kolom yang berisi komentar
TEXT_COLUMN = "Comment Text"


# =========================================================
# STOPWORDS BAHASA INDONESIA
# =========================================================

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu",
    "untuk", "dengan", "atau", "pada", "adalah",
    "akan", "ada", "juga", "tidak", "ya", "iya",
    "saya", "aku", "kamu", "dia", "mereka", "kita",
    "kami", "nya", "ku", "mu", "pun", "lah", "kah",
    "sebagai", "dalam", "oleh", "karena", "agar",
    "lebih", "sudah", "belum", "bisa", "banyak",
    "sangat", "hanya", "jadi", "kalau", "jika",
    "tapi", "tetapi", "namun", "seperti", "masih",
    "saat", "ketika", "sampai", "buat", "dapat",
    "mau", "ingin", "harus", "boleh", "jangan",
    "apa", "siapa", "mana", "kenapa", "mengapa",
    "bagaimana", "kok", "sih", "dong", "deh",
    "nih", "kan", "aja", "saja", "punya"
}


# =========================================================
# STEMMER
# =========================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()


# =========================================================
# 1. CLEANSING
# =========================================================

def cleansing(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Hapus URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Hapus mention
    text = re.sub(r"@\w+", " ", text)

    # Hapus hashtag symbol tetapi pertahankan katanya
    text = re.sub(r"#", "", text)

    # Hapus HTML
    text = re.sub(r"<.*?>", " ", text)

    # Hapus angka
    text = re.sub(r"\d+", " ", text)

    # Hapus emoji dan karakter non-alfabet
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Hapus whitespace berlebihan
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# 2. CASE FOLDING
# =========================================================

def case_folding(text):
    return text.lower()


# =========================================================
# 3. TOKENIZATION
# =========================================================

def tokenization(text):
    if not text:
        return []

    return text.split()


# =========================================================
# 4. FILTERING / STOPWORD REMOVAL
# =========================================================

def filtering(tokens):
    filtered = []

    for token in tokens:
        token = token.strip()

        if (
            token
            and token not in STOPWORDS
            and len(token) > 1
        ):
            filtered.append(token)

    return filtered


# =========================================================
# 5. STEMMING
# =========================================================

def stemming(tokens):
    if not tokens:
        return []

    text = " ".join(tokens)

    stemmed_text = stemmer.stem(text)

    return stemmed_text.split()


# =========================================================
# 6. ASPECT EXTRACTION
# =========================================================

def aspect_extraction(tokens, min_length=3):
    """
    Mengambil kandidat aspek berdasarkan kata yang tersisa
    setelah filtering dan stemming.

    Pendekatan:
    - hanya mengambil kata alfabet
    - minimal 3 karakter
    - menghindari stopword
    """

    aspects = []

    for token in tokens:
        token = token.strip().lower()

        if (
            token.isalpha()
            and len(token) >= min_length
            and token not in STOPWORDS
        ):
            aspects.append(token)

    # Hilangkan duplikasi tetapi pertahankan urutan
    aspects = list(dict.fromkeys(aspects))

    return aspects


# =========================================================
# LOAD DATASET
# =========================================================

print("========================================")
print("MEMULAI PREPROCESSING DATASET")
print("========================================")

print("\nMembaca dataset...")

df = pd.read_excel(INPUT_FILE)

print(f"Jumlah data awal : {len(df)}")
print(f"Jumlah kolom     : {len(df.columns)}")


# =========================================================
# VALIDASI KOLOM
# =========================================================

if TEXT_COLUMN not in df.columns:
    print("\nKolom yang tersedia:")

    for column in df.columns:
        print(f"- {column}")

    raise ValueError(
        f"\nKolom '{TEXT_COLUMN}' tidak ditemukan."
    )


# =========================================================
# CLEANSING
# =========================================================

print("\n[1/6] Cleansing...")

df["cleaned_text"] = df[TEXT_COLUMN].apply(
    cleansing
)


# =========================================================
# CASE FOLDING
# =========================================================

print("[2/6] Case Folding...")

df["case_folding"] = df["cleaned_text"].apply(
    case_folding
)


# =========================================================
# TOKENIZATION
# =========================================================

print("[3/6] Tokenization...")

df["tokens"] = df["case_folding"].apply(
    tokenization
)


# =========================================================
# FILTERING
# =========================================================

print("[4/6] Stopword Removal...")

df["filtered_tokens"] = df["tokens"].apply(
    filtering
)


# =========================================================
# STEMMING
# =========================================================

print("[5/6] Stemming...")

df["stemmed_tokens"] = df["filtered_tokens"].apply(
    stemming
)


# =========================================================
# ASPECT EXTRACTION
# =========================================================

print("[6/6] Aspect Extraction...")

df["aspects"] = df["stemmed_tokens"].apply(
    aspect_extraction
)


# =========================================================
# CREATE TEXT VERSIONS
# =========================================================

df["tokenized_text"] = df["tokens"].apply(
    lambda x: " ".join(x)
)

df["filtered_text"] = df["filtered_tokens"].apply(
    lambda x: " ".join(x)
)

df["stemmed_text"] = df["stemmed_tokens"].apply(
    lambda x: " ".join(x)
)

df["aspect_text"] = df["aspects"].apply(
    lambda x: ", ".join(x)
)


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# SAVE DATASET
# =========================================================

print("\nMenyimpan hasil preprocessing...")

df.to_excel(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n========================================")
print("PREPROCESSING SELESAI")
print("========================================")

print(f"Data awal       : {len(df)}")
print(f"Data akhir      : {len(df)}")
print(f"Jumlah kolom    : {len(df.columns)}")
print(f"Output          : {OUTPUT_FILE}")

print("\nKolom preprocessing:")

print("1. cleaned_text")
print("2. case_folding")
print("3. tokens")
print("4. filtered_tokens")
print("5. stemmed_tokens")
print("6. aspects")

print("\n========================================")