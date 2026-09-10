import pandas as pd
import re
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "Dataset",
    "combined_dataset.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Preprocessing"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "dataset_preprocessed.xlsx"
)

TEXT_COLUMN = "Comment Text"


# =========================================================
# STOPWORDS BAHASA INDONESIA
# =========================================================
#
# PENTING:
# Kata negasi seperti:
# tidak, bukan, jangan
# TIDAK dimasukkan ke stopword karena penting
# untuk analisis sentimen.
# =========================================================

STOPWORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "ini",
    "itu",
    "untuk",
    "dengan",
    "atau",
    "pada",
    "adalah",
    "akan",
    "ada",
    "juga",

    # "tidak" sengaja TIDAK dihapus
    # "bukan" sengaja TIDAK dihapus
    # "jangan" sengaja TIDAK dihapus

    "ya",
    "iya",
    "saya",
    "aku",
    "kamu",
    "dia",
    "mereka",
    "kita",
    "kami",

    "nya",
    "ku",
    "mu",
    "pun",
    "lah",
    "kah",

    "sebagai",
    "dalam",
    "oleh",
    "karena",
    "agar",

    "lebih",
    "sudah",
    "belum",
    "bisa",
    "banyak",
    "sangat",
    "hanya",
    "jadi",
    "kalau",
    "jika",

    "tapi",
    "tetapi",
    "namun",
    "seperti",
    "masih",

    "saat",
    "ketika",
    "sampai",
    "buat",
    "dapat",
    "mau",
    "ingin",
    "harus",
    "boleh",

    "apa",
    "siapa",
    "mana",
    "kenapa",
    "mengapa",
    "bagaimana",

    # Filler khas media sosial
    "kok",
    "sih",
    "dong",
    "deh",
    "nih",
    "kan",
    "aja",
    "saja",
    "punya"
}


# =========================================================
# NORMALIZATION DICTIONARY
# =========================================================

NORMALIZATION_DICT = {

    # -----------------------------------------------------
    # NEGASI
    # -----------------------------------------------------

    "gak": "tidak",
    "ga": "tidak",
    "gk": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "enggak": "tidak",
    "engga": "tidak",
    "tdk": "tidak",

    "bkn": "bukan",

    # -----------------------------------------------------
    # KATA UMUM
    # -----------------------------------------------------

    "yg": "yang",
    "yng": "yang",

    "dgn": "dengan",
    "dg": "dengan",

    "dr": "dari",

    "utk": "untuk",
    "unt": "untuk",

    "krn": "karena",
    "karna": "karena",

    "klo": "kalau",
    "kalo": "kalau",
    "kl": "kalau",

    "jd": "jadi",
    "jdi": "jadi",

    "jg": "juga",
    "jga": "juga",

    "aja": "saja",
    "aj": "saja",

    "sm": "sama",
    "sma": "sama",

    "tp": "tapi",
    "tpi": "tapi",

    "trs": "terus",
    "trus": "terus",

    "udah": "sudah",
    "udh": "sudah",
    "sdh": "sudah",

    "blm": "belum",
    "belom": "belum",

    "bgt": "banget",
    "bgtt": "banget",
    "bnget": "banget",

    # -----------------------------------------------------
    # PRONOUN
    # -----------------------------------------------------

    "gw": "saya",
    "gue": "saya",
    "gua": "saya",

    "lu": "kamu",
    "loe": "kamu",
    "lo": "kamu",

    # -----------------------------------------------------
    # ORANG
    # -----------------------------------------------------

    "org": "orang",
    "orng": "orang",

    # -----------------------------------------------------
    # KATA UMUM TIKTOK / MEDIA SOSIAL
    # -----------------------------------------------------

    "emg": "memang",
    "emang": "memang",

    "bener": "benar",
    "bnr": "benar",

    "tau": "tahu",
    "taw": "tahu",

    "pake": "pakai",

    "dapet": "dapat",
    "dpt": "dapat",

    "hrs": "harus",

    "mnrt": "menurut",

    "skrng": "sekarang",
    "skrg": "sekarang",

    # -----------------------------------------------------
    # TIDAK APA-APA
    # -----------------------------------------------------

    "gpp": "tidak apa apa",
    "gapapa": "tidak apa apa",
    "gkpp": "tidak apa apa",
    "gakpapa": "tidak apa apa",

    # -----------------------------------------------------
    # INTERNET
    # -----------------------------------------------------

    "fyp": "fyp",
    "tt": "tiktok",

    # -----------------------------------------------------
    # PANGGILAN
    # -----------------------------------------------------

    "min": "admin",
}


# =========================================================
# EMOJI NORMALIZATION
# =========================================================

EMOJI_DICT = {

    # Sedih
    "😭": "emoji_sedih",
    "😢": "emoji_sedih",
    "😥": "emoji_sedih",
    "😔": "emoji_sedih",

    # Tawa
    "😂": "emoji_tawa",
    "🤣": "emoji_tawa",
    "😆": "emoji_tawa",

    # Suka
    "😍": "emoji_suka",
    "🥰": "emoji_suka",
    "❤": "emoji_suka",
    "❤️": "emoji_suka",
    "♥️": "emoji_suka",

    # Marah
    "😡": "emoji_marah",
    "🤬": "emoji_marah",

    # Kaget
    "😱": "emoji_kaget",
    "😮": "emoji_kaget",
    "😲": "emoji_kaget",

    # Setuju
    "👍": "emoji_setuju",
    "👏": "emoji_setuju",

    # Mohon
    "🙏": "emoji_mohon",

    # Keren
    "🔥": "emoji_keren"
}


# =========================================================
# STEMMER
# =========================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()


# =========================================================
# 1. NORMALIZE EMOJI
# =========================================================

def normalize_emoji(text):

    for emoji, replacement in EMOJI_DICT.items():
        text = text.replace(
            emoji,
            f" {replacement} "
        )

    return text


# =========================================================
# 2. NORMALIZE REPEATED CHARACTERS
# =========================================================

def normalize_repeated_characters(text):

    """
    Contoh:

    bangettttt -> banget
    baguuuus -> bagus
    lucuuuuu -> lucu
    kerenzzz -> keren

    Maksimal dua karakter yang sama dipertahankan.
    """

    text = re.sub(
        r"(.)\1{2,}",
        r"\1",
        text
    )

    return text


# =========================================================
# 3. NORMALIZE REPEATED WORDS
# =========================================================

def normalize_repeated_words(text):

    """
    Contoh:

    bagus bagus bagus
    ->
    bagus

    wkwk wkwk
    ->
    wkwk
    """

    words = text.split()

    result = []

    previous = None

    for word in words:

        if word != previous:
            result.append(word)

        previous = word

    return " ".join(result)


# =========================================================
# 4. NORMALIZE DIGIT REPETITION
# =========================================================

def normalize_digit_repetition(text):

    """
    Contoh:

    anak2 -> anak dua
    orang2 -> orang dua
    teman2 -> teman dua
    """

    text = re.sub(
        r"\b([a-zA-Z]+)2\b",
        r"\1 dua",
        text
    )

    return text


# =========================================================
# 5. NORMALIZE SLANG
# =========================================================

def normalize_slang(text):

    words = text.split()

    normalized_words = []

    for word in words:

        # Bersihkan tanda baca
        clean_word = word.strip(
            ".,!?;:()[]{}\"'`"
        )

        if clean_word in NORMALIZATION_DICT:

            normalized_words.append(
                NORMALIZATION_DICT[clean_word]
            )

        else:

            normalized_words.append(
                clean_word
            )

    return " ".join(normalized_words)


# =========================================================
# 6. CLEANSING
# =========================================================

def cleansing(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # -----------------------------------------------------
    # Lowercase terlebih dahulu
    # -----------------------------------------------------

    text = text.lower()

    # -----------------------------------------------------
    # URL
    # -----------------------------------------------------

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # -----------------------------------------------------
    # Mention
    # -----------------------------------------------------

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # -----------------------------------------------------
    # Hashtag
    #
    # #tiktok -> tiktok
    # -----------------------------------------------------

    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # -----------------------------------------------------
    # HTML
    # -----------------------------------------------------

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # -----------------------------------------------------
    # Emoji
    #
    # Jangan langsung dihapus
    # -----------------------------------------------------

    text = normalize_emoji(text)

    # -----------------------------------------------------
    # Digit repetition
    # -----------------------------------------------------

    text = normalize_digit_repetition(text)

    # -----------------------------------------------------
    # Repeated characters
    # -----------------------------------------------------

    text = normalize_repeated_characters(text)

    # -----------------------------------------------------
    # Hapus angka yang tersisa
    # -----------------------------------------------------

    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # -----------------------------------------------------
    # Hanya pertahankan huruf dan underscore
    # -----------------------------------------------------

    text = re.sub(
        r"[^a-zA-Z_\s]",
        " ",
        text
    )

    # -----------------------------------------------------
    # Rapikan whitespace
    # -----------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# 7. NORMALIZATION
# =========================================================

def normalize_comment(text):

    # Cleansing
    text = cleansing(text)

    # Slang normalization
    text = normalize_slang(text)

    # Repeated words
    text = normalize_repeated_words(text)

    # Rapikan whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# 8. CASE FOLDING
# =========================================================

def case_folding(text):

    return text.lower()


# =========================================================
# 9. TOKENIZATION
# =========================================================

def tokenization(text):

    if not text:
        return []

    return text.split()


# =========================================================
# 10. FILTERING
# =========================================================

def filtering(tokens):

    filtered = []

    for token in tokens:

        token = token.strip().lower()

        if (
            token
            and token not in STOPWORDS
            and len(token) > 1
        ):
            filtered.append(token)

    return filtered


# =========================================================
# 11. STEMMING
# =========================================================

def stemming(tokens):

    if not tokens:
        return []

    text = " ".join(tokens)

    stemmed_text = stemmer.stem(text)

    return stemmed_text.split()


# =========================================================
# 12. ASPECT EXTRACTION
# =========================================================

# Kata yang biasanya bukan aspek
ASPECT_EXCLUSION = {

    # Negasi
    "tidak",
    "bukan",
    "jangan",

    # Emoji
    "emoji_sedih",
    "emoji_tawa",
    "emoji_suka",
    "emoji_marah",
    "emoji_kaget",
    "emoji_setuju",
    "emoji_mohon",
    "emoji_keren",

    # Filler
    "wkwk",
    "wkwkwk",
    "haha",
    "hehe",

    # Panggilan
    "bang",
    "kak",
    "sis",
    "bro",
    "admin",

    # Kata umum
    "orang",
    "saya",
    "kamu"
}


def aspect_extraction(tokens, min_length=3):

    aspects = []

    for token in tokens:

        token = token.strip().lower()

        if (
            token.isalpha()
            and len(token) >= min_length
            and token not in STOPWORDS
            and token not in ASPECT_EXCLUSION
        ):

            aspects.append(token)

    # Hilangkan duplikasi
    aspects = list(
        dict.fromkeys(aspects)
    )

    return aspects


# =========================================================
# LOAD DATASET
# =========================================================

print("========================================")
print("MEMULAI PREPROCESSING DATASET")
print("========================================")

print("\nMembaca dataset...")

df = pd.read_excel(
    INPUT_FILE
)

print(
    f"Jumlah data awal : {len(df)}"
)

print(
    f"Jumlah kolom     : {len(df.columns)}"
)


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
# 1. CLEANSING
# =========================================================

print("\n[1/7] Cleansing...")

df["cleaned_text"] = df[
    TEXT_COLUMN
].apply(cleansing)


# =========================================================
# 2. NORMALIZATION
# =========================================================

print("[2/7] Normalisasi slang, typo, emoji, dan repeated characters...")

df["normalized_text"] = df[
    TEXT_COLUMN
].apply(normalize_comment)


# =========================================================
# 3. CASE FOLDING
# =========================================================

print("[3/7] Case Folding...")

df["case_folding"] = df[
    "normalized_text"
].apply(case_folding)


# =========================================================
# 4. TOKENIZATION
# =========================================================

print("[4/7] Tokenization...")

df["tokens"] = df[
    "case_folding"
].apply(tokenization)


# =========================================================
# 5. FILTERING
# =========================================================

print("[5/7] Stopword Removal...")

df["filtered_tokens"] = df[
    "tokens"
].apply(filtering)


# =========================================================
# 6. STEMMING
# =========================================================

print("[6/7] Stemming menggunakan Sastrawi...")

df["stemmed_tokens"] = df[
    "filtered_tokens"
].apply(stemming)


# =========================================================
# 7. ASPECT EXTRACTION
# =========================================================

print("[7/7] Aspect Extraction...")

df["aspects"] = df[
    "stemmed_tokens"
].apply(aspect_extraction)


# =========================================================
# CREATE TEXT VERSIONS
# =========================================================

df["tokenized_text"] = df[
    "tokens"
].apply(
    lambda x: " ".join(x)
)

df["filtered_text"] = df[
    "filtered_tokens"
].apply(
    lambda x: " ".join(x)
)

df["stemmed_text"] = df[
    "stemmed_tokens"
].apply(
    lambda x: " ".join(x)
)

df["aspect_text"] = df[
    "aspects"
].apply(
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

print(
    f"Data awal       : {len(df)}"
)

print(
    f"Data akhir      : {len(df)}"
)

print(
    f"Jumlah kolom    : {len(df.columns)}"
)

print(
    f"Output          : {OUTPUT_FILE}"
)

print("\nKolom preprocessing:")

print("1. cleaned_text")
print("2. normalized_text")
print("3. case_folding")
print("4. tokens")
print("5. filtered_tokens")
print("6. stemmed_tokens")
print("7. aspects")

print("\n========================================")