# -*- coding: utf-8 -*-
"""
Tahap Preprocessing & Deteksi 4 Kategori Aspek
Acuan Revisi Bab I-III:
1. Mempertahankan kata negasi (tidak, bukan, jangan, belum, tanpa, kurang, tak)
   pada teks terfilter, teks hasil stemming, dan teks aspek.
2. Memperbaiki normalisasi angka ulang (anak2 -> anak, bukan 'anak dua').
3. Menghapus baris kosong/hanya simbol serta menduplikasi komentar identik
   agar setiap komentar unik memiliki Comment_ID unik (C0001..C3996) dan
   tidak bocor antar data latih dan data uji.
4. Memetakan kata kunci ke 4 kategori aspek sesuai Bab I-III:
   - Perlindungan Anak
   - Peran Orang Tua
   - Peran Pemerintah/Komdigi
   - Efektivitas Kebijakan
"""

import os
import re
import sys
from pathlib import Path
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# =========================================================
# CONFIGURATION & IMPORTS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "Label"))

from Dictionary import ASPECT_CATEGORIES, ASPECT_KEYWORDS, negation_words  # noqa: E402

INPUT_FILE = BASE_DIR / "Hasil_Merge" / "combined_dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "Preprocessing"
OUTPUT_FILE = OUTPUT_DIR / "dataset_preprocessed.xlsx"
ALL_ROWS_OUTPUT_FILE = OUTPUT_DIR / "dataset_preprocessed_all_rows.xlsx"
STAGE_SUMMARY_CSV = OUTPUT_DIR / "preprocessing_stage_summary.csv"

TEXT_COLUMN = "Comment Text"

# =========================================================
# STOPWORDS BAHASA INDONESIA (MEMPERTAHANKAN NEGASI)
# =========================================================
# PENTING (Acuan Revisi Poin 3):
# Kata negasi seperti "tidak", "bukan", "jangan", "belum", "tanpa", "kurang", "tak"
# TIDAK dimasukkan ke dalam STOPWORDS agar tetap dipertahankan untuk pelabelan dan klasifikasi.
# Kata penghubung kontras ("tapi", "tetapi", "namun", "padahal", "meski", "walaupun", "sedangkan")
# juga dipertahankan pada tahap klausa aspek agar pemisahan konteks antar aspek akurat.

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "atau",
    "pada", "adalah", "akan", "ada", "juga",
    # Kata negasi ("tidak", "bukan", "jangan", "belum", "tanpa", "kurang", "tak") SENGAJA TIDAK DIHAPUS
    "ya", "iya", "saya", "aku", "kamu", "dia", "mereka", "kita", "kami",
    "nya", "ku", "mu", "pun", "lah", "kah",
    "sebagai", "dalam", "oleh", "karena", "agar",
    "lebih", "sudah", "bisa", "banyak", "hanya", "jadi", "kalau", "jika",
    "seperti", "masih", "saat", "ketika", "sampai", "buat", "dapat",
    "mau", "ingin", "harus", "boleh",
    "apa", "siapa", "mana", "kenapa", "mengapa", "bagaimana",
    # Filler khas media sosial
    "kok", "sih", "dong", "deh", "nih", "kan", "aja", "saja", "punya",
}

# =========================================================
# NORMALIZATION DICTIONARY
# =========================================================

NORMALIZATION_DICT = {
    # NEGASI (dipertahankan penuh untuk pelabelan InSet & klasifikasi TF-IDF)
    "gak": "tidak",
    "ga": "tidak",
    "gk": "tidak",
    "gaa": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "enggak": "tidak",
    "engga": "tidak",
    "kagak": "tidak",
    "kaga": "tidak",
    "ngak": "tidak",
    "nda": "tidak",
    "ndak": "tidak",
    "tdk": "tidak",
    "tak": "tidak",
    "bkn": "bukan",
    "jgn": "jangan",
    "blm": "belum",
    "belom": "belum",
    "gabisa": "tidak bisa",
    "gakbisa": "tidak bisa",
    "gboleh": "tidak boleh",
    "gbkl": "tidak bakal",
    "gabakal": "tidak bakal",
    "gabakalan": "tidak bakal",
    "gabakaln": "tidak bakal",
    "gkada": "tidak ada",
    "gaada": "tidak ada",
    "gada": "tidak ada",

    # KATA UMUM & SLANG MEDIA SOSIAL
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
    "bgt": "banget",
    "bgtt": "banget",
    "bnget": "banget",
    "bngt": "banget",
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "lu": "kamu",
    "loe": "kamu",
    "lo": "kamu",
    "org": "orang",
    "orng": "orang",
    "ank": "anak",
    "ortu": "orang tua",
    "ortunya": "orang tua",
    "orangtua": "orang tua",
    "thn": "tahun",
    "th": "tahun",
    "tgl": "tanggal",
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
    "tetep": "tetap",
    "smua": "semua",
    "bs": "bisa",
    "lg": "lagi",
    "pls": "tolong",
    "please": "tolong",
    "plis": "tolong",
    "pliss": "tolong",
    "plisss": "tolong",
    "kyk": "kayak",
    "kek": "kayak",
    "kayaknya": "kayak",
    "urusin": "urus",
    "ngurusin": "urus",
    "ngatur": "atur",
    "batasin": "batas",
    "dibatasin": "batas",
    "dibatesin": "batas",
    "biarin": "biar",
    "dibiarin": "biar",
    "keblokir": "blokir",
    "diblokir": "blokir",
    "keblok": "blokir",
    "diblok": "blokir",
    "setujuu": "setuju",
    "alhamdulilah": "alhamdulillah",
    "terimakasih": "terima kasih",
    "gpp": "tidak apa apa",
    "gapapa": "tidak apa apa",
    "gkpp": "tidak apa apa",
    "gakpapa": "tidak apa apa",
    "fyp": "fyp",
    "tt": "tiktok",
    "min": "admin",
}

# =========================================================
# EMOJI NORMALIZATION
# =========================================================

EMOJI_DICT = {
    "😭": "emoji_sedih",
    "😢": "emoji_sedih",
    "😥": "emoji_sedih",
    "😔": "emoji_sedih",
    "😂": "emoji_tawa",
    "🤣": "emoji_tawa",
    "😆": "emoji_tawa",
    "😍": "emoji_suka",
    "🥰": "emoji_suka",
    "❤": "emoji_suka",
    "❤️": "emoji_suka",
    "♥️": "emoji_suka",
    "😡": "emoji_marah",
    "🤬": "emoji_marah",
    "😱": "emoji_kaget",
    "😮": "emoji_kaget",
    "😲": "emoji_kaget",
    "👍": "emoji_setuju",
    "👏": "emoji_setuju",
    "🙏": "emoji_mohon",
    "🔥": "emoji_keren",
}

# =========================================================
# STEMMER DENGAN CACHE TOKEN
# =========================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()
_STEM_CACHE: dict[str, str] = {}

# Kata yang tidak boleh diubah ke homograf ambigu oleh Sastrawi
_PROTECTED_STEMS = {
    "tidak": "tidak",
    "bukan": "bukan",
    "jangan": "jangan",
    "belum": "belum",
    "tanpa": "tanpa",
    "kurang": "kurang",
    "tak": "tidak",
    "setuju": "tuju",
    "tuju": "tuju",
}


def stem_token_cached(token: str) -> str:
    if token in _PROTECTED_STEMS:
        return _PROTECTED_STEMS[token]
    if token not in _STEM_CACHE:
        _STEM_CACHE[token] = stemmer.stem(token)
    return _STEM_CACHE[token]


# =========================================================
# FUNGSI TAHAPAN PREPROCESSING
# =========================================================

def normalize_emoji(text: str) -> str:
    for emoji, replacement in EMOJI_DICT.items():
        text = text.replace(emoji, f" {replacement} ")
    return text


def normalize_repeated_characters(text: str) -> str:
    # Ubah 3+ karakter berulang menjadi 1 karakter (misal: bangettttt -> banget)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text


def normalize_trailing_double_chars(word: str) -> str:
    # Rapikan huruf ganda di akhir kata slang (misal: akuu -> aku, setujuu -> setuju, lohh -> loh)
    if len(word) > 3 and word[-1] == word[-2] and word not in {"maaf", "taat", "saat", "manfaat"}:
        return word[:-1]
    return word


def normalize_repeated_words(text: str) -> str:
    words = text.split()
    result = []
    previous = None
    for word in words:
        if word != previous:
            result.append(word)
        previous = word
    return " ".join(result)


def normalize_digit_repetition(text: str) -> str:
    """
    Perbaikan bug revisi:
    anak2 -> anak (bukan 'anak dua' agar tidak memunculkan kata 'dua' palsu)
    orang2 -> orang
    teman2 -> teman
    """
    text = re.sub(r"\b([a-zA-Z]+)2\b", r"\1", text)
    return text


def normalize_slang(text: str) -> str:
    words = text.split()
    normalized_words = []
    for word in words:
        clean_word = word.strip(".,!?;:()[]{}\"'`")
        clean_word = normalize_trailing_double_chars(clean_word)
        if clean_word in NORMALIZATION_DICT:
            normalized_words.append(NORMALIZATION_DICT[clean_word])
        else:
            normalized_words.append(clean_word)
    return " ".join(normalized_words)


def cleansing(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"<.*?>", " ", text)
    text = normalize_emoji(text)
    text = normalize_digit_repetition(text)
    text = normalize_repeated_characters(text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_comment(text) -> str:
    text = cleansing(text)
    text = normalize_slang(text)
    text = normalize_repeated_words(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def case_folding(text: str) -> str:
    return str(text).lower()


def tokenization(text: str) -> list[str]:
    if not text:
        return []
    return text.split()


def filtering(tokens: list[str]) -> list[str]:
    filtered = []
    for token in tokens:
        token = token.strip().lower()
        if token and (token in negation_words or (token not in STOPWORDS and len(token) > 1)):
            filtered.append(token)
    return filtered


def stemming(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    out = []
    for tok in tokens:
        tok_clean = normalize_trailing_double_chars(tok)
        mapped = NORMALIZATION_DICT.get(tok_clean, tok_clean)
        for sub_tok in mapped.split():
            st = stem_token_cached(sub_tok)
            for part in st.split():
                if not out or out[-1] != part:
                    out.append(part)
    return out


# =========================================================
# DETEKSI 4 KATEGORI ASPEK & EKSTRAKSI KATA FITUR
# =========================================================

# PENTING: Kata negasi TIDAK dimasukkan ke ASPECT_EXCLUSION agar tetap utuh!
ASPECT_EXCLUSION = {
    "emoji_tawa",
    "emoji_kaget",
    "emoji_mohon",
    "wkwk",
    "wkwkwk",
    "haha",
    "hehe",
    "bang",
    "kak",
    "sis",
    "bro",
    "admin",
}

ASPECT_PATTERNS = {
    aspect: re.compile(
        r"\b(?:"
        + "|".join(re.escape(kw) for kw in sorted(set(kws), key=len, reverse=True))
        + r")\b"
    )
    for aspect, kws in ASPECT_KEYWORDS.items()
}


def detect_four_aspects(raw_text: str, norm_text: str, stem_text: str) -> list[str]:
    """Mendeteksi kategori aspek (0 s.d. 4 aspek) pada komentar."""
    combined = f"{str(raw_text).lower()} {norm_text} {stem_text}"
    detected = []
    for aspect in ASPECT_CATEGORIES:
        if ASPECT_PATTERNS[aspect].search(combined):
            detected.append(aspect)
    return detected


def aspect_extraction(tokens: list[str], min_length: int = 3) -> list[str]:
    """Mengekstrak token fitur dengan tetap mempertahankan kata negasi."""
    aspects = []
    for token in tokens:
        token = token.strip().lower()
        if token in negation_words:
            aspects.append(token)
            continue
        if (
            (token.isalpha() or "_" in token)
            and len(token) >= min_length
            and token not in STOPWORDS
            and token not in ASPECT_EXCLUSION
        ):
            aspects.append(token)
    return aspects


def main():
    print("========================================")
    print("TAHAP 2: PREPROCESSING & DETEKSI 4 ASPEK")
    print("========================================")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Berkas input tidak ditemukan: {INPUT_FILE}")

    df_raw = pd.read_excel(INPUT_FILE)
    n_raw_total = len(df_raw)
    print(f"Jumlah data awal (combined_dataset) : {n_raw_total}")

    if TEXT_COLUMN not in df_raw.columns:
        raise ValueError(f"Kolom '{TEXT_COLUMN}' tidak ditemukan pada {INPUT_FILE}")

    n_missing_text = int(df_raw[TEXT_COLUMN].isna().sum())

    print("\n[1/7] Cleansing...")
    df_raw["cleaned_text"] = df_raw[TEXT_COLUMN].apply(cleansing)

    print("[2/7] Normalisasi slang, typo, angka ulang, dan karakter berulang...")
    df_raw["normalized_text"] = df_raw[TEXT_COLUMN].apply(normalize_comment)

    print("[3/7] Case Folding...")
    df_raw["case_folding"] = df_raw["normalized_text"].apply(case_folding)

    print("[4/7] Tokenization...")
    df_raw["tokens"] = df_raw["case_folding"].apply(tokenization)

    print("[5/7] Stopword Removal (mempertahankan kata negasi: tidak, bukan, jangan, belum, tanpa, kurang)...")
    df_raw["filtered_tokens"] = df_raw["tokens"].apply(filtering)

    print("[6/7] Stemming menggunakan Sastrawi...")
    df_raw["stemmed_tokens"] = df_raw["filtered_tokens"].apply(stemming)

    df_raw["tokenized_text"] = df_raw["tokens"].apply(lambda x: " ".join(x))
    df_raw["filtered_text"] = df_raw["filtered_tokens"].apply(lambda x: " ".join(x))
    df_raw["stemmed_text"] = df_raw["stemmed_tokens"].apply(lambda x: " ".join(x))

    # Hapus baris yang kosong setelah pembersihan & stemming (mis. hanya berisi emoji/tanda baca/angka)
    valid_mask = df_raw["stemmed_text"].fillna("").str.strip() != ""
    n_empty_after_clean = int((~valid_mask).sum()) - n_missing_text
    n_total_removed_empty = int((~valid_mask).sum())

    df_valid = df_raw.loc[valid_mask].copy().reset_index(drop=True)
    n_valid_rows = len(df_valid)

    # Deduplikasi komentar dengan teks hasil stemming yang identik agar tidak ada kebocoran latih/uji
    df_unique = df_valid.drop_duplicates(subset=["stemmed_text"], keep="first").copy().reset_index(drop=True)
    n_duplicate_removed = n_valid_rows - len(df_unique)
    n_unique_comments = len(df_unique)

    # Berikan ID unik per komentar: C0001 .. C3996
    df_unique.insert(0, "Comment_ID", [f"C{i + 1:04d}" for i in range(n_unique_comments)])

    print("[7/7] Deteksi 4 Kategori Aspek (Perlindungan Anak, Peran Orang Tua, Peran Pemerintah/Komdigi, Efektivitas Kebijakan)...")
    df_unique["aspects"] = df_unique["stemmed_tokens"].apply(aspect_extraction)
    df_unique["aspect_text"] = df_unique["aspects"].apply(lambda x: " ".join(x))
    df_unique["detected_aspects_list"] = [
        detect_four_aspects(r, n, s)
        for r, n, s in zip(
            df_unique[TEXT_COLUMN].fillna(""),
            df_unique["normalized_text"].fillna(""),
            df_unique["stemmed_text"].fillna(""),
        )
    ]
    df_unique["detected_aspects"] = df_unique["detected_aspects_list"].apply(lambda x: ", ".join(x))
    df_unique["aspect_count"] = df_unique["detected_aspects_list"].apply(len)

    n_with_aspect = int((df_unique["aspect_count"] > 0).sum())
    n_without_aspect = int((df_unique["aspect_count"] == 0).sum())
    n_total_pairs = int(df_unique["aspect_count"].sum())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Simpan dataset unik hasil preprocessing
    df_save = df_unique.drop(columns=["detected_aspects_list"]).copy()
    df_save.to_excel(OUTPUT_FILE, index=False)

    # Simpan ringkasan jumlah data per tahap
    stage_rows = [
        {"Tahap": "1. Data Mentah Gabungan (combined_dataset.xlsx)", "Jumlah_Data": n_raw_total, "Keterangan": "Gabungan 7 berkas Excel dataset komentar TikTok"},
        {"Tahap": "2. Baris Kosong / Tanpa Teks / Hanya Simbol Dihapus", "Jumlah_Data": n_total_removed_empty, "Keterangan": f"{n_missing_text} baris kosong awal + {n_empty_after_clean} baris hanya simbol/angka/emoji netral"},
        {"Tahap": "3. Data Komentar Valid Setelah Pembersihan", "Jumlah_Data": n_valid_rows, "Keterangan": "Baris dengan teks valid setelah cleansing & stemming"},
        {"Tahap": "4. Duplikasi Komentar Identik Dihapus", "Jumlah_Data": n_duplicate_removed, "Keterangan": "Komentar berulang/identik dihapus agar tidak bocor antara latih & uji"},
        {"Tahap": "5. Komentar Unik Bersih (dataset_preprocessed.xlsx)", "Jumlah_Data": n_unique_comments, "Keterangan": "Komentar unik ber-ID C0001 s.d. C3996"},
        {"Tahap": "6. Komentar Tanpa Aspek Terdeteksi (Dikeluarkan)", "Jumlah_Data": n_without_aspect, "Keterangan": "Tidak otomatis diberi label netral sesuai Poin 3 revisi"},
        {"Tahap": "7. Komentar Unik Memiliki >= 1 Aspek", "Jumlah_Data": n_with_aspect, "Keterangan": "Komentar yang memuat minimal 1 dari 4 kategori aspek"},
        {"Tahap": "8. Total Pasangan Komentar-Aspek Terbentuk", "Jumlah_Data": n_total_pairs, "Keterangan": "Satu komentar multi-aspek dipecah menjadi beberapa pasangan (komentar, aspek)"},
    ]
    pd.DataFrame(stage_rows).to_csv(STAGE_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n========================================")
    print("PREPROCESSING SELESAI")
    print("========================================")
    print(f"Data mentah gabungan                 : {n_raw_total}")
    print(f"Baris kosong/simbol dihapus          : {n_total_removed_empty} ({n_missing_text} NaN + {n_empty_after_clean} simbol)")
    print(f"Data valid setelah cleansing         : {n_valid_rows}")
    print(f"Duplikat komentar identik dihapus    : {n_duplicate_removed}")
    print(f"Komentar unik bersih                 : {n_unique_comments}")
    print(f"Komentar tanpa aspek (dikeluarkan)   : {n_without_aspect}")
    print(f"Komentar dengan >= 1 aspek           : {n_with_aspect}")
    print(f"Total pasangan (komentar, aspek)     : {n_total_pairs}")
    print(f"Output utama                         : {OUTPUT_FILE}")
    print(f"Ringkasan tahap                      : {STAGE_SUMMARY_CSV}")
    print("========================================")


if __name__ == "__main__":
    main()
