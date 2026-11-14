# -*- coding: utf-8 -*-
"""
Tahap Preprocessing, Validasi Periode & Relevansi Komentar, serta Deteksi 4 Kategori Aspek
Acuan Revisi Bab I-III:
1. Validasi periode komentar dan relevansi sesuai batasan penelitian (Bab I Batasan Masalah
   & Bab III Tabel 3.6):
   - Mencatat dan mengeluarkan baris kosong/NaN, baris duplikat antarsumber scrape,
     baris di luar periode penelitian (setelah 31 Mei 2026: Juni–Agustus 2026),
     baris hanya simbol/angka/emoji tanpa kata, komentar tidak relevan / di luar topik /
     berbahasa asing (mis. "ronaldo kehilangan 90% fansnya", "gua yg di malay", "akun ryu kintaro",
     spam follow-back "fb"), serta komentar duplikat identik.
   - Memperbaiki anomali autofill tanggal Excel pada berkas `Comments_1787985625.091926.xlsx`
     (baris 425–2266 yang ter-drag menjadi tahun 2027–3868 pada tanggal 8-3-2026) dan mencatatnya.
   - Menyimpan rekapitulasi jumlah data yang dikeluarkan beserta alasan rincinya pada
     `Preprocessing/preprocessing_stage_summary.csv` dan `Preprocessing/data_dikeluarkan_validasi.xlsx`.
2. Mempertahankan kata negasi (tidak, bukan, jangan, belum, tanpa, kurang, tak)
   pada teks terfilter, teks hasil stemming, klausa aspek, dan teks aspek.
3. Menyimpan teks ber-tanda-baca klausa (`clause_text`) agar pemisahan konteks sentimen
   per aspek di `Label/Labeling.py` dapat memisahkan klausa baik dengan konjungsi kontras
   maupun dengan tanda baca (`.`, `,`, `;`, `!`, `?`, `\n`) tanpa kata penghubung "tetapi"/"namun".
4. Memetakan kata kunci ke 4 kategori aspek dengan aturan konteks yang diperketat (Poin 4 revisi):
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

from Dictionary import (  # noqa: E402
    ASPECT_CATEGORIES,
    ASPECT_KEYWORDS,
    detect_aspects_strict,
    negation_words,
)

INPUT_FILE = BASE_DIR / "Hasil_Merge" / "combined_dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "Preprocessing"
OUTPUT_FILE = OUTPUT_DIR / "dataset_preprocessed.xlsx"
ALL_ROWS_OUTPUT_FILE = OUTPUT_DIR / "dataset_preprocessed_all_rows.xlsx"
EXCLUDED_VALIDATION_FILE = OUTPUT_DIR / "data_dikeluarkan_validasi.xlsx"
STAGE_SUMMARY_CSV = OUTPUT_DIR / "preprocessing_stage_summary.csv"

TEXT_COLUMN = "Comment Text"

# =========================================================
# STOPWORDS BAHASA INDONESIA (MEMPERTAHANKAN NEGASI)
# =========================================================
# PENTING (Acuan Revisi Poin 2 & 3):
# Kata negasi ("tidak", "bukan", "jangan", "belum", "tanpa", "kurang", "tak")
# TIDAK dimasukkan ke dalam STOPWORDS agar tetap dipertahankan sampai masukan Naive Bayes.

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "atau",
    "pada", "adalah", "akan", "ada", "juga",
    "ya", "iya", "saya", "aku", "kamu", "dia", "mereka", "kita", "kami",
    "nya", "ku", "mu", "pun", "lah", "kah",
    "sebagai", "dalam", "oleh", "karena", "agar",
    "lebih", "sudah", "bisa", "banyak", "hanya", "jadi", "kalau", "jika",
    "seperti", "masih", "saat", "ketika", "sampai", "buat", "dapat",
    "mau", "ingin", "harus", "boleh",
    "apa", "siapa", "mana", "kenapa", "mengapa", "bagaimana",
    "kok", "sih", "dong", "deh", "nih", "kan", "aja", "saja", "punya",
}

# =========================================================
# NORMALIZATION DICTIONARY
# =========================================================

NORMALIZATION_DICT = {
    # NEGASI (dipertahankan penuh untuk pelabelan InSet & klasifikasi TF-IDF Naive Bayes)
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
    "kgk": "tidak",
    "ngak": "tidak",
    "nda": "tidak",
    "ndak": "tidak",
    "tdk": "tidak",
    "tak": "tidak",
    "ora": "tidak",
    "bkn": "bukan",
    "jgn": "jangan",
    "jan": "jangan",
    "blm": "belum",
    "belom": "belum",
    "gabisa": "tidak bisa",
    "gakbisa": "tidak bisa",
    "gbs": "tidak bisa",
    "gboleh": "tidak boleh",
    "gaboleh": "tidak boleh",
    "gamau": "tidak mau",
    "gakmau": "tidak mau",
    "gausah": "tidak usah",
    "gausa": "tidak usah",
    "gbkl": "tidak bakal",
    "gabakal": "tidak bakal",
    "gabakalan": "tidak bakal",
    "gabakaln": "tidak bakal",
    "gkada": "tidak ada",
    "gaada": "tidak ada",
    "gada": "tidak ada",
    "gapernah": "tidak pernah",

    # KATA UMUM & SLANG MEDIA SOSIAL
    "yg": "yang",
    "yng": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "dr": "dari",
    "dri": "dari",
    "utk": "untuk",
    "unt": "untuk",
    "krn": "karena",
    "karna": "karena",
    "klo": "kalau",
    "kalo": "kalau",
    "kl": "kalau",
    "klw": "kalau",
    "jd": "jadi",
    "jdi": "jadi",
    "jg": "juga",
    "jga": "juga",
    "aja": "saja",
    "aj": "saja",
    "sm": "sama",
    "sma": "sama",
    "ama": "sama",
    "tp": "tapi",
    "tpi": "tapi",
    "trs": "terus",
    "trus": "terus",
    "udah": "sudah",
    "udh": "sudah",
    "sdh": "sudah",
    "dah": "sudah",
    "dh": "sudah",
    "bgt": "banget",
    "bgtt": "banget",
    "bnget": "banget",
    "bngt": "banget",
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "sy": "saya",
    "ak": "aku",
    "akk": "aku",
    "lu": "kamu",
    "loe": "kamu",
    "lo": "kamu",
    "km": "kamu",
    "org": "orang",
    "orng": "orang",
    "ank": "anak",
    "ortu": "orang tua",
    "ortunya": "orang tua",
    "orangtua": "orang tua",
    "thn": "tahun",
    "th": "tahun",
    "taun": "tahun",
    "tgl": "tanggal",
    "emg": "memang",
    "emang": "memang",
    "bener": "benar",
    "bnr": "benar",
    "tau": "tahu",
    "taw": "tahu",
    "pake": "pakai",
    "pke": "pakai",
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
    "lgi": "lagi",
    "pls": "tolong",
    "please": "tolong",
    "plis": "tolong",
    "pliss": "tolong",
    "plisss": "tolong",
    "tlg": "tolong",
    "kyk": "kayak",
    "kek": "kayak",
    "kayaknya": "kayak",
    "urusin": "urus",
    "ngurusin": "urus",
    "ngurus": "urus",
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
    "blok": "blokir",
    "block": "blokir",
    "ban": "blokir",
    "banned": "blokir",
    "band": "blokir",
    "setujuu": "setuju",
    "stuju": "setuju",
    "stj": "setuju",
    "se7": "setuju",
    "alhamdulilah": "alhamdulillah",
    "terimakasih": "terima kasih",
    "makasih": "terima kasih",
    "trimakasih": "terima kasih",
    "mksh": "terima kasih",
    "gpp": "tidak apa apa",
    "gapapa": "tidak apa apa",
    "gkpp": "tidak apa apa",
    "gakpapa": "tidak apa apa",
    "bgs": "bagus",
    "bljr": "belajar",
    "tt": "tiktok",
    "yutup": "youtube",
    "ytb": "youtube",
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
# VALIDASI PERIODE & RELEVANSI KOMENTAR (POIN 8 REVISI)
# =========================================================
# Sesuai Bab I (Batasan Masalah) & Bab III (Tabel 3.6):
# 1. Periode penelitian mencakup pengumuman hingga pemberlakuan kebijakan pembatasan
#    media sosial anak di bawah 16 tahun (Maret–31 Mei 2026). Komentar dengan tanggal
#    setelah 31 Mei 2026 (Juni–Agustus 2026) dikeluarkan karena berada di luar periode.
# 2. Komentar tidak relevan dengan topik kebijakan / berbahasa asing / spam follow-back
#    (seperti dicontohkan di Tabel 3.6 Bab III: "ronaldo kehilangan 90% fansnya",
#    "gua yg di malay", "akun ryu kintaro", dll.) dikeluarkan dan dicatat alasannya.

RESEARCH_START_DATE = pd.Timestamp("2026-03-01")
RESEARCH_END_DATE = pd.Timestamp("2026-05-31")

IRRELEVANT_PATTERNS = [
    # Contoh eksplisit pada Bab III Tabel 3.6 & komentar luar topik sejenis di dataset
    (re.compile(r"\bronaldo\b|\bmessi\b|\bbarca\b|\bmu\b|\beminem\b|\bkpop\b|\bbts\b", re.I), "Komentar di luar topik kebijakan (membahas tokoh/selebritas/olahraga seperti Ronaldo)"),
    (re.compile(r"\bryu\s*kintaro\b", re.I), "Komentar di luar topik kebijakan (membahas akun kreator Ryu Kintaro)"),
    (re.compile(r"\b(?:gua|gw|aku|saya)\s+(?:yg|yang)\s+di\s+malay\b|\bmalay\s+mana\s+ada\b", re.I), "Komentar pengguna di luar yurisdiksi Indonesia / bahasa Melayu non-konteks"),
    (re.compile(r"\bfollback\b|\bsaling\s+follow\b|\bmutualan\b|\bflks\b|\blfl\b|\bfff\b", re.I), "Spam permintaan follow-back (tidak membahas kebijakan)"),
    (re.compile(r"\bsebahattin\s+ali\b|\banatolia\b", re.I), "Spam salinan teks soal ujian literasi (tidak relevan dengan topik)"),
    (re.compile(r"\bbangun\s+nuklir\b|\bknalpot\s+brong\b|\bgerhana\b", re.I), "Komentar di luar topik pembatasan media sosial anak"),
    (re.compile(r"^\s*(?:first|pertama| kedua|ketiga|hadir|absen|nyimak|nitip\s+sendal|cek\s+profil|cek\s+bio|fyp)\s*[.!?\d]*\s*$", re.I), "Komentar spam kehadiran/FYP tanpa opini maupun aspek"),
]


def parse_and_validate_comment_date(row: pd.Series) -> tuple[object, str, bool]:
    """
    Memvalidasi kolom `Time` pada dataset:
    - Memperbaiki korupsi autofill Excel pada `Comments_1787985625.091926.xlsx`
      (baris 425-2266 di mana tanggal `8-3-2026` ter-drag menjadi `8-3-2027` s.d. `8-3-3868`).
    - Mengembalikan `(parsed_timestamp, keterangan_tanggal, is_within_research_period)`.
    """
    raw_time = row.get("Time", None)
    sf = str(row.get("Source_File", "")).strip()
    if pd.isna(raw_time):
        return pd.NaT, "Tanggal Kosong", False

    s = str(raw_time).strip()
    if not s or s.lower() == "nan":
        return pd.NaT, "Tanggal Kosong", False

    # Tanggal relatif pada saat scraping tanggal 28 Maret 2026 ("1d ago", "5h ago", dll.)
    if any(k in s.lower() for k in ("ago", "hari", "jam", "menit", "detik")):
        ts = pd.Timestamp("2026-03-28")
        return ts, "Tanggal Relatif Saat Scraping (28-03-2026)", True

    try:
        date_part = s.split(" ")[0]
        parts = date_part.split("-")
        if len(parts) == 3:
            if len(parts[0]) == 4:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                d, m, y = int(parts[0]), int(parts[1]), int(parts[2])

            # Koreksi autofill tahun pada Comments_1787985625.091926.xlsx (tanggal asli 8-3-2026)
            if y > 2026 and sf == "Comments_1787985625.091926.xlsx" and d == 8 and m == 3:
                ts = pd.Timestamp(year=2026, month=3, day=8)
                return ts, f"Koreksi Autofill Excel ({s} -> 08-03-2026)", True

            if y > 2026 or y < 2026:
                return pd.NaT, f"Tahun Tidak Valid ({s})", False

            ts = pd.Timestamp(year=y, month=m, day=d)
            in_period = RESEARCH_START_DATE <= ts <= RESEARCH_END_DATE
            if in_period:
                if ts < pd.Timestamp("2026-03-28"):
                    return ts, "Periode Pengumuman Kebijakan (7–27 Maret 2026)", True
                return ts, "Periode Implementasi Kebijakan (28 Maret–31 Mei 2026)", True
            else:
                return ts, f"Di Luar Periode Penelitian ({ts.strftime('%d-%m-%Y')} > 31-05-2026)", False
    except Exception:
        pass

    return pd.NaT, f"Format Tanggal Tidak Dikenali ({s})", False


def check_topic_relevance(raw_text: str) -> tuple[bool, str]:
    """
    Memeriksa apakah komentar relevan dengan topik kebijakan pembatasan medsos anak di bawah 16 tahun.
    Mengembalikan `(is_relevant, alasan_jika_tidak_relevan)`.
    """
    text_str = str(raw_text).strip()
    for pat, reason in IRRELEVANT_PATTERNS:
        if pat.search(text_str):
            return False, reason
    return True, "Relevan"


# =========================================================
# STEMMER DENGAN CACHE TOKEN
# =========================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()
_STEM_CACHE: dict[str, str] = {}

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
    "stuju": "tuju",
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
    return re.sub(r"(.)\1{2,}", r"\1", text)


def normalize_trailing_double_chars(word: str) -> str:
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
    return re.sub(r"\b([a-zA-Z]+)2\b", r"\1", text)


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


def build_clause_preserved_text(text) -> str:
    """
    Membentuk teks ternormalisasi yang mempertahankan pemisah klausa (` | `)
    dari tanda baca (`.`, `,`, `;`, `!`, `?`, `\n`) agar komentar multi-aspek
    tanpa kata hubung "tetapi"/"namun" tetap dapat dipisahkan per klausa aspeknya
    pada tahap pelabelan (Revisi Poin 5).
    """
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#(\w+)", r"\1", s)
    s = re.sub(r"\[sticker\]", " ", s)
    s = normalize_emoji(s)
    s = normalize_digit_repetition(s)
    s = normalize_repeated_characters(s)
    # Ubah tanda baca pemisah klausa/kalimat menjadi token pembatas '|'
    s = re.sub(r"[.,!?;:\n\r]+", " | ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^a-zA-Z_|\s]", " ", s)

    clauses_out = []
    for raw_cl in s.split("|"):
        norm_cl = normalize_slang(raw_cl)
        norm_cl = normalize_repeated_words(norm_cl)
        toks = [
            stem_token_cached(t)
            for t in norm_cl.split()
            if t and (t in negation_words or (t not in STOPWORDS and len(t) > 1))
        ]
        if toks:
            clauses_out.append(" ".join(toks))
    return " | ".join(clauses_out)


def cleansing(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\[sticker\]", " ", text)
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
# DETEKSI 4 KATEGORI ASPEK TERKETAT & EKSTRAKSI KATA FITUR
# =========================================================

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


def detect_four_aspects(raw_text: str, norm_text: str, stem_text: str) -> list[str]:
    """Mendeteksi kategori aspek (0 s.d. 4 aspek) dengan aturan konteks yang diperketat."""
    aspect_map = detect_aspects_strict(raw_text, norm_text, stem_text)
    return list(aspect_map.keys())


def aspect_extraction(tokens: list[str], min_length: int = 2) -> list[str]:
    """
    Mengekstrak token fitur dengan tetap mempertahankan kata negasi
    serta token senyawa negasi (`tidak_<kata>`, `jangan_<kata>`, dll.).
    """
    aspects = []
    n = len(tokens)
    for i, token in enumerate(tokens):
        token = token.strip().lower()
        if token in negation_words:
            aspects.append(token)
            if i + 1 < n:
                nxt = tokens[i + 1].strip().lower()
                if nxt and nxt not in negation_words:
                    aspects.append(f"{token}_{nxt}")
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
    print("TAHAP 2: VALIDASI PERIODE/RELEVANSI, PREPROCESSING & DETEKSI 4 ASPEK")
    print("========================================")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Berkas input tidak ditemukan: {INPUT_FILE}")

    df_raw = pd.read_excel(INPUT_FILE)
    n_raw_total = len(df_raw)
    print(f"Jumlah data awal (combined_dataset) : {n_raw_total}")

    if TEXT_COLUMN not in df_raw.columns:
        raise ValueError(f"Kolom '{TEXT_COLUMN}' tidak ditemukan pada {INPUT_FILE}")

    excluded_log_rows = []

    # 1. Validasi Baris Kosong / Tanpa Teks Komentar
    missing_mask = df_raw[TEXT_COLUMN].isna() | (df_raw[TEXT_COLUMN].astype(str).str.strip() == "")
    n_missing_text = int(missing_mask.sum())
    for _, r in df_raw[missing_mask].iterrows():
        excluded_log_rows.append({
            "Tahap_Validasi": "1. Teks Komentar Kosong (NaN)",
            "Source_File": r.get("Source_File", ""),
            "Time": r.get("Time", ""),
            "User": r.get("User @", ""),
            "Comment Text": "",
            "Alasan_Dikeluarkan": "Kolom Comment Text kosong / bernilai NaN",
        })

    df_step1 = df_raw[~missing_mask].copy().reset_index(drop=True)

    # 2. Validasi Periode Komentar (Revisi Poin 8: Batas Akhir 31 Mei 2026 & Koreksi Autofill 8-3-2026)
    date_results = df_step1.apply(parse_and_validate_comment_date, axis=1)
    df_step1["parsed_date"] = [r[0] for r in date_results]
    df_step1["date_validation_note"] = [r[1] for r in date_results]
    df_step1["in_research_period"] = [r[2] for r in date_results]

    n_autofill_fixed = int(df_step1["date_validation_note"].str.startswith("Koreksi Autofill Excel").sum())
    out_of_period_mask = ~df_step1["in_research_period"]
    n_out_of_period = int(out_of_period_mask.sum())

    for _, r in df_step1[out_of_period_mask].iterrows():
        excluded_log_rows.append({
            "Tahap_Validasi": "2. Di Luar Periode Penelitian (> 31 Mei 2026)",
            "Source_File": r.get("Source_File", ""),
            "Time": r.get("Time", ""),
            "User": r.get("User @", ""),
            "Comment Text": r.get(TEXT_COLUMN, ""),
            "Alasan_Dikeluarkan": r.get("date_validation_note", "Di luar periode penelitian"),
        })

    df_step2 = df_step1[~out_of_period_mask].copy().reset_index(drop=True)

    # 3. Validasi Relevansi Topik (Revisi Poin 8 & Tabel 3.6 Bab III)
    rel_results = df_step2[TEXT_COLUMN].apply(check_topic_relevance)
    df_step2["is_relevant_topic"] = [r[0] for r in rel_results]
    df_step2["relevance_note"] = [r[1] for r in rel_results]

    irrelevant_mask = ~df_step2["is_relevant_topic"]
    n_irrelevant_topic = int(irrelevant_mask.sum())
    for _, r in df_step2[irrelevant_mask].iterrows():
        excluded_log_rows.append({
            "Tahap_Validasi": "3. Tidak Relevan dengan Topik / Spam / Luar Konteks",
            "Source_File": r.get("Source_File", ""),
            "Time": r.get("Time", ""),
            "User": r.get("User @", ""),
            "Comment Text": r.get(TEXT_COLUMN, ""),
            "Alasan_Dikeluarkan": r.get("relevance_note", "Tidak relevan dengan topik kebijakan"),
        })

    df_step3 = df_step2[~irrelevant_mask].copy().reset_index(drop=True)

    # 4. Tahapan Preprocessing Teks
    print("\n[1/7] Cleansing...")
    df_step3["cleaned_text"] = df_step3[TEXT_COLUMN].apply(cleansing)

    print("[2/7] Normalisasi slang, typo, angka ulang, dan karakter berulang...")
    df_step3["normalized_text"] = df_step3[TEXT_COLUMN].apply(normalize_comment)

    print("[3/7] Case Folding...")
    df_step3["case_folding"] = df_step3["normalized_text"].apply(case_folding)

    print("[4/7] Tokenization...")
    df_step3["tokens"] = df_step3["case_folding"].apply(tokenization)

    print("[5/7] Stopword Removal (mempertahankan kata negasi: tidak, bukan, jangan, belum, tanpa, kurang)...")
    df_step3["filtered_tokens"] = df_step3["tokens"].apply(filtering)

    print("[6/7] Stemming menggunakan Sastrawi & Ekstraksi Klausa Ber-tanda-baca...")
    df_step3["stemmed_tokens"] = df_step3["filtered_tokens"].apply(stemming)
    df_step3["clause_text"] = df_step3[TEXT_COLUMN].apply(build_clause_preserved_text)

    df_step3["tokenized_text"] = df_step3["tokens"].apply(lambda x: " ".join(x))
    df_step3["filtered_text"] = df_step3["filtered_tokens"].apply(lambda x: " ".join(x))
    df_step3["stemmed_text"] = df_step3["stemmed_tokens"].apply(lambda x: " ".join(x))

    # 5. Hapus baris yang kosong setelah pembersihan & stemming (hanya berisi emoji netral/tanda baca/angka)
    valid_mask = df_step3["stemmed_text"].fillna("").str.strip() != ""
    n_empty_after_clean = int((~valid_mask).sum())
    for _, r in df_step3[~valid_mask].iterrows():
        excluded_log_rows.append({
            "Tahap_Validasi": "4. Hanya Simbol / Angka / Emoji Tanpa Kata Bermakna",
            "Source_File": r.get("Source_File", ""),
            "Time": r.get("Time", ""),
            "User": r.get("User @", ""),
            "Comment Text": r.get(TEXT_COLUMN, ""),
            "Alasan_Dikeluarkan": "Kosong setelah cleansing & stopword removal (hanya simbol/angka/stiker)",
        })

    df_valid = df_step3.loc[valid_mask].copy().reset_index(drop=True)
    n_valid_rows = len(df_valid)

    # 6. Deduplikasi komentar identik (termasuk 424 baris duplikat scrape antara file 091926 & 166978)
    dup_mask = df_valid.duplicated(subset=["stemmed_text"], keep="first")
    n_duplicate_removed = int(dup_mask.sum())
    for _, r in df_valid[dup_mask].iterrows():
        excluded_log_rows.append({
            "Tahap_Validasi": "5. Duplikasi Komentar Identik",
            "Source_File": r.get("Source_File", ""),
            "Time": r.get("Time", ""),
            "User": r.get("User @", ""),
            "Comment Text": r.get(TEXT_COLUMN, ""),
            "Alasan_Dikeluarkan": "Teks hasil stemming identik dengan komentar sebelumnya (deduplikasi anti-kebocoran data)",
        })

    df_unique = df_valid.loc[~dup_mask].copy().reset_index(drop=True)
    n_unique_comments = len(df_unique)

    # Berikan ID unik per komentar: C0001 .. Cxxxx
    df_unique.insert(0, "Comment_ID", [f"C{i + 1:04d}" for i in range(n_unique_comments)])

    print("[7/7] Deteksi 4 Kategori Aspek Terketat (Perlindungan Anak, Peran Orang Tua, Peran Pemerintah/Komdigi, Efektivitas Kebijakan)...")
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

    # Rincian periode pada komentar unik yang lolos
    n_period_announce = int(df_unique["date_validation_note"].str.contains("Pengumuman|Autofill", regex=True).sum())
    n_period_impl = int(df_unique["date_validation_note"].str.contains("Implementasi|Relatif", regex=True).sum())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Simpan dataset unik hasil preprocessing
    df_save = df_unique.drop(columns=["detected_aspects_list"]).copy()
    df_save.to_excel(OUTPUT_FILE, index=False)

    # Simpan daftar lengkap data yang dikeluarkan pada tahap validasi
    excluded_val_df = pd.DataFrame(excluded_log_rows)
    excluded_val_df.to_excel(EXCLUDED_VALIDATION_FILE, index=False)

    # Simpan ringkasan jumlah data per tahap (Poin 8 & Catatan Revisi)
    stage_rows = [
        {
            "Tahap": "1. Data Mentah Gabungan (combined_dataset.xlsx)",
            "Jumlah_Data": n_raw_total,
            "Keterangan": f"Gabungan 7 berkas Excel komentar TikTok (termasuk {n_autofill_fixed} baris tanggal 08-03-2026 yang diperbaiki dari autofill tahun Excel)",
        },
        {
            "Tahap": "2. Dikeluarkan: Baris Kosong / Tanpa Teks (NaN)",
            "Jumlah_Data": n_missing_text,
            "Keterangan": "Kolom Comment Text kosong sejak dari hasil scraping awal",
        },
        {
            "Tahap": "3. Dikeluarkan: Di Luar Periode Penelitian (> 31 Mei 2026)",
            "Jumlah_Data": n_out_of_period,
            "Keterangan": "Komentar bertanggal Juni–Agustus 2026 dikeluarkan sesuai batasan waktu penelitian Bab I (s.d. 31 Mei 2026)",
        },
        {
            "Tahap": "4. Dikeluarkan: Komentar Tidak Relevan / Luar Topik / Spam",
            "Jumlah_Data": n_irrelevant_topic,
            "Keterangan": "Membahas topik luar (mis. Ronaldo, Ryu Kintaro, pengguna Malaysia, spam follow-back/FYP) sesuai Tabel 3.6 Bab III",
        },
        {
            "Tahap": "5. Dikeluarkan: Hanya Simbol / Angka / Emoji Tanpa Kata",
            "Jumlah_Data": n_empty_after_clean,
            "Keterangan": "Tidak menyisakan kata bermakna setelah cleansing & stopword removal",
        },
        {
            "Tahap": "6. Data Komentar Valid Sebelum Deduplikasi",
            "Jumlah_Data": n_valid_rows,
            "Keterangan": "Komentar berbahasa Indonesia yang valid dan berada dalam periode penelitian",
        },
        {
            "Tahap": "7. Dikeluarkan: Duplikasi Komentar Identik",
            "Jumlah_Data": n_duplicate_removed,
            "Keterangan": "Mencakup 424 baris duplikat antarsumber scrape (file 091926 vs 166978) & komentar berulang agar 0 overlap latih-uji",
        },
        {
            "Tahap": "8. Komentar Unik Bersih (dataset_preprocessed.xlsx)",
            "Jumlah_Data": n_unique_comments,
            "Keterangan": f"Komentar unik ber-ID C0001 s.d. C{n_unique_comments:04d} ({n_period_announce} tahap pengumuman Maret + {n_period_impl} tahap implementasi 28 Maret–31 Mei 2026)",
        },
        {
            "Tahap": "9. Dikeluarkan pada Tahap Aspek: Komentar Tanpa Aspek Terdeteksi",
            "Jumlah_Data": n_without_aspect,
            "Keterangan": "Tidak memenuhi aturan ketat 4 aspek; dikeluarkan & tidak otomatis dianggap netral (Revisi Poin 3 & 4)",
        },
        {
            "Tahap": "10. Komentar Unik Memiliki >= 1 Aspek",
            "Jumlah_Data": n_with_aspect,
            "Keterangan": "Komentar yang memuat minimal 1 dari 4 kategori aspek sesuai konteks",
        },
        {
            "Tahap": "11. Total Pasangan Komentar-Aspek Terbentuk",
            "Jumlah_Data": n_total_pairs,
            "Keterangan": "Satu komentar multi-aspek dipecah menjadi beberapa pasangan (komentar, aspek)",
        },
    ]
    pd.DataFrame(stage_rows).to_csv(STAGE_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n========================================")
    print("PREPROCESSING & VALIDASI SELESAI")
    print("========================================")
    print(f"Data mentah gabungan                       : {n_raw_total}")
    print(f" - Koreksi autofill tanggal (08-03-2026)   : {n_autofill_fixed}")
    print(f" - Dikeluarkan (Teks kosong/NaN)           : {n_missing_text}")
    print(f" - Dikeluarkan (Di luar periode >31 Mei)   : {n_out_of_period}")
    print(f" - Dikeluarkan (Tidak relevan/spam topik)  : {n_irrelevant_topic}")
    print(f" - Dikeluarkan (Hanya simbol/angka/emoji)  : {n_empty_after_clean}")
    print(f"Data valid sebelum deduplikasi             : {n_valid_rows}")
    print(f" - Dikeluarkan (Duplikat komentar identik) : {n_duplicate_removed}")
    print(f"Komentar unik bersih                       : {n_unique_comments}")
    print(f" - Dikeluarkan (Tanpa aspek terdeteksi)    : {n_without_aspect}")
    print(f"Komentar unik dengan >= 1 aspek            : {n_with_aspect}")
    print(f"Total pasangan (komentar, aspek)           : {n_total_pairs}")
    print(f"Output utama                               : {OUTPUT_FILE}")
    print(f"Log data dikeluarkan                       : {EXCLUDED_VALIDATION_FILE}")
    print(f"Ringkasan tahap                            : {STAGE_SUMMARY_CSV}")
    print("========================================")


if __name__ == "__main__":
    main()
