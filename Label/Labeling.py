# -*- coding: utf-8 -*-
"""
Tahap Pembentukan Pasangan Komentar-Aspek & Pelabelan Sentimen Berbasis Kamus InSet Asli
Acuan Revisi Bab I-III:
1. Menggunakan Kamus InSet dari sumber aslinya (Fajri Koto & Rahmaningtyas, 2017:
   3.609 entri positif & 6.609 entri negatif) beserta pencatatan terpisah atas
   modifikasi dan tambahan kata/bobot (`Label/inset_modifikasi_dan_tambahan.csv`).
2. Mempertahankan kata negasi ("tidak", "bukan", "jangan", "belum", "tanpa", "kurang", "tak")
   sampai menjadi masukan Naive Bayes (`aspect_text`), termasuk ketika kata negasi diikuti
   kata tindakan/platform non-kamus seperti "tidak blokir fb" -> menghasilkan fitur
   "tidak tidak_blokir blokir fb" sehingga makna negasinya TIDAK hilang (Revisi Poin 2).
3. Memperbaiki `build_aspect_classification_text()` agar mempertahankan seluruh konteks
   aspek yang relevan; TIDAK hanya mengambil kata opini dan TIDAK membatasi teks tanpa
   opini menjadi 3 kata pertama (Revisi Poin 3).
4. Menggunakan aturan pemetaan 4 aspek yang diperketat berdasarkan konteks (Revisi Poin 4).
5. Memperbaiki pemisahan konteks sentimen setiap aspek pada komentar multi-aspek,
   mencakup pemisahan berdasarkan:
   (a) Tanda baca / batas kalimat (`.`, `,`, `;`, `!`, `?`, `\n`) tanpa kata "tetapi"/"namun",
   (b) Konjungsi kontras ("tapi", "tetapi", "namun", "padahal", "sedangkan", "kecuali", dll.),
   (c) Jendela jangkar kata kunci aspek (aspect-anchor local windowing) apabila beberapa
       aspek berada dalam satu kalimat tanpa tanda baca maupun konjungsi kontras (Revisi Poin 5).
"""

import re
import sys
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# ============================================================
# IMPORT KAMUS INSET ASLI & DEFINISI 4 ASPEK TERKETAT
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(CURRENT_DIR))

from Dictionary import (  # noqa: E402
    ASPECT_CATEGORIES,
    ASPECT_KEYWORDS,
    detect_aspects_strict,
    inset_positive,
    inset_negative,
    intensifiers,
    negation_words,
    contrastive_conjunctions,
    export_inset_tsv,
)

# ============================================================
# KONFIGURASI FILE
# ============================================================

INPUT_FILE = BASE_DIR / "Preprocessing" / "dataset_preprocessed.xlsx"
OUTPUT_DIR = BASE_DIR / "Label"
PAIRS_OUTPUT_FILE = OUTPUT_DIR / "dataset_pasangan_komentar_aspek.xlsx"
OUTPUT_FILE = OUTPUT_DIR / "dataset_labeled.xlsx"
EXCLUDED_NO_ASPECT_FILE = OUTPUT_DIR / "komentar_tanpa_aspek_dikeluarkan.xlsx"
SEGMENTATION_TEST_CSV = OUTPUT_DIR / "uji_pemisahan_konteks_multiaspek.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pola regex pencocokan kata kunci aspek pada klausa/token
CLAUSE_PATTERNS = {
    aspect: re.compile(
        r"\b(?:"
        + "|".join(
            re.escape(kw)
            for kw in sorted(
                set(
                    kws
                    + (["awas", "ortu", "orang tua", "asuh", "didik", "damping", "pantau"] if aspect == "Peran Orang Tua" else [])
                    + (["perintah", "pemerintah", "komdigi", "menteri", "bijak", "atur", "blokir", "batas", "larang", "judol", "korupsi", "ktp"] if aspect == "Peran Pemerintah/Komdigi" else [])
                    + (["anak", "bocil", "remaja", "mental", "candu", "lindung", "bahaya", "porno", "bully", "rundung"] if aspect == "Perlindungan Anak" else [])
                    + (["efektif", "gagal", "berhasil", "percuma", "sia", "solusi", "terap", "implementasi", "palsu", "vpn", "bobol", "akun", "tahap", "youtube", "tiktok", "roblox", "sosmed", "medsos", "game"] if aspect == "Efektivitas Kebijakan" else [])
                ),
                key=len,
                reverse=True,
            )
        )
        + r")\b"
    )
    for aspect, kws in ASPECT_KEYWORDS.items()
}

# Daftar kata kerja tindakan regulasi yang berpolaritas negatif bila dinegasikan
# ("tidak blokir", "jangan blokir", "jangan batas", "jangan hapus", "jangan larang",
#  "tidak batas", "tidak larang", "tidak urus", "tidak tindak")
ACTION_VERBS_NEGATABLE = {
    "blokir", "blok", "block", "ban", "banned",
    "batas", "batasin", "dibatasin",
    "hapus", "nonaktif",
    "larang", "tutup",
    "urus", "tindak", "berantas",
    "awasi", "awas", "didik", "asuh", "damping", "pantau",
}


# ============================================================
# FUNGSI PERHITUNGAN SKOR SENTIMEN INSET + PENANGANAN NEGASI
# ============================================================

def score_inset_tokens(tokens: list[str]) -> tuple[float, float, float, int, str]:
    """
    Menghitung skor positif, skor negatif, skor bersih (net), jumlah kata sentimen (hits),
    dan label sentimen ('positif', 'negatif', 'netral') menggunakan Kamus InSet Asli
    dengan penanganan kata negasi (pembalikan polaritas pada jarak 1-2 token sebelum kata opini
    serta penanganan negasi pada kata kerja tindakan seperti 'jangan blokir' / 'tidak blokir')
    dan intensifier.
    """
    pos_score = 0.0
    neg_score = 0.0
    hits = 0
    i = 0
    n = len(tokens)

    while i < n:
        w = tokens[i]
        bigram = f"{w} {tokens[i + 1]}" if i + 1 < n else ""
        trigram = f"{w} {tokens[i + 1]} {tokens[i + 2]}" if i + 2 < n else ""
        val = 0.0
        step = 1

        # 1. Cek trigram spesifik terlebih dahulu
        if trigram and trigram in inset_positive:
            val = float(inset_positive[trigram])
            step = 3
        elif trigram and trigram in inset_negative:
            val = float(inset_negative[trigram])
            step = 3
        # 2. Cek bigram di kamus InSet
        elif bigram and bigram in inset_positive:
            val = float(inset_positive[bigram])
            step = 2
        elif bigram and bigram in inset_negative:
            val = float(inset_negative[bigram])
            step = 2
        # 3. Cek negasi + kata kerja tindakan kebijakan
        #    - "jangan blokir", "jangan batas", "jangan hapus", "jangan larang" -> penolakan larangan (-4.0)
        #    - "tidak urus", "tidak tindak", "tidak berantas", "tidak awas" -> kritik kelalaian (-3.0)
        #    - Adapun "tidak blokir", "tidak batas" pada laporan status ("akun gw tidak keblokir") bersifat netral (0.0),
        #      namun fitur negasinya ("tidak", "tidak_blokir") tetap dipertahankan penuh di build_aspect_classification_text()!
        elif w in {"jangan", "jgn"} and i + 1 < n and tokens[i + 1] in ACTION_VERBS_NEGATABLE:
            val = -4.0
            step = 2
        elif w in negation_words and i + 1 < n and tokens[i + 1] in {"urus", "tindak", "berantas", "awasi", "awas", "didik", "asuh", "damping", "pantau"}:
            val = -3.0
            step = 2
        # 4. Cek unigram di kamus InSet
        elif w in inset_positive:
            val = float(inset_positive[w])
            step = 1
        elif w in inset_negative:
            val = float(inset_negative[w])
            step = 1

        if val != 0.0:
            hits += 1
            has_negation = False
            # Jika token pertama bukan kata negasi, periksa 1-2 token sebelumnya untuk membalik polaritas
            if w not in negation_words:
                for back in (1, 2):
                    if i - back >= 0 and tokens[i - back] in negation_words:
                        has_negation = True
                        break

            multiplier = 1.0
            if i - 1 >= 0 and tokens[i - 1] in intensifiers:
                multiplier = float(intensifiers[tokens[i - 1]])
            elif i + step < n and tokens[i + step] in intensifiers:
                multiplier = float(intensifiers[tokens[i + step]])

            weighted_val = abs(val) * multiplier
            if (val > 0 and not has_negation) or (val < 0 and has_negation):
                pos_score += weighted_val
            else:
                neg_score += weighted_val

        i += step

    net_score = pos_score - neg_score
    if hits == 0 or net_score == 0:
        label = "netral"
    elif net_score > 0:
        label = "positif"
    else:
        label = "negatif"

    return pos_score, neg_score, net_score, hits, label


# ============================================================
# PEMISAHAN KONTEKS SENTIMEN SETIAP ASPEK (POIN 5 REVISI)
# ============================================================

def split_into_subclauses(clause_text: str, stem_text: str) -> list[list[str]]:
    """
    Memecah komentar menjadi daftar sub-klausa (masing-masing berupa list token)
    menggunakan:
    1. Batas tanda baca (`.` `,` `;` `!` `?` `\n`) yang telah dipertahankan sebagai `|` pada `clause_text`
    2. Kata penghubung kontras (`tapi`, `tetapi`, `namun`, `padahal`, `sedangkan`, `meski`,
       `walaupun`, `cuma`, `kecuali`, `mending`, `daripada`, `sebaliknya`)
    sehingga komentar multi-aspek TANPA kata "tetapi"/"namun" tetap terpecah pada batas klausa!
    """
    base_str = str(clause_text).strip() if clause_text and str(clause_text).strip() else str(stem_text).strip()
    raw_segments = [seg.strip() for seg in base_str.split("|") if seg.strip()]

    subclauses: list[list[str]] = []
    for seg in raw_segments:
        toks = seg.split()
        curr: list[str] = []
        for tok in toks:
            if tok in contrastive_conjunctions:
                if curr:
                    subclauses.append(curr)
                    curr = []
            else:
                curr.append(tok)
        if curr:
            subclauses.append(curr)

    return subclauses if subclauses else [str(stem_text).split()]


def segment_clause_by_aspect_anchors(
    tokens: list[str],
    target_aspect: str,
    detected_aspects: list[str],
) -> list[str]:
    """
    Apabila sebuah klausa tunggal (tanpa tanda baca dan tanpa kata hubung "tetapi"/"namun")
    memuat lebih dari satu aspek sekaligus — misalnya:
      "aturan komdigi bagus lindungi anak orang tua malah lalai"
    maka setiap token dalam klausa tersebut dipetakan ke jangkar aspek (aspect keyword anchor)
    terdekat berdasarkan jarak posisi indeks token, sehingga masing-masing aspek hanya
    mengambil kata opini di sekitar jangkar aspeknya sendiri!
    """
    if len(detected_aspects) <= 1 or len(tokens) <= 4:
        return tokens

    # Cari posisi indeks token yang menjadi jangkar (anchor) untuk masing-masing aspek yang terdeteksi
    anchor_positions: dict[str, list[int]] = {asp: [] for asp in detected_aspects}
    n = len(tokens)

    for idx in range(n):
        w = tokens[idx]
        bg = f"{w} {tokens[idx + 1]}" if idx + 1 < n else ""
        for asp in detected_aspects:
            pat = CLAUSE_PATTERNS[asp]
            if pat.search(w) or (bg and pat.search(bg)):
                anchor_positions[asp].append(idx)

    # Jika target_aspect memiliki jangkar dan minimal ada 1 aspek lain yang juga memiliki jangkar
    # pada posisi yang berbeda di klausa yang sama:
    active_aspects = [asp for asp, pos_list in anchor_positions.items() if pos_list]
    if target_aspect not in active_aspects or len(active_aspects) <= 1:
        return tokens

    # Periksa apakah posisi jangkar target_aspect benar-benar terpisah dari aspek lain
    target_pos = anchor_positions[target_aspect]
    other_pos = [p for asp in active_aspects if asp != target_aspect for p in anchor_positions[asp]]
    if not other_pos or set(target_pos) == set(other_pos):
        return tokens

    # Petakan setiap posisi token i (0..n-1) ke aspek dengan jarak jangkar terdekat.
    # Jika kata negasi berada tepat sebelum token yang masuk ke target_aspect, sertakan juga negasinya.
    selected_indices: set[int] = set()
    for idx in range(n):
        dist_target = min(abs(idx - tp) for tp in target_pos)
        dist_other = min(abs(idx - op) for op in other_pos)
        # Beri toleransi +1 jika kata berada tepat di sebelah jangkar aspek target
        if dist_target <= dist_other or dist_target <= 1:
            selected_indices.add(idx)
            # Pastikan kata negasi/intensifier 1-2 token sebelumnya ikut terbawa jika mengikat kata ini
            if idx - 1 >= 0 and (tokens[idx - 1] in negation_words or tokens[idx - 1] in intensifiers):
                selected_indices.add(idx - 1)

    if not selected_indices:
        return tokens

    return [tokens[i] for i in sorted(selected_indices)]


def extract_aspect_context_and_sentiment(
    raw_text: str,
    norm_text: str,
    stem_text: str,
    clause_text: str,
    aspect: str,
    detected_aspects: list[str],
    matched_kws: list[str],
) -> tuple[str, str, float, float, float, int, str]:
    """
    Menentukan segmen/klausa konteks spesifik untuk pasangan (komentar, aspek)
    dan menghitung skor InSet pada konteks aspek tersebut (Revisi Poin 5):
    1. Komentar dipecah menjadi sub-klausa menggunakan tanda baca (`.`, `,`, `;`, `!`, `?`, `\n`)
       MAUPUN konjungsi kontras (`tapi`, `tetapi`, `namun`, `padahal`, `sedangkan`, `kecuali`, dll.).
    2. Jika suatu sub-klausa memuat kata kunci untuk `aspect`, klausa tersebut dipilih.
       Jika sub-klausa tersebut tidak memuat opini langsung tetapi klausa tetangganya tidak memiliki
       aspek lain (klausa opini umum yang menerangkan klausa aspek), klausa tetangga tersebut digabungkan.
    3. Jika di dalam sub-klausa terpilih masih terdapat lebih dari 1 aspek tanpa tanda baca/konjungsi,
       dilakukan pemisahan berbasis jangkar aspek (`segment_clause_by_aspect_anchors`).
    """
    stem_tokens = str(stem_text).split()
    matched_kw_str = ", ".join(matched_kws[:5])

    if len(detected_aspects) <= 1 or len(stem_tokens) <= 4:
        p, ng, nt, h, lbl = score_inset_tokens(stem_tokens)
        return stem_text, matched_kw_str, p, ng, nt, h, lbl

    subclauses = split_into_subclauses(clause_text, stem_text)

    # Identifikasi aspek apa saja yang muncul di setiap sub-klausa
    clause_aspects: list[set[str]] = []
    for cl in subclauses:
        cl_str = " ".join(cl)
        asps_in_cl = {
            asp for asp in detected_aspects if CLAUSE_PATTERNS[asp].search(cl_str)
        }
        clause_aspects.append(asps_in_cl)

    # Ambil sub-klausa yang memuat aspek target
    target_clause_indices = [
        idx for idx, asps in enumerate(clause_aspects) if aspect in asps
    ]

    if target_clause_indices:
        local_tokens: list[str] = []
        for idx in target_clause_indices:
            cl_toks = subclauses[idx]
            # Jika di dalam sub-klausa ini ada >1 aspek sekaligus, pisahkan dengan anchor windowing
            asps_here = sorted(clause_aspects[idx])
            if len(asps_here) > 1:
                cl_toks = segment_clause_by_aspect_anchors(cl_toks, aspect, asps_here)
            local_tokens.extend(cl_toks)

        # Jika klausa aspek ini belum memuat kata opini (hits == 0), periksa apakah ada klausa
        # tetangga terdekat yang tidak ber-aspek (mis. klausa pembuka "setuju banget," lalu klausa aspek)
        p, ng, nt, h, lbl = score_inset_tokens(local_tokens)
        if h == 0 and len(subclauses) > 1:
            extended_tokens = list(local_tokens)
            for idx in target_clause_indices:
                if idx - 1 >= 0 and len(clause_aspects[idx - 1]) == 0:
                    extended_tokens = subclauses[idx - 1] + extended_tokens
                if idx + 1 < len(subclauses) and len(clause_aspects[idx + 1]) == 0:
                    extended_tokens = extended_tokens + subclauses[idx + 1]
            p2, ng2, nt2, h2, lbl2 = score_inset_tokens(extended_tokens)
            if h2 > 0:
                return " ".join(extended_tokens), matched_kw_str, p2, ng2, nt2, h2, lbl2

        return " ".join(local_tokens), matched_kw_str, p, ng, nt, h, lbl

    # Jika tidak terpecah oleh tanda baca/konjungsi, gunakan anchor windowing pada seluruh token
    anchored_tokens = segment_clause_by_aspect_anchors(stem_tokens, aspect, detected_aspects)
    p, ng, nt, h, lbl = score_inset_tokens(anchored_tokens)
    return " ".join(anchored_tokens), matched_kw_str, p, ng, nt, h, lbl


# ============================================================
# PEMBENTUKAN FITUR TEKS KLASIFIKASI (POIN 2 & POIN 3 REVISI)
# ============================================================

NOISE_TOKENS = {
    "emoji", "tawa", "mohon", "kaget", "sticker",
    "emoji_tawa", "emoji_mohon", "emoji_kaget",
}


def build_aspect_classification_text(aspect_context: str, aspect: str | None = None) -> str:
    """
    Acuan Revisi Poin 2 & Poin 3:
    1. Poin 2: Pertahankan negasi sampai menjadi masukan Naive Bayes.
       Pada konteks seperti "tidak blokir fb" atau "jangan hapus akun", kata negasi ("tidak",
       "bukan", "jangan", "belum", "tanpa", "kurang") WAJIB dipertahankan sebagai token mandiri
       MAUPUN fitur gabungan negasi (`tidak_blokir`, `jangan_hapus`) meskipun kata setelahnya
       bukan kata di dalam kamus InSet! Dengan demikian "tidak blokir fb" menghasilkan
       `"tidak tidak_blokir blokir fb"` dan TIDAK berubah menjadi `"blokir fb"`.
    2. Poin 3: Pertahankan konteks aspek yang relevan; JANGAN hanya mengambil kata opini
       dan JANGAN membatasi teks tanpa opini menjadi tiga kata pertama.
       Seluruh token konteks pada `aspect_context` dipertahankan secara utuh bersama
       fitur senyawa negasi dan bigram opini.
    """
    toks = [t.strip().lower() for t in str(aspect_context).split() if t.strip()]
    if not toks:
        return ""

    features: list[str] = []
    n = len(toks)
    i = 0

    while i < n:
        w = toks[i]
        if w in NOISE_TOKENS:
            i += 1
            continue

        # 1. Jika token saat ini adalah kata negasi, pertahankan kata negasinya
        #    DAN bentuk token senyawa `<negasi>_<kata_berikutnya>` (Revisi Poin 2)
        if w in negation_words:
            features.append(w)
            if i + 1 < n:
                nxt = toks[i + 1]
                if nxt not in NOISE_TOKENS and nxt not in negation_words:
                    features.append(f"{w}_{nxt}")
                    if w != "tidak":
                        features.append(f"tidak_{nxt}")
                    # Jika kata setelah negasi membentuk bigram InSet (mis. "masuk akal"), ikat juga
                    if i + 2 < n:
                        bg_next = f"{nxt} {toks[i + 2]}"
                        if bg_next in inset_positive or bg_next in inset_negative:
                            features.append(f"{w}_{nxt}_{toks[i + 2]}")
            i += 1
            continue

        # 2. Cek apakah token saat ini membentuk bigram di kamus InSet
        bigram = f"{w} {toks[i + 1]}" if i + 1 < n else ""
        if bigram and (bigram in inset_positive or bigram in inset_negative):
            # Pertahankan token asli pembentuk konteks serta token bigram senyawa
            features.append(w)
            features.append(toks[i + 1])
            features.append(f"{w}_{toks[i + 1]}")
            i += 2
            continue

        # 3. Pertahankan seluruh kata konteks dan kata opini (Revisi Poin 3:
        #    tidak membuang konteks non-opini dan tidak memotong 3 kata pertama)
        features.append(w)
        i += 1

    if not features:
        return " ".join(toks)

    return " ".join(features)


# ============================================================
# PENGUJIAN PEMISAHAN KONTEKS MULTI-ASPEK & NEGASI (POIN 2, 3, 5)
# ============================================================

def run_segmentation_and_negation_verification() -> pd.DataFrame:
    """
    Menguji secara otomatis:
    1. Pelestarian negasi pada "tidak blokir fb" hingga ke keluaran `build_aspect_classification_text()`
    2. Komentar multi-aspek DENGAN kata penghubung kontras ("tetapi", "tapi", "namun")
    3. Komentar multi-aspek TANPA kata penghubung "tetapi"/"namun" (dipisahkan tanda baca koma/titik
       maupun berada dalam satu kalimat tanpa tanda baca).
    Hasil pengujian disimpan ke `Label/uji_pemisahan_konteks_multiaspek.csv`.
    """
    test_cases = [
        {
            "id_uji": "UJI-01 (Negasi Non-Opini)",
            "tipe": "Negasi pada kata non-opini ('tidak blokir fb')",
            "raw_text": "komdigi tidak blokir fb",
            "norm_text": "komdigi tidak blokir fb",
            "stem_text": "komdigi tidak blokir fb",
            "clause_text": "komdigi tidak blokir fb",
        },
        {
            "id_uji": "UJI-02 (Multi-Aspek dengan 'tetapi')",
            "tipe": "Multi-aspek dengan konjungsi kontras ('tetapi')",
            "raw_text": "Saya sangat setuju kebijakan pemerintah melindungi mental anak, tetapi implementasi sistem verifikasi umur masih mudah dibobol pakai vpn",
            "norm_text": "saya sangat setuju kebijakan pemerintah melindungi mental anak tetapi implementasi sistem verifikasi umur masih mudah dibobol pakai vpn",
            "stem_text": "sangat tuju bijak perintah lindung mental anak tetapi implementasi sistem verifikasi umur mudah bobol pakai vpn",
            "clause_text": "sangat tuju bijak perintah lindung mental anak | tetapi implementasi sistem verifikasi umur mudah bobol pakai vpn",
        },
        {
            "id_uji": "UJI-03 (Multi-Aspek TANPA 'tetapi'/'namun' - Tanda Baca Titik/Koma)",
            "tipe": "Multi-aspek tanpa 'tetapi'/'namun' (dipisahkan tanda baca titik/koma)",
            "raw_text": "Aturan pemerintah bagus sekali melindungi mental anak kecil. Sayangnya orang tua di rumah malah lalai dan lepas tangan.",
            "norm_text": "aturan pemerintah bagus sekali melindungi mental anak kecil sayangnya orang tua di rumah malah lalai dan lepas tangan",
            "stem_text": "atur perintah bagus sekali lindung mental anak kecil sayang orang tua rumah malah lalai lepas tangan",
            "clause_text": "atur perintah bagus sekali lindung mental anak kecil | sayang orang tua rumah malah lalai lepas tangan",
        },
        {
            "id_uji": "UJI-04 (Multi-Aspek TANPA 'tetapi'/'namun' & Tanpa Tanda Baca)",
            "tipe": "Multi-aspek tanpa 'tetapi'/'namun' dalam satu klausa (anchor windowing)",
            "raw_text": "langkah komdigi mantap menjaga kesehatan mental anak orang tua justru lalai lepas tangan",
            "norm_text": "langkah komdigi mantap menjaga kesehatan mental anak orang tua justru lalai lepas tangan",
            "stem_text": "langkah komdigi mantap jaga sehat mental anak orang tua justru lalai lepas tangan",
            "clause_text": "langkah komdigi mantap jaga sehat mental anak orang tua justru lalai lepas tangan",
        },
        {
            "id_uji": "UJI-05 (Multi-Aspek TANPA 'tetapi'/'namun' - Opini Berbeda)",
            "tipe": "Multi-aspek tanpa 'tetapi'/'namun' (Pengawasan Orang Tua Positif vs Efektivitas Negatif)",
            "raw_text": "Pengawasan orang tua sangat bagus mendampingi anak belajar. Pemblokiran akun sosmed percuma karena tanggal lahir bisa dipalsukan.",
            "norm_text": "pengawasan orang tua sangat bagus mendampingi anak belajar pemblokiran akun sosmed percuma karena tanggal lahir bisa dipalsukan",
            "stem_text": "awas orang tua sangat bagus damping anak ajar blokir akun sosmed percuma tanggal lahir palsu",
            "clause_text": "awas orang tua sangat bagus damping anak ajar | blokir akun sosmed percuma tanggal lahir palsu",
        },
    ]

    rows = []
    for tc in test_cases:
        asp_map = detect_aspects_strict(tc["raw_text"], tc["norm_text"], tc["stem_text"])
        det_aspects = list(asp_map.keys())
        for asp in det_aspects:
            ctx, m_kw, ps, ns, nets, h, lbl = extract_aspect_context_and_sentiment(
                raw_text=tc["raw_text"],
                norm_text=tc["norm_text"],
                stem_text=tc["stem_text"],
                clause_text=tc["clause_text"],
                aspect=asp,
                detected_aspects=det_aspects,
                matched_kws=asp_map[asp],
            )
            cls_txt = build_aspect_classification_text(ctx, asp)
            rows.append({
                "ID_Uji": tc["id_uji"],
                "Tipe_Pengujian": tc["tipe"],
                "Komentar_Uji": tc["raw_text"],
                "Aspek_Terdeteksi": asp,
                "Kata_Kunci_Aspek": m_kw,
                "Konteks_Klausa_Aspek": ctx,
                "Fitur_Masukan_Naive_Bayes (aspect_text)": cls_txt,
                "Skor_Positif": ps,
                "Skor_Negatif": ns,
                "Skor_Bersih": nets,
                "Label_Sentimen": lbl,
            })

    ver_df = pd.DataFrame(rows)
    ver_df.to_csv(SEGMENTATION_TEST_CSV, index=False, encoding="utf-8-sig")
    return ver_df


def main():
    print("========================================")
    print("TAHAP 3: PEMBENTUKAN PASANGAN KOMENTAR-ASPEK & PELABELAN INSET ASLI")
    print("========================================")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset preprocessing tidak ditemukan: {INPUT_FILE}\n"
            "Jalankan Preprocessing/preprocessing.py terlebih dahulu."
        )

    # 1. Ekspor kamus InSet asli + log modifikasi & tambahan
    pos_tsv, neg_tsv = export_inset_tsv(OUTPUT_DIR)
    print(f"Kamus InSet Aktif (berbasis InSet Asli Fajri Koto, 2017) diekspor ke:")
    print(f" - {pos_tsv} ({len(inset_positive)} entri positif)")
    print(f" - {neg_tsv} ({len(inset_negative)} entri negatif)")
    print(f" - {OUTPUT_DIR / 'inset_modifikasi_dan_tambahan.csv'} (catatan perubahan & tambahan)")

    # 2. Jalankan verifikasi negasi & pemisahan klausa multi-aspek (dengan & tanpa 'tetapi'/'namun')
    ver_df = run_segmentation_and_negation_verification()
    print(f"\nVerifikasi pemisahan konteks multi-aspek & negasi disimpan di: {SEGMENTATION_TEST_CSV}")
    for _, r in ver_df.iterrows():
        print(
            f"  [{r['ID_Uji']}] {r['Aspek_Terdeteksi']:25s} -> Label: {r['Label_Sentimen']:7s} "
            f"| aspect_text: {r['Fitur_Masukan_Naive_Bayes (aspect_text)']}"
        )

    df = pd.read_excel(INPUT_FILE)
    total_unique_comments = len(df)
    print(f"\nTotal komentar unik hasil preprocessing : {total_unique_comments}")

    pair_records = []
    excluded_records = []

    for idx, row in df.iterrows():
        cid = str(row.get("Comment_ID", f"C{idx + 1:04d}"))
        raw_text = str(row.get("Comment Text", ""))
        norm_text = str(row.get("normalized_text", ""))
        stem_text = str(row.get("stemmed_text", ""))
        clause_text = str(row.get("clause_text", stem_text))
        source_file = str(row.get("Source_File", ""))

        # Deteksi aspek dengan aturan konteks yang diperketat (Revisi Poin 4)
        aspect_kw_map = detect_aspects_strict(raw_text, norm_text, stem_text)
        detected_aspects = list(aspect_kw_map.keys())

        # Komentar yang tidak terdeteksi aspeknya dikeluarkan (tidak otomatis dianggap netral)
        if not detected_aspects:
            excluded_records.append({
                "Comment_ID": cid,
                "Source_File": source_file,
                "Comment Text": raw_text,
                "normalized_text": norm_text,
                "stemmed_text": stem_text,
                "Alasan_Dikeluarkan": "Tidak memenuhi aturan kata kunci & konteks dari 4 kategori aspek",
            })
            continue

        for aspect in detected_aspects:
            ctx, matched_kw, pos_s, neg_s, net_s, hits, sentiment_lbl = extract_aspect_context_and_sentiment(
                raw_text=raw_text,
                norm_text=norm_text,
                stem_text=stem_text,
                clause_text=clause_text,
                aspect=aspect,
                detected_aspects=detected_aspects,
                matched_kws=aspect_kw_map[aspect],
            )
            cls_text = build_aspect_classification_text(ctx, aspect)
            pair_records.append({
                "Pair_ID": f"P{len(pair_records) + 1:04d}",
                "Comment_ID": cid,
                "Source_File": source_file,
                "Comment Text": raw_text,
                "normalized_text": norm_text,
                "stemmed_text": stem_text,
                "clause_text": clause_text,
                "aspect": aspect,
                "matched_keywords": matched_kw,
                "aspect_context": ctx,
                "aspect_text": cls_text,
                "positive_score": pos_s,
                "negative_score": neg_s,
                "sentiment_score": net_s,
                "inset_hits": hits,
                "sentiment": sentiment_lbl,
            })

    pairs_df = pd.DataFrame(pair_records)
    excluded_df = pd.DataFrame(excluded_records)

    # Simpan pasangan komentar-aspek (sebelum pembagian 80:20 dan sebelum penyeimbangan)
    pairs_df.to_excel(PAIRS_OUTPUT_FILE, index=False)
    pairs_df.to_excel(OUTPUT_FILE, index=False)
    excluded_df.to_excel(EXCLUDED_NO_ASPECT_FILE, index=False)

    n_comments_with_aspect = pairs_df["Comment_ID"].nunique()
    n_excluded = len(excluded_df)
    n_multi_aspect_comments = int((pairs_df.groupby("Comment_ID")["aspect"].count() > 1).sum())

    # Hitung berapa komentar multi-aspek yang memiliki label berbeda antar aspeknya
    multi_diff_sent = int(
        (pairs_df.groupby("Comment_ID")["sentiment"].nunique() > 1).sum()
    )

    print(f"Komentar tanpa aspek (dikeluarkan)      : {n_excluded}")
    print(f"Komentar dengan >= 1 aspek              : {n_comments_with_aspect} ({n_multi_aspect_comments} komentar multi-aspek, {multi_diff_sent} memiliki sentimen berbeda antar-aspek)")
    print(f"Total pasangan (komentar, aspek)        : {len(pairs_df)}")

    print("\nDistribusi Sentimen Pasangan Komentar-Aspek (Sebelum Penyeimbangan):")
    sent_counts = pairs_df["sentiment"].value_counts()
    for lbl in ["positif", "negatif", "netral"]:
        cnt = int(sent_counts.get(lbl, 0))
        pct = (cnt / len(pairs_df) * 100) if len(pairs_df) else 0.0
        print(f" - {lbl.capitalize():8s}: {cnt:4d} ({pct:.2f}%)")

    print("\nTabel Distribusi 4 Aspek (Sebelum Penyeimbangan):")
    ctab = pd.crosstab(pairs_df["aspect"], pairs_df["sentiment"])
    for lbl in ["positif", "negatif", "netral"]:
        if lbl not in ctab.columns:
            ctab[lbl] = 0
    ctab = ctab.reindex(ASPECT_CATEGORIES, fill_value=0)[["positif", "negatif", "netral"]]
    ctab["Total"] = ctab.sum(axis=1)
    print(ctab.rename(columns={"positif": "Positif", "negatif": "Negatif", "netral": "Netral"}))

    print(f"\nBerkas pasangan komentar-aspek disimpan di : {PAIRS_OUTPUT_FILE}")
    print(f"Berkas dataset berlabel disimpan di        : {OUTPUT_FILE}")
    print(f"Berkas komentar tanpa aspek disimpan di    : {EXCLUDED_NO_ASPECT_FILE}")
    print("========================================")


if __name__ == "__main__":
    main()
