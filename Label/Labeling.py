# -*- coding: utf-8 -*-
"""
Tahap Pembentukan Pasangan Komentar-Aspek & Pelabelan Sentimen Berbasis Kamus InSet
Acuan Revisi Bab I-III:
1. Pemetaan ke 4 kategori aspek:
   - Perlindungan Anak
   - Peran Orang Tua
   - Peran Pemerintah/Komdigi
   - Efektivitas Kebijakan
2. Membentuk pasangan komentar-aspek (satu komentar dapat memiliki >1 aspek,
   dan tiap aspek diberi label positif, negatif, atau netral sesuai konteksnya).
3. Menggunakan Kamus InSet dengan mempertahankan kata negasi ("tidak", "bukan",
   "jangan", "belum", "tanpa", "kurang", "tak"). Komentar yang tidak terdeteksi
   aspeknya dikeluarkan (tidak otomatis dianggap netral).
4. TIDAK melakukan penyeimbangan (balancing) pada seluruh dataset sebelum split 80:20;
   penyeimbangan hanya dilakukan pada data latih di tahap klasifikasi.
"""

import os
import re
import sys
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# ============================================================
# IMPORT KAMUS INSET & DEFINISI 4 ASPEK
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(CURRENT_DIR))

from Dictionary import (  # noqa: E402
    ASPECT_CATEGORIES,
    ASPECT_KEYWORDS,
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pola regex untuk ke-4 kategori aspek (urutkan frasa terpanjang lebih dulu)
ASPECT_PATTERNS = {
    aspect: re.compile(
        r"\b(?:"
        + "|".join(re.escape(kw) for kw in sorted(set(kws), key=len, reverse=True))
        + r")\b"
    )
    for aspect, kws in ASPECT_KEYWORDS.items()
}

CLAUSE_PATTERNS = {
    aspect: re.compile(
        r"\b(?:"
        + "|".join(
            re.escape(kw)
            for kw in sorted(
                set(kws + (["awas"] if aspect == "Peran Orang Tua" else [])),
                key=len,
                reverse=True,
            )
        )
        + r")\b"
    )
    for aspect, kws in ASPECT_KEYWORDS.items()
}


# ============================================================
# FUNGSI PERHITUNGAN SKOR SENTIMEN INSET + PENANGANAN NEGASI
# ============================================================

def score_inset_tokens(tokens: list[str]) -> tuple[float, float, float, int, str]:
    """
    Menghitung skor positif, skor negatif, skor bersih (net), jumlah kata sentimen (hits),
    dan label sentimen ('positif', 'negatif', 'netral') menggunakan Kamus InSet
    dengan penanganan kata negasi (pembalikan polaritas pada jarak 1-2 token sebelum kata opini)
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
        val = 0.0
        step = 1

        if bigram in inset_positive:
            val = float(inset_positive[bigram])
            step = 2
        elif bigram in inset_negative:
            val = float(inset_negative[bigram])
            step = 2
        elif w in inset_positive:
            val = float(inset_positive[w])
            step = 1
        elif w in inset_negative:
            val = float(inset_negative[w])
            step = 1

        if val != 0.0:
            hits += 1
            has_negation = False
            # Jika token pertama bigram belum merupakan kata negasi, periksa 1-2 token sebelumnya
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
# EKSTRAKSI KONTEKS KLAUSA PER ASPEK & FITUR TEKS KLASIFIKASI
# ============================================================

def extract_aspect_context_and_sentiment(
    raw_text: str,
    norm_text: str,
    stem_text: str,
    aspect: str,
    total_detected_aspects: int,
) -> tuple[str, str, float, float, float, int, str]:
    """
    Menentukan segmen/klausa konteks spesifik untuk pasangan (komentar, aspek)
    dan menghitung skor InSet pada konteks aspek tersebut.
    Jika komentar membahas beberapa aspek yang dipisahkan konjungsi kontras
    (seperti 'tapi', 'namun', 'sedangkan', 'padahal', 'meski', 'walaupun'),
    maka tiap aspek dievaluasi pada klausa yang memuat kata kunci aspeknya.
    """
    stem_tokens = str(stem_text).split()
    pattern = ASPECT_PATTERNS[aspect]
    combined = f"{str(raw_text).lower()} {norm_text} {stem_text}"
    matched_kws = sorted(set(pattern.findall(combined)), key=len, reverse=True)
    matched_kw_str = ", ".join(matched_kws[:5])

    if total_detected_aspects <= 1 or len(stem_tokens) <= 5:
        p, ng, nt, h, lbl = score_inset_tokens(stem_tokens)
        return stem_text, matched_kw_str, p, ng, nt, h, lbl

    # Pecah berdasarkan kata penghubung kontras pada komentar multi-aspek
    clauses: list[list[str]] = []
    current_clause: list[str] = []
    has_contrast = False

    for tok in stem_tokens:
        if tok in contrastive_conjunctions and current_clause:
            clauses.append(current_clause)
            current_clause = []
            has_contrast = True
        else:
            current_clause.append(tok)
    if current_clause:
        clauses.append(current_clause)

    clause_pat = CLAUSE_PATTERNS[aspect]
    matched_clauses = [cl for cl in clauses if clause_pat.search(" ".join(cl))]
    if matched_clauses and len(clauses) > 1:
        local_tokens = [tok for cl in matched_clauses for tok in cl]
        p, ng, nt, h, lbl = score_inset_tokens(local_tokens)
        if h > 0 or has_contrast:
            return " ".join(local_tokens), matched_kw_str, p, ng, nt, h, lbl

    p, ng, nt, h, lbl = score_inset_tokens(stem_tokens)
    return stem_text, matched_kw_str, p, ng, nt, h, lbl


def build_aspect_classification_text(aspect_context: str) -> str:
    """
    Membentuk representasi fitur teks aspek (`aspect_text`) yang mempertahankan
    kata negasi ('tidak', 'bukan', 'jangan', 'belum', 'tanpa', 'kurang') serta
    pasangan negasi-opini (mis. 'tidak_tuju', 'jangan_blokir') dan kata konteks
    untuk perhitungan TF-IDF dan klasifikasi Multinomial Naive Bayes.
    """
    toks = str(aspect_context).split()
    opinion_feats: list[str] = []
    neutral_feats: list[str] = []
    i = 0
    n = len(toks)

    while i < n:
        w = toks[i]
        bigram = f"{w} {toks[i + 1]}" if i + 1 < n else ""
        step = 1
        is_op = False
        phrase_tok = w

        if bigram in inset_positive or bigram in inset_negative:
            phrase_tok = f"{w}_{toks[i + 1]}"
            step = 2
            is_op = True
        elif w in inset_positive or w in inset_negative:
            phrase_tok = w
            step = 1
            is_op = True

        if is_op:
            has_neg = False
            neg_word = None
            if w not in negation_words:
                for back in (1, 2):
                    if i - back >= 0 and toks[i - back] in negation_words:
                        has_neg = True
                        neg_word = toks[i - back]
                        break
            if has_neg and neg_word:
                opinion_feats.append(neg_word)
                opinion_feats.append(f"{neg_word}_{phrase_tok}")
                opinion_feats.append(f"tidak_{phrase_tok}")
            else:
                if w in negation_words:
                    opinion_feats.append(w)
                opinion_feats.append(phrase_tok)

            if i - 1 >= 0 and toks[i - 1] in intensifiers:
                opinion_feats.append(toks[i - 1])
            elif i + step < n and toks[i + step] in intensifiers:
                opinion_feats.append(toks[i + step])
        else:
            if (
                w not in negation_words
                and w not in intensifiers
                and w not in {"emoji", "tawa", "mohon", "kaget", "sticker", "emoji_tawa", "emoji_mohon", "emoji_kaget"}
            ):
                neutral_feats.append(w)
        i += step

    dedup_neutral = list(dict.fromkeys(neutral_feats))
    if opinion_feats:
        return " ".join(opinion_feats)
    if dedup_neutral:
        return " ".join(dedup_neutral[:3])
    return " ".join(toks[:2])


def main():
    print("========================================")
    print("TAHAP 3: PEMBENTUKAN PASANGAN KOMENTAR-ASPEK & PELABELAN INSET")
    print("========================================")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset preprocessing tidak ditemukan: {INPUT_FILE}\n"
            "Jalankan Preprocessing/preprocessing.py terlebih dahulu."
        )

    # Ekspor kamus InSet ke berkas TSV
    pos_tsv, neg_tsv = export_inset_tsv(OUTPUT_DIR)
    print(f"Kamus InSet diekspor ke:\n - {pos_tsv}\n - {neg_tsv}")

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
        source_file = str(row.get("Source_File", ""))

        combined = f"{raw_text.lower()} {norm_text} {stem_text}"
        detected_aspects = [
            asp for asp in ASPECT_CATEGORIES if ASPECT_PATTERNS[asp].search(combined)
        ]

        # Acuan Revisi Poin 3: Komentar yang tidak terdeteksi aspeknya JANGAN otomatis dianggap netral
        if not detected_aspects:
            excluded_records.append({
                "Comment_ID": cid,
                "Source_File": source_file,
                "Comment Text": raw_text,
                "normalized_text": norm_text,
                "stemmed_text": stem_text,
                "Alasan_Dikeluarkan": "Tidak mengandung kata kunci dari 4 kategori aspek",
            })
            continue

        for aspect in detected_aspects:
            ctx, matched_kw, pos_s, neg_s, net_s, hits, sentiment_lbl = extract_aspect_context_and_sentiment(
                raw_text=raw_text,
                norm_text=norm_text,
                stem_text=stem_text,
                aspect=aspect,
                total_detected_aspects=len(detected_aspects),
            )
            cls_text = build_aspect_classification_text(ctx)
            pair_records.append({
                "Pair_ID": f"P{len(pair_records) + 1:04d}",
                "Comment_ID": cid,
                "Source_File": source_file,
                "Comment Text": raw_text,
                "normalized_text": norm_text,
                "stemmed_text": stem_text,
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

    print(f"Komentar tanpa aspek (dikeluarkan)      : {n_excluded}")
    print(f"Komentar dengan >= 1 aspek              : {n_comments_with_aspect} ({n_multi_aspect_comments} komentar multi-aspek)")
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
