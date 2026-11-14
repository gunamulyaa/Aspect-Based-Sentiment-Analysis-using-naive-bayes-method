# -*- coding: utf-8 -*-
"""
Tahap Pembagian Data 80:20 (Bebas Kebocoran Komentar), Penyeimbangan Data Latih,
Perhitungan TF-IDF (Sesuai Rumus Bab III), Pelatihan Multinomial Naive Bayes,
dan Evaluasi Keseluruhan serta Per Aspek.

Acuan Revisi Bab I-III:
- Poin 4: Perhitungan TF-IDF sesuai rumus Bab III:
    TF(t, d)  = f(t, d) / |d|
    IDF(t)    = log(N / df(t)) + 1
    TF-IDF    = TF(t, d) * IDF(t)
  Kosakata (vocabulary) dan nilai IDF HANYA dibentuk dari data latih.
- Poin 5: Pisahkan data latih dan uji 80:20 SEBELUM penyeimbangan berdasarkan Comment_ID
  agar komentar yang sama beserta seluruh pasangan aspeknya tidak tersebar ke kedua kelompok
  (0 tumpang tindih / zero overlap). Penyeimbangan (oversampling) hanya dilakukan pada data latih.
- Poin 6: Latih ulang Naive Bayes dan buat evaluasi keseluruhan serta per aspek:
  Confusion Matrix, Accuracy, Precision, Recall, F1-Score, dan Jumlah Data Uji.
- Poin 7: Buat ulang tabel distribusi 4 aspek dengan kolom Positif, Negatif, Netral, dan Total
  menggunakan data SEBELUM penyeimbangan; tampilkan jumlah data setelah penyeimbangan secara
  terpisah untuk menjelaskan eksperimen.
"""

import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# =========================================================
# KONFIGURASI DIREKTORI & BERKAS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
LABEL_DIR = BASE_DIR / "Label"
KLASIFIKASI_DIR = BASE_DIR / "Klasifikasi"
VISUALISASI_DIR = BASE_DIR / "Visualisasi"

sys.path.insert(0, str(LABEL_DIR))
from Dictionary import ASPECT_CATEGORIES  # noqa: E402

KLASIFIKASI_DIR.mkdir(parents=True, exist_ok=True)
VISUALISASI_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PAIRS_FILE = LABEL_DIR / "dataset_pasangan_komentar_aspek.xlsx"
if not INPUT_PAIRS_FILE.exists():
    INPUT_PAIRS_FILE = LABEL_DIR / "dataset_labeled.xlsx"

TRAIN_RAW_FILE = KLASIFIKASI_DIR / "data_latih_80.xlsx"
TRAIN_BALANCED_FILE = KLASIFIKASI_DIR / "data_latih_80_balanced.xlsx"
LABEL_BALANCED_FILE = LABEL_DIR / "dataset_labeled_balanced.xlsx"
TEST_FILE = KLASIFIKASI_DIR / "data_uji_20.xlsx"
TEST_PREDICTIONS_FILE = KLASIFIKASI_DIR / "hasil_prediksi_data_uji.xlsx"

ASPECT_SUMMARY_FILE = KLASIFIKASI_DIR / "aspect_sentiment_summary.xlsx"
BALANCING_EXP_FILE = KLASIFIKASI_DIR / "tabel_eksperimen_balancing_latih.xlsx"
FULL_EVAL_EXCEL = KLASIFIKASI_DIR / "evaluasi_naive_bayes_lengkap.xlsx"
OVERLAP_PROOF_CSV = KLASIFIKASI_DIR / "bukti_tidak_tumpang_tindih.csv"
MODEL_PKL_FILE = KLASIFIKASI_DIR / "naive_bayes_model.pkl"

TEXT_COLUMN = "aspect_text"
LABEL_COLUMN = "sentiment"
SENTIMENT_LABELS = ["positif", "negatif", "netral"]


# =========================================================
# 1. PEMBAGIAN DATA LATIH & UJI 80:20 PER KOMENTAR UNIK
# =========================================================

def split_train_test_by_comment(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Memisahkan data latih (80%) dan data uji (20%) berdasarkan Comment_ID unik
    SEBELUM penyeimbangan, sehingga seluruh pasangan aspek dari komentar yang sama
    berada di kelompok yang sama (0 overlap).
    """
    unique_cids = (
        df["Comment_ID"]
        .drop_duplicates()
        .sample(frac=1.0, random_state=random_state)
        .tolist()
    )
    n_total_comments = len(unique_cids)
    n_train_comments = int(round(n_total_comments * (1.0 - test_size)))

    train_cid_set = set(unique_cids[:n_train_comments])
    test_cid_set = set(unique_cids[n_train_comments:])

    train_df = df[df["Comment_ID"].isin(train_cid_set)].copy().reset_index(drop=True)
    test_df = df[df["Comment_ID"].isin(test_cid_set)].copy().reset_index(drop=True)

    # Pembuktian 0 tumpang tindih (Zero Overlap Proof)
    overlap_cid = set(train_df["Comment_ID"]) & set(test_df["Comment_ID"])
    overlap_raw = set(train_df["Comment Text"]) & set(test_df["Comment Text"])
    overlap_norm = set(train_df["normalized_text"]) & set(test_df["normalized_text"])
    overlap_stem = set(train_df["stemmed_text"]) & set(test_df["stemmed_text"])

    assert len(overlap_cid) == 0, f"Terdapat overlap Comment_ID: {len(overlap_cid)}"
    assert len(overlap_raw) == 0, f"Terdapat overlap Comment Text: {len(overlap_raw)}"
    assert len(overlap_norm) == 0, f"Terdapat overlap normalized_text: {len(overlap_norm)}"
    assert len(overlap_stem) == 0, f"Terdapat overlap stemmed_text: {len(overlap_stem)}"

    proof = {
        "Total_Komentar_Unik_Beraspek": n_total_comments,
        "Komentar_Unik_Latih_80": len(train_cid_set),
        "Komentar_Unik_Uji_20": len(test_cid_set),
        "Pasangan_Aspek_Latih_Sebelum_Balancing": len(train_df),
        "Pasangan_Aspek_Uji": len(test_df),
        "Total_Pasangan_Aspek": len(df),
        "Overlap_Comment_ID": len(overlap_cid),
        "Overlap_Comment_Text": len(overlap_raw),
        "Overlap_Normalized_Text": len(overlap_norm),
        "Overlap_Stemmed_Text": len(overlap_stem),
        "Status_Validasi": "TERBUKTI 0 TUMPANG TINDIH (ZERO OVERLAP)",
    }
    return train_df, test_df, proof


# =========================================================
# 2. PENYEIMBANGAN (BALANCING) HANYA PADA DATA LATIH
# =========================================================

def balance_training_data(train_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Menyeimbangkan kelas sentimen ('positif', 'negatif', 'netral') HANYA pada data latih
    menggunakan random oversampling menuju jumlah kelas terbanyak di data latih.
    Data uji sama sekali tidak disentuh.
    """
    counts = train_df[LABEL_COLUMN].value_counts()
    target_count = int(counts.max())

    balanced_parts = []
    for label in SENTIMENT_LABELS:
        subset = train_df[train_df[LABEL_COLUMN] == label]
        if len(subset) < target_count:
            deficit = target_count - len(subset)
            extra = subset.sample(n=deficit, replace=True, random_state=random_state)
            sampled = pd.concat([subset, extra], ignore_index=True)
        else:
            sampled = subset.sample(n=target_count, replace=False, random_state=random_state)
        balanced_parts.append(sampled)

    balanced_train_df = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    return balanced_train_df


# =========================================================
# 3. PERHITUNGAN TF-IDF SESUAI RUMUS BAB III (DARI DATA LATIH)
# =========================================================
# Acuan Revisi Poin 7 (Dicocokkan dengan Bab II Persamaan 2.1-2.3 & Bab III Tabel 3.12):
#   TF(t, d)     = f(t, d) / |d|         (frekuensi relatif kata t terhadap total kata di dokumen d)
#   IDF(t)       = log10(N / df(t))      (logaritma basis 10 TANPA penambahan konstanta +1,
#                                         sesuai Tabel 3.12: N=3, df=1 -> log10(3/1)=0.477;
#                                         df=2 -> log10(3/2)=0.176; df=3 -> log10(3/3)=0.000)
#   W(d, t)      = TF(t, d) * IDF(t)


def verify_tfidf_formula_against_bab3() -> pd.DataFrame:
    """
    Memverifikasi kesesuaian rumus TF-IDF di kode dengan perhitungan manual Tabel 3.11 & Tabel 3.12
    pada Bab III Laporan Seminar:
      D1: "tuju bijak perintah lindung anak" (5 kata; pada Tabel 3.11 diasumsikan panjang dokumen 6 kata)
      Untuk N = 3 dokumen:
      - DF = 1 -> IDF = log10(3/1) = 0.4771 (dibulatkan 3 desimal: 0.477), TF = 1/6 = 0.167 -> W = 0.080
      - DF = 2 -> IDF = log10(3/2) = 0.1761 (dibulatkan 3 desimal: 0.176), TF = 1/6 = 0.167 -> W = 0.029
      - DF = 3 -> IDF = log10(3/3) = 0.0000 (dibulatkan 3 desimal: 0.000), TF = 1/6 = 0.167 -> W = 0.000
    """
    n_bab3 = 3
    rows = []
    expected_table_3_12 = {
        1: (0.477, 0.080),
        2: (0.176, 0.029),
        3: (0.000, 0.000),
    }
    tf_bab3 = 1.0 / 6.0
    for df_val, (exp_idf, exp_w) in expected_table_3_12.items():
        calc_idf = math.log10(n_bab3 / df_val)
        calc_w = round(tf_bab3, 3) * round(calc_idf, 3)
        assert round(calc_idf, 3) == exp_idf, f"IDF mismatch for df={df_val}: {round(calc_idf, 3)} != {exp_idf}"
        assert round(calc_w, 3) == exp_w, f"TF-IDF mismatch for df={df_val}: {round(calc_w, 3)} != {exp_w}"
        rows.append({
            "N_Dokumen": n_bab3,
            "DF": df_val,
            "TF_Bab_III (1/6)": round(tf_bab3, 3),
            "Rumus_IDF": f"log10({n_bab3}/{df_val})",
            "IDF_Kode (3 desimal)": round(calc_idf, 3),
            "IDF_Tabel_3_12_Bab_III": exp_idf,
            "TF_IDF_Kode (3 desimal)": round(calc_w, 3),
            "TF_IDF_Tabel_3_12_Bab_III": exp_w,
            "Status": "SESUAI PERSIS DENGAN TABEL 3.12 BAB III",
        })
    ver_df = pd.DataFrame(rows)
    ver_df.to_csv(KLASIFIKASI_DIR / "verifikasi_rumus_tfidf_bab3.csv", index=False, encoding="utf-8-sig")
    ver_df.to_csv(VISUALISASI_DIR / "verifikasi_rumus_tfidf_bab3.csv", index=False, encoding="utf-8-sig")
    return ver_df


def compute_tfidf_from_train(train_texts: list[str]) -> tuple[list[str], dict[str, float], dict[str, int]]:
    """
    Menghitung kosakata (vocabulary), Document Frequency (df), dan Inverse Document Frequency (IDF)
    HANYA dari dokumen data latih sesuai rumus pada Bab II & Bab III (Tabel 3.12):
      TF(t, d)     = f(t, d) / |d|
      IDF(t)       = log10(N / df(t))
      TF-IDF(t, d) = TF(t, d) * IDF(t)
    dengan N = jumlah dokumen latih, df(t) = jumlah dokumen latih yang memuat kata t,
    logaritma menggunakan basis 10 (`math.log10`), dan TANPA konstanta `+ 1`.
    """
    docs: list[list[str]] = []
    vocab_set: set[str] = set()

    for text in train_texts:
        tokens = str(text).split()
        if not tokens:
            continue
        docs.append(tokens)
        vocab_set.update(tokens)

    vocab = sorted(vocab_set)
    n_docs = len(docs)
    df_count: dict[str, int] = defaultdict(int)

    for tokens in docs:
        for term in set(tokens):
            df_count[term] += 1

    idf: dict[str, float] = {}
    for term in vocab:
        # Rumus Bab III Tabel 3.12: IDF(t) = log10(N / df(t)) tanpa konstanta +1
        idf[term] = math.log10(n_docs / df_count[term])

    return vocab, idf, dict(df_count)


def vectorize_tfidf(text: str, vocab_set: set[str], idf: dict[str, float]) -> dict[str, float]:
    """
    Menghitung bobot TF-IDF untuk satu dokumen menggunakan kosakata dan IDF dari data latih.
    Kata di data uji yang tidak ada di kosakata data latih diabaikan (OOV ignored).
    """
    tokens = str(text).split()
    if not tokens:
        return {}
    total_terms = float(len(tokens))
    tf_counts = Counter(tokens)
    vec: dict[str, float] = {}
    for term, count in tf_counts.items():
        if term in vocab_set:
            tf_val = count / total_terms
            vec[term] = tf_val * idf[term]
    return vec


# =========================================================
# 4. PELATIHAN & PREDIKSI MULTINOMIAL NAIVE BAYES
# =========================================================

def train_multinomial_nb(
    train_df: pd.DataFrame,
    vocab: list[str],
    idf: dict[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float], dict[str, int]]:
    """
    Melatih model Multinomial Naive Bayes menggunakan bobot TF-IDF data latih
    dengan Laplace smoothing (+1).
    """
    vocab_set = set(vocab)
    class_counts: dict[str, int] = defaultdict(int)
    class_term_totals: dict[str, float] = defaultdict(float)
    class_term_counts: dict[str, dict[str, float]] = {lbl: defaultdict(float) for lbl in SENTIMENT_LABELS}

    for _, row in train_df.iterrows():
        label = str(row[LABEL_COLUMN]).strip().lower()
        text = str(row[TEXT_COLUMN])
        vec = vectorize_tfidf(text, vocab_set, idf)
        class_counts[label] += 1
        for term, weight in vec.items():
            if weight > 0.0:
                class_term_counts[label][term] += weight
                class_term_totals[label] += weight

    total_docs = float(len(train_df))
    priors = {
        label: math.log(class_counts[label] / total_docs)
        for label in SENTIMENT_LABELS
    }
    return priors, {k: dict(v) for k, v in class_term_counts.items()}, dict(class_term_totals), dict(class_counts)


def predict_multinomial_nb(
    text: str,
    priors: dict[str, float],
    class_term_counts: dict[str, dict[str, float]],
    class_term_totals: dict[str, float],
    vocab: list[str],
    vocab_set: set[str],
    idf: dict[str, float],
) -> str:
    vec = vectorize_tfidf(text, vocab_set, idf)
    v_size = float(len(vocab))
    scores: dict[str, float] = {}

    for label in SENTIMENT_LABELS:
        score = priors[label]
        denom = class_term_totals.get(label, 0.0) + v_size
        term_weights = class_term_counts.get(label, {})
        for term, weight in vec.items():
            if weight > 0.0:
                w_count = term_weights.get(term, 0.0)
                score += weight * math.log((w_count + 1.0) / denom)
        scores[label] = score

    return max(scores, key=scores.get)


# =========================================================
# 5. EVALUASI METRIK & CONFUSION MATRIX (KESELURUHAN & PER ASPEK)
# =========================================================

def compute_evaluation_metrics(y_true: list[str], y_pred: list[str], scope_name: str = "Keseluruhan") -> dict:
    """
    Menghitung Confusion Matrix (3x3), Accuracy, Precision (Macro & Per Kelas),
    Recall (Macro & Per Kelas), F1-Score (Macro & Per Kelas), serta Jumlah Data Uji.
    """
    labels = SENTIMENT_LABELS
    confusion = {actual: {pred: 0 for pred in labels} for actual in labels}
    for actual, pred in zip(y_true, y_pred):
        if actual in confusion and pred in confusion[actual]:
            confusion[actual][pred] += 1

    n_samples = len(y_true)
    tp_total = sum(confusion[lbl][lbl] for lbl in labels)
    accuracy = tp_total / n_samples if n_samples > 0 else 0.0

    per_class = {}
    prec_list, rec_list, f1_list = [], [], []
    weighted_prec, weighted_rec, weighted_f1 = 0.0, 0.0, 0.0

    for lbl in labels:
        tp = confusion[lbl][lbl]
        fp = sum(confusion[other][lbl] for other in labels if other != lbl)
        fn = sum(confusion[lbl][other] for other in labels if other != lbl)
        support = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[lbl] = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        if n_samples > 0:
            weighted_prec += prec * (support / n_samples)
            weighted_rec += rec * (support / n_samples)
            weighted_f1 += f1 * (support / n_samples)

    macro_precision = sum(prec_list) / len(labels)
    macro_recall = sum(rec_list) / len(labels)
    macro_f1 = sum(f1_list) / len(labels)

    return {
        "scope": scope_name,
        "n_test": n_samples,
        "tp_total": tp_total,
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        "confusion": confusion,
        "per_class": per_class,
    }


# =========================================================
# 6. TABEL DISTRIBUSI 4 ASPEK (SEBELUM BALANCING) & TABEL BALANCING
# =========================================================

def build_four_aspect_distribution_table(df_before_balancing: pd.DataFrame) -> pd.DataFrame:
    """
    Acuan Revisi Poin 7:
    Membuat tabel distribusi 4 aspek dengan kolom:
    Aspek | Positif | Negatif | Netral | Total
    menggunakan data SEBELUM penyeimbangan.
    """
    rows = []
    for aspect in ASPECT_CATEGORIES:
        sub = df_before_balancing[df_before_balancing["aspect"] == aspect]
        vc = sub[LABEL_COLUMN].value_counts()
        pos = int(vc.get("positif", 0))
        neg = int(vc.get("negatif", 0))
        neu = int(vc.get("netral", 0))
        tot = pos + neg + neu
        rows.append({
            "Aspek": aspect,
            "Positif": pos,
            "Negatif": neg,
            "Netral": neu,
            "Total": tot,
        })

    summary_df = pd.DataFrame(rows)
    total_row = pd.DataFrame([{
        "Aspek": "Total Keseluruhan",
        "Positif": int(summary_df["Positif"].sum()),
        "Negatif": int(summary_df["Negatif"].sum()),
        "Netral": int(summary_df["Netral"].sum()),
        "Total": int(summary_df["Total"].sum()),
    }])
    return pd.concat([summary_df, total_row], ignore_index=True)


def build_balancing_experiment_tables(
    full_df: pd.DataFrame,
    train_raw_df: pd.DataFrame,
    train_bal_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Acuan Revisi Poin 7:
    Menampilkan jumlah data sebelum dan sesudah penyeimbangan secara terpisah
    untuk menjelaskan eksperimen bahwa balancing HANYA dilakukan pada data latih (80%).
    """
    def count_sent(d: pd.DataFrame) -> dict[str, int]:
        vc = d[LABEL_COLUMN].value_counts()
        return {
            "Positif": int(vc.get("positif", 0)),
            "Negatif": int(vc.get("negatif", 0)),
            "Netral": int(vc.get("netral", 0)),
            "Total": len(d),
        }

    c_full = count_sent(full_df)
    c_tr_raw = count_sent(train_raw_df)
    c_tr_bal = count_sent(train_bal_df)
    c_test = count_sent(test_df)

    overview_rows = [
        {
            "Kelompok_Data": "1. Seluruh Pasangan Komentar-Aspek (100% - Sebelum Split & Balancing)",
            "Komentar_Unik": int(full_df["Comment_ID"].nunique()),
            **c_full,
        },
        {
            "Kelompok_Data": "2. Data Latih (80% - Sebelum Balancing)",
            "Komentar_Unik": int(train_raw_df["Comment_ID"].nunique()),
            **c_tr_raw,
        },
        {
            "Kelompok_Data": "3. Data Latih (80% - Sesudah Balancing / Oversampling)",
            "Komentar_Unik": int(train_bal_df["Comment_ID"].nunique()),
            **c_tr_bal,
        },
        {
            "Kelompok_Data": "4. Data Uji (20% - Alami / Tanpa Balancing)",
            "Komentar_Unik": int(test_df["Comment_ID"].nunique()),
            **c_test,
        },
    ]
    overview_df = pd.DataFrame(overview_rows)

    # Rincian per aspek pada Data Latih (Sebelum vs Sesudah Balancing) & Data Uji
    aspect_exp_rows = []
    for aspect in ASPECT_CATEGORIES:
        tr_sub_raw = train_raw_df[train_raw_df["aspect"] == aspect]
        tr_sub_bal = train_bal_df[train_bal_df["aspect"] == aspect]
        te_sub = test_df[test_df["aspect"] == aspect]

        vc_raw = tr_sub_raw[LABEL_COLUMN].value_counts()
        vc_bal = tr_sub_bal[LABEL_COLUMN].value_counts()
        vc_te = te_sub[LABEL_COLUMN].value_counts()

        aspect_exp_rows.append({
            "Aspek": aspect,
            "Latih_Sebelum_Positif": int(vc_raw.get("positif", 0)),
            "Latih_Sebelum_Negatif": int(vc_raw.get("negatif", 0)),
            "Latih_Sebelum_Netral": int(vc_raw.get("netral", 0)),
            "Latih_Sebelum_Total": len(tr_sub_raw),
            "Latih_Sesudah_Positif": int(vc_bal.get("positif", 0)),
            "Latih_Sesudah_Negatif": int(vc_bal.get("negatif", 0)),
            "Latih_Sesudah_Netral": int(vc_bal.get("netral", 0)),
            "Latih_Sesudah_Total": len(tr_sub_bal),
            "Uji_Alami_Positif": int(vc_te.get("positif", 0)),
            "Uji_Alami_Negatif": int(vc_te.get("negatif", 0)),
            "Uji_Alami_Netral": int(vc_te.get("netral", 0)),
            "Uji_Alami_Total": len(te_sub),
        })
    aspect_exp_df = pd.DataFrame(aspect_exp_rows)
    return overview_df, aspect_exp_df


def main():
    print("========================================")
    print("TAHAP 4: SPLIT 80:20, BALANCING LATIH, TF-IDF & NAIVE BAYES")
    print("========================================")

    if not INPUT_PAIRS_FILE.exists():
        raise FileNotFoundError(f"Dataset pasangan komentar-aspek tidak ditemukan: {INPUT_PAIRS_FILE}")

    df = pd.read_excel(INPUT_PAIRS_FILE)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna("netral").astype(str).str.strip().str.lower()

    # 1. Tabel distribusi 4 aspek SEBELUM penyeimbangan
    aspect_dist_df = build_four_aspect_distribution_table(df)
    aspect_dist_df.to_excel(ASPECT_SUMMARY_FILE, index=False)
    aspect_dist_df.to_csv(
        VISUALISASI_DIR / "distribusi_4_aspek_sebelum_balancing.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[1/5] Tabel Distribusi 4 Aspek (Data Sebelum Penyeimbangan):")
    print(aspect_dist_df.to_string(index=False))

    # 2. Split 80:20 berdasarkan Comment_ID SEBELUM penyeimbangan
    train_raw_df, test_df, overlap_proof = split_train_test_by_comment(df, test_size=0.20, random_state=42)
    pd.DataFrame([overlap_proof]).to_csv(OVERLAP_PROOF_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame([overlap_proof]).to_csv(VISUALISASI_DIR / "bukti_tidak_tumpang_tindih.csv", index=False, encoding="utf-8-sig")

    print("\n[2/5] Pembagian Data Latih (80%) & Data Uji (20%) Berdasarkan Comment_ID:")
    for k, v in overlap_proof.items():
        print(f" - {k:40s}: {v}")

    # 3. Penyeimbangan HANYA pada data latih
    train_balanced_df = balance_training_data(train_raw_df, random_state=42)

    train_raw_df.to_excel(TRAIN_RAW_FILE, index=False)
    train_balanced_df.to_excel(TRAIN_BALANCED_FILE, index=False)
    train_balanced_df.to_excel(LABEL_BALANCED_FILE, index=False)
    test_df.to_excel(TEST_FILE, index=False)

    balancing_overview_df, balancing_aspect_df = build_balancing_experiment_tables(
        df, train_raw_df, train_balanced_df, test_df
    )
    with pd.ExcelWriter(BALANCING_EXP_FILE, engine="openpyxl") as writer:
        balancing_overview_df.to_excel(writer, sheet_name="Ringkasan_Balancing", index=False)
        balancing_aspect_df.to_excel(writer, sheet_name="Balancing_Per_Aspek", index=False)

    balancing_overview_df.to_csv(
        VISUALISASI_DIR / "tabel_eksperimen_balancing.csv",
        index=False,
        encoding="utf-8-sig",
    )
    balancing_aspect_df.to_csv(
        VISUALISASI_DIR / "tabel_eksperimen_balancing_per_aspek.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[3/5] Tabel Eksperimen Penyeimbangan Data Latih (Data Uji Tidak Di-balancing):")
    print(balancing_overview_df.to_string(index=False))

    # 4. Verifikasi Rumus TF-IDF Bab III (Tabel 3.12) & Pembentukan Kosakata + IDF dari Data Latih
    tfidf_ver_df = verify_tfidf_formula_against_bab3()
    vocab, idf, df_counts = compute_tfidf_from_train(train_balanced_df[TEXT_COLUMN].tolist())
    vocab_set = set(vocab)
    print(f"\n[4/5] Perhitungan TF-IDF (Rumus Bab III Tabel 3.12: TF=f/|d|, IDF=log10(N/df) tanpa +1):")
    print(tfidf_ver_df.to_string(index=False))
    print(f" - Jumlah dokumen latih (N) : {len(train_balanced_df)}")
    print(f" - Ukuran kosakata latih |V|: {len(vocab)} kata unik")

    # Latih Multinomial Naive Bayes
    priors, class_term_counts, class_term_totals, class_counts = train_multinomial_nb(
        train_balanced_df, vocab, idf
    )

    # 5. Evaluasi pada Data Uji (20%) Keseluruhan dan Per Aspek
    y_true = test_df[LABEL_COLUMN].tolist()
    y_pred = [
        predict_multinomial_nb(
            text=t,
            priors=priors,
            class_term_counts=class_term_counts,
            class_term_totals=class_term_totals,
            vocab=vocab,
            vocab_set=vocab_set,
            idf=idf,
        )
        for t in test_df[TEXT_COLUMN].tolist()
    ]

    test_pred_df = test_df.copy()
    test_pred_df["predicted_sentiment"] = y_pred
    test_pred_df["is_correct"] = test_pred_df[LABEL_COLUMN] == test_pred_df["predicted_sentiment"]
    test_pred_df["Status_Prediksi"] = test_pred_df["is_correct"].map({True: " Sesuai", False: " Tidak Sesuai"}).str.strip()
    test_pred_df.to_excel(TEST_PREDICTIONS_FILE, index=False)

    overall_eval = compute_evaluation_metrics(y_true, y_pred, scope_name="Keseluruhan (4 Aspek)")

    print("\n========================================")
    print("[5/5] HASIL EVALUASI MODEL NAIVE BAYES PADA DATA UJI (20%)")
    print("========================================")
    print(f"Jumlah Data Uji (Pasangan Komentar-Aspek) : {overall_eval['n_test']} (dari {test_df['Comment_ID'].nunique()} komentar unik)")
    print(f"Prediksi Benar                            : {overall_eval['tp_total']} / {overall_eval['n_test']}")
    print(f"Accuracy                                  : {overall_eval['accuracy'] * 100:.2f}% ({overall_eval['accuracy']:.4f})")
    print(f"Precision (Macro)                         : {overall_eval['precision'] * 100:.2f}% ({overall_eval['precision']:.4f})")
    print(f"Recall (Macro)                            : {overall_eval['recall'] * 100:.2f}% ({overall_eval['recall']:.4f})")
    print(f"F1-Score (Macro)                          : {overall_eval['f1_score'] * 100:.2f}% ({overall_eval['f1_score']:.4f})")

    print("\nConfusion Matrix Keseluruhan (Baris = Aktual, Kolom = Prediksi):")
    cm_overall_rows = []
    header_str = "Aktual \\ Prediksi\t" + "\t".join(lbl.capitalize() for lbl in SENTIMENT_LABELS)
    print(header_str)
    for actual in SENTIMENT_LABELS:
        row_vals = [overall_eval["confusion"][actual][pred] for pred in SENTIMENT_LABELS]
        print(f"{actual.capitalize():16s}\t" + "\t".join(str(v) for v in row_vals))
        cm_overall_rows.append({
            "Aktual": actual.capitalize(),
            "Prediksi_Positif": overall_eval["confusion"][actual]["positif"],
            "Prediksi_Negatif": overall_eval["confusion"][actual]["negatif"],
            "Prediksi_Netral": overall_eval["confusion"][actual]["netral"],
            "Total_Aktual": sum(row_vals),
        })
    cm_overall_df = pd.DataFrame(cm_overall_rows)
    cm_overall_df.to_csv(VISUALISASI_DIR / "confusion_matrix_overall.csv", index=False, encoding="utf-8-sig")

    # Evaluasi Per Kelas Sentimen
    per_class_rows = []
    for lbl in SENTIMENT_LABELS:
        pc = overall_eval["per_class"][lbl]
        per_class_rows.append({
            "Kelas_Sentimen": lbl.capitalize(),
            "Precision": round(pc["precision"], 4),
            "Recall": round(pc["recall"], 4),
            "F1_Score": round(pc["f1_score"], 4),
            "Jumlah_Data_Uji": pc["support"],
        })
    per_class_df = pd.DataFrame(per_class_rows)

    # Evaluasi Per Aspek
    per_aspect_rows = []
    cm_aspect_rows = []

    print("\nEvaluasi Per Kategori Aspek:")
    for aspect in ASPECT_CATEGORIES:
        sub_df = test_pred_df[test_pred_df["aspect"] == aspect]
        ev_asp = compute_evaluation_metrics(
            sub_df[LABEL_COLUMN].tolist(),
            sub_df["predicted_sentiment"].tolist(),
            scope_name=aspect,
        )
        per_aspect_rows.append({
            "Aspek": aspect,
            "Jumlah_Data_Uji": ev_asp["n_test"],
            "Prediksi_Benar": ev_asp["tp_total"],
            "Accuracy": round(ev_asp["accuracy"], 4),
            "Precision": round(ev_asp["precision"], 4),
            "Recall": round(ev_asp["recall"], 4),
            "F1_Score": round(ev_asp["f1_score"], 4),
            "Accuracy_Pct": round(ev_asp["accuracy"] * 100, 2),
            "Precision_Pct": round(ev_asp["precision"] * 100, 2),
            "Recall_Pct": round(ev_asp["recall"] * 100, 2),
            "F1_Score_Pct": round(ev_asp["f1_score"] * 100, 2),
        })
        print(
            f" - {aspect:26s} | N_uji={ev_asp['n_test']:3d} | "
            f"Acc={ev_asp['accuracy']*100:6.2f}% | Prec={ev_asp['precision']*100:6.2f}% | "
            f"Rec={ev_asp['recall']*100:6.2f}% | F1={ev_asp['f1_score']*100:6.2f}%"
        )
        for actual in SENTIMENT_LABELS:
            cm_aspect_rows.append({
                "Aspek": aspect,
                "Aktual": actual.capitalize(),
                "Prediksi_Positif": ev_asp["confusion"][actual]["positif"],
                "Prediksi_Negatif": ev_asp["confusion"][actual]["negatif"],
                "Prediksi_Netral": ev_asp["confusion"][actual]["netral"],
                "Total_Aktual": sum(ev_asp["confusion"][actual].values()),
            })

    # Tambahkan baris Keseluruhan ke tabel evaluasi per aspek
    per_aspect_rows.append({
        "Aspek": "Keseluruhan (Total)",
        "Jumlah_Data_Uji": overall_eval["n_test"],
        "Prediksi_Benar": overall_eval["tp_total"],
        "Accuracy": round(overall_eval["accuracy"], 4),
        "Precision": round(overall_eval["precision"], 4),
        "Recall": round(overall_eval["recall"], 4),
        "F1_Score": round(overall_eval["f1_score"], 4),
        "Accuracy_Pct": round(overall_eval["accuracy"] * 100, 2),
        "Precision_Pct": round(overall_eval["precision"] * 100, 2),
        "Recall_Pct": round(overall_eval["recall"] * 100, 2),
        "F1_Score_Pct": round(overall_eval["f1_score"] * 100, 2),
    })

    per_aspect_df = pd.DataFrame(per_aspect_rows)
    cm_aspect_df = pd.DataFrame(cm_aspect_rows)

    per_aspect_df.to_csv(VISUALISASI_DIR / "evaluation_per_aspect.csv", index=False, encoding="utf-8-sig")
    cm_aspect_df.to_csv(VISUALISASI_DIR / "confusion_matrix_per_aspect.csv", index=False, encoding="utf-8-sig")

    # Simpan evaluation_metrics.csv untuk grafik keseluruhan
    metrics_df = pd.DataFrame([
        {"metric": "Accuracy", "score": round(overall_eval["accuracy"], 4), "n_test": overall_eval["n_test"]},
        {"metric": "Precision", "score": round(overall_eval["precision"], 4), "n_test": overall_eval["n_test"]},
        {"metric": "Recall", "score": round(overall_eval["recall"], 4), "n_test": overall_eval["n_test"]},
        {"metric": "F1-Score", "score": round(overall_eval["f1_score"], 4), "n_test": overall_eval["n_test"]},
    ])
    metrics_df.to_csv(VISUALISASI_DIR / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")

    # Pilih sampel komentar representatif (mencakup ke-4 aspek & komentar multi-aspek) untuk diperiksa
    sample_rows = []
    # 1. Ambil contoh komentar multi-aspek di data uji
    test_cid_counts = test_pred_df["Comment_ID"].value_counts()
    multi_cids = test_cid_counts[test_cid_counts >= 2].index.tolist()
    picked_cids = multi_cids[:6]
    for cid in picked_cids:
        sub = test_pred_df[test_pred_df["Comment_ID"] == cid]
        for _, r in sub.iterrows():
            sample_rows.append(r)

    # 2. Pastikan setiap aspek & setiap kelas sentimen terwakili
    for asp in ASPECT_CATEGORIES:
        for sent in SENTIMENT_LABELS:
            sub = test_pred_df[
                (test_pred_df["aspect"] == asp)
                & (test_pred_df[LABEL_COLUMN] == sent)
                & (~test_pred_df["Comment_ID"].isin(picked_cids))
            ]
            if not sub.empty:
                sample_rows.append(sub.iloc[0])

    sample_inspect_df = pd.DataFrame(sample_rows).drop_duplicates(subset=["Pair_ID"]).reset_index(drop=True)
    sample_export_df = sample_inspect_df[[
        "Pair_ID", "Comment_ID", "Comment Text", "normalized_text",
        "aspect", "matched_keywords", "aspect_context",
        LABEL_COLUMN, "predicted_sentiment", "Status_Prediksi",
    ]].rename(columns={
        "Comment Text": "Komentar_Asli",
        "normalized_text": "Komentar_Normalisasi",
        "aspect": "Aspek",
        "matched_keywords": "Kata_Kunci_Aspek",
        "aspect_context": "Konteks_Klausa_Aspek",
        LABEL_COLUMN: "Label_Acuan_InSet",
        "predicted_sentiment": "Hasil_Prediksi_NB",
    })
    sample_export_df.to_csv(
        VISUALISASI_DIR / "contoh_prediksi_komentar_aspek.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Simpan seluruh tabel evaluasi ke satu berkas Excel multi-sheet
    with pd.ExcelWriter(FULL_EVAL_EXCEL, engine="openpyxl") as writer:
        per_aspect_df.to_excel(writer, sheet_name="Evaluasi_Per_Aspek", index=False)
        per_class_df.to_excel(writer, sheet_name="Evaluasi_Per_Kelas", index=False)
        cm_overall_df.to_excel(writer, sheet_name="Confusion_Matrix_Total", index=False)
        cm_aspect_df.to_excel(writer, sheet_name="Confusion_Matrix_Aspek", index=False)
        aspect_dist_df.to_excel(writer, sheet_name="Distribusi_4_Aspek_Awal", index=False)
        balancing_overview_df.to_excel(writer, sheet_name="Eksperimen_Balancing", index=False)
        pd.DataFrame([overlap_proof]).to_excel(writer, sheet_name="Bukti_Zero_Overlap", index=False)
        sample_export_df.to_excel(writer, sheet_name="Contoh_Prediksi_Uji", index=False)

    # Simpan model Naive Bayes
    model_payload = {
        "vocab": vocab,
        "idf": idf,
        "df_counts": df_counts,
        "priors": priors,
        "class_term_counts": class_term_counts,
        "class_term_totals": class_term_totals,
        "labels": SENTIMENT_LABELS,
        "aspect_categories": ASPECT_CATEGORIES,
        "overlap_proof": overlap_proof,
    }
    with open(MODEL_PKL_FILE, "wb") as f:
        pickle.dump(model_payload, f)

    print(f"\nBerkas hasil klasifikasi & evaluasi disimpan di:")
    print(f" - Distribusi 4 aspek (sebelum balancing) : {ASPECT_SUMMARY_FILE}")
    print(f" - Tabel eksperimen balancing data latih  : {BALANCING_EXP_FILE}")
    print(f" - Evaluasi lengkap Naive Bayes           : {FULL_EVAL_EXCEL}")
    print(f" - Hasil prediksi data uji lengkap        : {TEST_PREDICTIONS_FILE}")
    print(f" - Bukti 0 tumpang tindih latih/uji       : {OVERLAP_PROOF_CSV}")
    print(f" - Model tersimpan                        : {MODEL_PKL_FILE}")
    print("========================================")


if __name__ == "__main__":
    main()
