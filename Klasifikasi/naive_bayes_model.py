import math
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
LABEL_DIR = BASE_DIR / "Label"

# Prioritaskan dataset yang sudah seimbang agar distribusi sentimen tidak didominasi netral
DATASET_FILE = LABEL_DIR / "dataset_labeled_balanced.xlsx"
if not DATASET_FILE.exists():
    DATASET_FILE = LABEL_DIR / "dataset_labeled.xlsx"

TEXT_COLUMN = "aspect_text"
LABEL_COLUMN = "sentiment"

TOPIC_ASPECT_MAP = {
    "perlindungan_anak": [
        "perlindungan anak", "melindungi anak", "menjaga anak", "lindungi anak",
        "keamanan anak", "anak aman", "safety child", "proteksi anak",
        "kesehatan anak", "anak terlindung", "pelindung anak", "safety", "santunan anak"
    ],
    "batas_usia": [
        "batas usia", "umur minimum", "usia minimum", "batas umur", "di bawah 16",
        "dibawah 16", "under 16", "minimum age", "usia max 16", "usia 16",
        "usia anak dibawah 16", "umur anak di bawah 16", "batasan usia",
        "anak umur 16", "anak dibawah umur"
    ],
    "verifikasi_usia": [
        "verifikasi usia", "cek usia", "validasi usia", "konfirmasi usia",
        "usia harus diverifikasi", "verifikasi umur", "age verification",
        "cek umur", "validasi umur", "konfirmasi umur", "check age",
        "konfirmasi umur anak", "cek kelayakan usia"
    ],
    "pengawasan_orangtua": [
        "orang tua", "orangtua", "pengawasan orang tua", "kontrol orang tua",
        "didampingi orang tua", "orang tua harus", "pengawasan ortu",
        "pengawasan orangtua", "orang tua memantau"
    ],
    "konten_negatif": [
        "konten negatif", "konten tidak layak", "konten dewasa", "konten berbahaya",
        "iklan aneh", "video tidak layak", "bahaya", "berbahaya", "hoax",
        "konten tidak sehat", "konten merugikan anak"
    ],
    "kecanduan": [
        "kecanduan", "ketagihan", "ketergantungan", "candu", "terlalu sering",
        "bergantung", "addictive", "ketergantungan media", "menghabiskan waktu"
    ]
}

TOPIC_ASPECT_LABELS = {
    "perlindungan_anak": "Perlindungan Anak",
    "batas_usia": "Batas Usia",
    "verifikasi_usia": "Verifikasi Usia",
    "pengawasan_orangtua": "Pengawasan Orang Tua",
    "konten_negatif": "Konten Negatif",
    "kecanduan": "Kecanduan"
}


def map_topic_aspect(aspect):
    aspect = str(aspect).strip().lower()
    if not aspect:
        return "lainnya"

    for canonical, phrases in TOPIC_ASPECT_MAP.items():
        for phrase in phrases:
            if phrase in aspect:
                return canonical

    return "lainnya"


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_train_test(df, test_size=0.2, random_state=42):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = max(1, int(len(df) * (1 - test_size)))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def compute_tfidf(train_texts):
    docs = []
    vocab = set()
    for text in train_texts:
        tokens = text.split()
        if not tokens:
            continue
        docs.append(tokens)
        vocab.update(tokens)

    vocab = sorted(vocab)
    vocab_index = {term: i for i, term in enumerate(vocab)}
    n_docs = len(docs)
    df_count = defaultdict(int)
    for tokens in docs:
        unique_terms = set(tokens)
        for term in unique_terms:
            df_count[term] += 1

    idf = {}
    for term in vocab:
        idf[term] = math.log((1 + n_docs) / (1 + df_count.get(term, 0))) + 1.0

    tfidf_matrix = []
    for tokens in docs:
        vec = {term: 0.0 for term in vocab}
        total_terms = len(tokens)
        if total_terms == 0:
            tfidf_matrix.append(vec)
            continue
        tf = defaultdict(float)
        for token in tokens:
            tf[token] += 1.0
        for term, count in tf.items():
            term_tf = count / total_terms
            vec[term] = term_tf * idf.get(term, 1.0)
        tfidf_matrix.append(vec)

    return vocab, idf, tfidf_matrix


def vectorize_text(text, vocab, idf):
    tokens = text.split()
    if not tokens:
        return {term: 0.0 for term in vocab}
    total_terms = len(tokens)
    vec = {term: 0.0 for term in vocab}
    tf = defaultdict(float)
    for token in tokens:
        tf[token] += 1.0
    for term, count in tf.items():
        if term in vocab:
            vec[term] = (count / total_terms) * idf.get(term, 1.0)
    return vec


def train_nb(train_df, vocab, idf):
    class_counts = defaultdict(int)
    class_term_totals = defaultdict(float)
    class_term_counts = defaultdict(dict)

    for _, row in train_df.iterrows():
        label = row[LABEL_COLUMN]
        text = row[TEXT_COLUMN]
        vec = vectorize_text(text, vocab, idf)
        class_counts[label] += 1

        inner = class_term_counts.setdefault(label, {})
        for term, weight in vec.items():
            if weight <= 0:
                continue
            inner[term] = inner.get(term, 0.0) + weight
            class_term_totals[label] += weight

    total_docs = len(train_df)
    priors = {label: math.log(class_counts[label] / total_docs) for label in class_counts}

    return priors, class_term_counts, class_term_totals, class_counts


def predict_text(text, priors, class_term_counts, class_term_totals, vocab, idf, labels):
    vec = vectorize_text(text, vocab, idf)
    scores = {}

    for label in labels:
        score = priors[label]
        denom = class_term_totals[label] + len(vocab)

        for term, weight in vec.items():
            if weight <= 0:
                continue
            count = class_term_counts[label].get(term, 0.0)
            score += weight * math.log((count + 1.0) / denom)

        scores[label] = score

    return max(scores, key=scores.get)


def evaluate(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    confusion = {label: {target: 0 for target in labels} for label in labels}
    for actual, pred in zip(y_true, y_pred):
        confusion[actual][pred] = confusion[actual].get(pred, 0) + 1

    total = len(y_true)
    tp_total = 0
    for label in labels:
        tp_total += confusion[label].get(label, 0)

    accuracy = tp_total / total if total else 0.0

    print("\nConfusion Matrix:")
    print("\t" + "\t".join(labels))
    for label in labels:
        row = [confusion[label].get(target, 0) for target in labels]
        print(label, "\t" + "\t".join(map(str, row)))

    print("\nAccuracy:", round(accuracy, 4))

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    for label in labels:
        tp = confusion[label].get(label, 0)
        predicted_total = sum(confusion[target].get(label, 0) for target in labels)
        actual_total = sum(confusion[label].get(target, 0) for target in labels)

        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / actual_total if actual_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    macro_precision = total_precision / len(labels) if labels else 0.0
    macro_recall = total_recall / len(labels) if labels else 0.0
    macro_f1 = total_f1 / len(labels) if labels else 0.0

    print(f"Precision (macro): {macro_precision:.4f}")
    print(f"Recall (macro): {macro_recall:.4f}")
    print(f"F1-Score (macro): {macro_f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "labels": labels,
    }


def parse_aspects(value):
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    separators = [",", ";", "|", "\n"]
    for sep in separators:
        text = text.replace(sep, ",")

    aspects = []
    for part in text.split(","):
        aspect = part.strip().lower()
        if aspect:
            aspects.append(aspect)
    return aspects


def summarize_by_aspect(df):
    records = []
    for _, row in df.iterrows():
        aspect_text = row.get(TEXT_COLUMN, "")
        sentiment = str(row.get(LABEL_COLUMN, "netral")).strip().lower()

        if sentiment not in {"positif", "netral"}:
            continue

        aspects = parse_aspects(aspect_text)
        if not aspects:
            continue

        for aspect in aspects:
            mapped_aspect = map_topic_aspect(aspect)
            if mapped_aspect in TOPIC_ASPECT_LABELS:
                records.append({"aspect": mapped_aspect, "sentiment": sentiment})

    if not records:
        return pd.DataFrame(columns=["Aspek", "Positif", "Netral", "N Aspek"])

    aspect_df = pd.DataFrame(records)
    summary = (
        aspect_df.groupby("aspect", as_index=False)["sentiment"]
        .value_counts()
        .pivot(index="aspect", columns="sentiment", values="count")
        .fillna(0)
        .astype(int)
    )

    for label in ["positif", "netral"]:
        if label not in summary.columns:
            summary[label] = 0

    summary = summary[["positif", "netral"]].reset_index().rename(columns={"aspect": "Aspek"})
    summary["Aspek"] = summary["Aspek"].map(TOPIC_ASPECT_LABELS)
    summary = summary.rename(columns={"positif": "Positif", "netral": "Netral"})
    summary["N Aspek"] = summary[["Positif", "Netral"]].sum(axis=1)
    summary = summary[["Aspek", "Positif", "Netral", "N Aspek"]]
    summary = summary.sort_values(["N Aspek", "Positif", "Netral"], ascending=False).reset_index(drop=True)
    return summary


def save_evaluation_csv(metrics):
    metrics_df = pd.DataFrame([
        {"metric": "Accuracy", "score": float(metrics.get("accuracy", 0.0))},
        {"metric": "Precision", "score": float(metrics.get("precision", 0.0))},
        {"metric": "Recall", "score": float(metrics.get("recall", 0.0))},
        {"metric": "F1-Score", "score": float(metrics.get("f1_score", 0.0))},
    ])

    output_metrics_path = BASE_DIR / "Visualisasi" / "evaluation_metrics.csv"
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_metrics_path, index=False)
    print(f"\nHasil evaluasi disimpan di: {output_metrics_path}")
    return output_metrics_path


def main():
    dataset_path = DATASET_FILE
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_path}")

    df = pd.read_excel(dataset_path)
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(f"Kolom {TEXT_COLUMN} atau {LABEL_COLUMN} tidak tersedia.")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").apply(clean_text)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna("netral").astype(str).str.strip().str.lower()
    df = df[df[TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)

    train_df, test_df = split_train_test(df, test_size=0.2, random_state=42)

    vocab, idf, _ = compute_tfidf(train_df[TEXT_COLUMN].tolist())
    priors, class_term_counts, class_term_totals, class_counts = train_nb(train_df, vocab, idf)

    labels = sorted(class_counts.keys())
    y_pred = []
    for text in test_df[TEXT_COLUMN].tolist():
        if not text.strip():
            y_pred.append("netral")
            continue
        y_pred.append(predict_text(text, priors, class_term_counts, class_term_totals, vocab, idf, labels))

    y_true = test_df[LABEL_COLUMN].tolist()

    print("\n========================================")
    print("TF-IDF + NAIVE BAYES")
    print("========================================")
    print(f"Jumlah dataset total: {len(df)}")
    print(f"Train set: {len(train_df)}")
    print(f"Test set: {len(test_df)}")
    print(f"Rasio split: 80:20")
    metrics = evaluate(y_true, y_pred)
    save_evaluation_csv(metrics)

    aspect_summary = summarize_by_aspect(df)
    if not aspect_summary.empty:
        print("\n========================================")
        print("RINGKASAN ASPEK SENTIMEN")
        print("========================================")
        print(aspect_summary.to_string(index=False))

        summary_path = BASE_DIR / "Klasifikasi" / "aspect_sentiment_summary.xlsx"
        if summary_path.exists():
            try:
                summary_path.unlink()
            except PermissionError:
                pass
        aspect_summary.to_excel(summary_path, index=False)
        print(f"\nHasil per aspek tersimpan di: {summary_path}")

    model_path = BASE_DIR / "Klasifikasi" / "naive_bayes_model.pkl"
    payload = {
        "vocab": vocab,
        "idf": idf,
        "priors": priors,
        "class_term_counts": class_term_counts,
        "class_term_totals": class_term_totals,
        "labels": labels,
    }

    with open(model_path, "wb") as f:
        import pickle
        pickle.dump(payload, f)

    print(f"\nModel tersimpan di: {model_path}")


if __name__ == "__main__":
    main()
