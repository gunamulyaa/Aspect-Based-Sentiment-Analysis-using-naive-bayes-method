# -*- coding: utf-8 -*-
"""
Kamus InSet (Indonesian Sentiment Lexicon) Asli & Pemetaan 4 Kategori Aspek Terketat
Acuan Revisi Bab I-III:
1. Menggunakan Kamus InSet dari sumber aslinya:
   Fajri Koto & Gemala Y. Rahmaningtyas (2017), "InSet Lexicon: Evaluation of a Word List for
   Indonesian Sentiment Analysis in Microblogs", IEEE IALP 2017.
   Repositori resmi: https://github.com/fajri91/InSet
   - Label/inset_original_positive.tsv (3.609 entri baris positif)
   - Label/inset_original_negative.tsv (6.609 entri baris negatif)
   Setiap modifikasi bobot (penyelesaian 1.142 kata ganda/overlap positif-negatif,
   penetralan kata benda/fungsi domain, penanganan operator negasi) dan penambahan
   kata dasar/stem Sastrawi maupun slang domain dicatat secara terpisah dan transparan
   pada berkas `Label/inset_modifikasi_dan_tambahan.csv` dan `.xlsx`.
2. Mempertahankan kata negasi (tidak, bukan, jangan, belum, tanpa, kurang, tak)
   sebagai operator pembalik polaritas dan fitur eksplisit hingga ke masukan Naive Bayes.
3. Memperketat aturan pemetaan 4 kategori aspek (Perlindungan Anak, Peran Orang Tua,
   Peran Pemerintah/Komdigi, Efektivitas Kebijakan):
   Kata ambigu seperti "hp", "beli hp", "youtube", "tiktok", "roblox", "game", "sosmed",
   "akun", "sekolah", "rumah", "ibu", "bapak", "umur", "usia" TIDAK langsung menentukan
   kategori aspek tanpa konteks relevan yang sesuai dengan definisi aspek pada Bab I-III.
"""

import csv
import re
from pathlib import Path
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent

# ==============================================================================
# DOKUMENTASI SUMBER ASLI KAMUS INSET (POIN 1 REVISI)
# ==============================================================================

INSET_SOURCE_METADATA = {
    "nama_kamus": "InSet (Indonesian Sentiment Lexicon)",
    "penulis": "Fajri Koto & Gemala Y. Rahmaningtyas",
    "tahun": 2017,
    "publikasi": (
        "InSet Lexicon: Evaluation of a Word List for Indonesian Sentiment Analysis "
        "in Microblogs, International Conference on Asian Language Processing (IALP), IEEE, 2017"
    ),
    "url_repositori": "https://github.com/fajri91/InSet",
    "url_positive_tsv": "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv",
    "url_negative_tsv": "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv",
    "file_lokal_positif": "Label/inset_original_positive.tsv",
    "file_lokal_negatif": "Label/inset_original_negative.tsv",
    "jumlah_entri_asli_positif": 3609,
    "jumlah_entri_asli_negatif": 6609,
}

ORIGINAL_POS_TSV = CURRENT_DIR / "inset_original_positive.tsv"
ORIGINAL_NEG_TSV = CURRENT_DIR / "inset_original_negative.tsv"


def load_original_inset_lexicon() -> tuple[dict[str, int], dict[str, int]]:
    """
    Memuat kamus InSet langsung dari berkas TSV sumber aslinya (Fajri Koto & Rahmaningtyas, 2017):
    - inset_original_positive.tsv (3.609 baris, bobot +1 s.d. +5)
    - inset_original_negative.tsv (6.609 baris, bobot -1 s.d. -5)
    """
    orig_pos: dict[str, int] = {}
    orig_neg: dict[str, int] = {}

    if ORIGINAL_POS_TSV.exists():
        with open(ORIGINAL_POS_TSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                w = str(row.get("word", "")).strip().lower()
                if w:
                    orig_pos[w] = int(float(row["weight"]))

    if ORIGINAL_NEG_TSV.exists():
        with open(ORIGINAL_NEG_TSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                w = str(row.get("word", "")).strip().lower()
                if w:
                    orig_neg[w] = int(float(row["weight"]))

    return orig_pos, orig_neg


inset_original_positive, inset_original_negative = load_original_inset_lexicon()


# ==============================================================================
# DAFTAR MODIFIKASI & TAMBAHAN PADA KAMUS INSET (DICATAT TERPISAH - POIN 1)
# ==============================================================================
# Catatan Metodologis:
# Kamus InSet asli dibangun secara otomatis dari korpus Twitter umum sehingga:
# (a) Memiliki 1.142 kata yang muncul sekaligus di positive.tsv DAN negative.tsv
#     (contoh: "setuju" +4 & -3, "tuju" +3 & -4, "bagus" +2 & -4, "baik" +3 & -1,
#      "senang" +5 & -4, "benar" +2 & -4, "adil" +4 & -3, "peduli" +4 & -1,
#      "rusak" +3 & -4, "candu" +3 & -3, "percuma" +4 & -4, "malu" +2 & -4).
# (b) Mencantumkan kata negasi ("tidak": -5, "bukan": -3, "jangan": -3, "kurang": -3)
#     dan intensifier ("sangat": +3/-5, "sekali": -1, "banget": +1) sebagai kata berpolaritas,
#     padahal dalam analisis sentimen berbasis aturan negasi, kata negasi berfungsi sebagai
#     operator pembalik polaritas (polarity shifter) dan intensifier sebagai pengali bobot.
# (c) Memberikan bobot sentimen pada kata benda/kata fungsi netral domain kebijakan
#     (contoh: "anak": -3, "tua": -4, "umur": +4/-2, "sekolah": +3, "aplikasi": -4,
#      "tanggal": -4, "atur": +4/-4, "bijak": +3/-4, "batas": -3, "blokir": -3, "larang": -3).
#
# Oleh karena itu, seluruh kamus InSet asli (3.609 positif & 6.609 negatif) tetap dimuat
# sebagai kamus dasar, dan setiap penyesuaian/tambahan dicatat secara eksplisit di bawah ini.

# 1. Kata yang dinetralkan (bobot = 0) karena merupakan operator negasi/intensifier,
#    kata fungsi/konjungsi, atau kata benda/istilah domain netral yang bukan opini:
NEUTRALIZED_INSET_WORDS: dict[str, str] = {
    # Operator negasi & intensifier (diproses khusus oleh aturan negasi & intensifier)
    "tidak": "Operator negasi (pembalik polaritas), bukan kata opini mandiri",
    "bukan": "Operator negasi (pembalik polaritas), bukan kata opini mandiri",
    "jangan": "Operator negasi/prohibisi (pembalik polaritas), bukan kata opini mandiri",
    "belum": "Operator negasi temporal, bukan kata opini mandiri",
    "tanpa": "Operator negasi preposisional, bukan kata opini mandiri",
    "kurang": "Operator negasi/pelemah derajat, bukan kata opini mandiri",
    "tak": "Operator negasi, bukan kata opini mandiri",
    "gak": "Operator negasi slang, bukan kata opini mandiri",
    "ga": "Operator negasi slang, bukan kata opini mandiri",
    "gk": "Operator negasi slang, bukan kata opini mandiri",
    "nggak": "Operator negasi slang, bukan kata opini mandiri",
    "enggak": "Operator negasi slang, bukan kata opini mandiri",
    "kgk": "Operator negasi slang, bukan kata opini mandiri",
    "ora": "Operator negasi daerah, bukan kata opini mandiri",
    "sangat": "Intensifier (pengali bobot 1.5x), bukan kata opini mandiri",
    "banget": "Intensifier (pengali bobot 1.5x), bukan kata opini mandiri",
    "sekali": "Intensifier (pengali bobot 1.5x), bukan kata opini mandiri",
    "amat": "Intensifier (pengali bobot 1.5x), bukan kata opini mandiri",
    "paling": "Intensifier (pengali bobot 1.5x), bukan kata opini mandiri",
    "terlalu": "Intensifier/adverbia derajat, bukan kata opini mandiri",
    "makin": "Adverbia derajat netral",

    # Entitas & istilah domain kebijakan (kata benda/kerja topik yang netral secara leksikal)
    "anak": "Entitas domain utama (Perlindungan Anak), netral secara leksikal",
    "bocil": "Entitas domain (anak kecil), netral secara leksikal",
    "bocah": "Entitas domain (anak), netral secara leksikal",
    "remaja": "Entitas domain (kelompok usia), netral secara leksikal",
    "dewasa": "Entitas domain (kelompok usia), netral secara leksikal",
    "bayi": "Entitas domain (kelompok usia), netral secara leksikal",
    "abg": "Entitas domain (kelompok usia), netral secara leksikal",
    "adek": "Entitas domain (adik/anak), netral secara leksikal",
    "adik": "Entitas domain (adik/anak), netral secara leksikal",
    "generasi": "Entitas domain (generasi muda), netral secara leksikal",
    "siswa": "Entitas domain (pelajar), netral secara leksikal",
    "pelajar": "Entitas domain (pelajar), netral secara leksikal",
    "sekolah": "Entitas domain pendidikan, netral secara leksikal",
    "kelas": "Entitas domain pendidikan, netral secara leksikal",
    "ajar": "Kata dasar 'belajar/mengajar' (aktivitas pendidikan), netral secara leksikal",
    "didik": "Kata dasar 'mendidik/pendidikan', netral secara leksikal",
    "bimbing": "Kata dasar 'membimbing', netral secara leksikal",
    "latih": "Kata dasar 'melatih/latihan', netral secara leksikal",
    "guru": "Entitas domain pendidik, netral secara leksikal",
    "ilmu": "Entitas domain pendidikan, netral secara leksikal",
    "buku": "Entitas benda pendidikan, netral secara leksikal",
    " pustaka": "Entitas pendidikan, netral secara leksikal",
    "tugas": "Entitas akademik (tugas sekolah), netral secara leksikal",
    "sains": "Entitas bidang ilmu, netral secara leksikal",
    "tua": "Bagian frasa 'orang tua' / usia, netral secara leksikal",
    "muda": "Kelompok usia, netral secara leksikal",
    "kecil": "Bagian frasa 'anak kecil', netral secara leksikal",
    "besar": "Ukuran/usia netral",
    "gede": "Ukuran/usia netral",
    "umur": "Atribut demografis batas usia kebijakan, netral secara leksikal",
    "usia": "Atribut demografis batas usia kebijakan, netral secara leksikal",
    "tahun": "Satuan waktu/usia (16 tahun), netral secara leksikal",
    "taun": "Variasi ejaan 'tahun', netral secara leksikal",
    "bulan": "Satuan waktu, netral secara leksikal",
    "tanggal": "Bagian frasa 'tanggal lahir', netral secara leksikal",
    "lahir": "Bagian frasa 'tahun/tanggal lahir', netral secara leksikal",
    "ibu": "Entitas domain (Peran Orang Tua / Menteri), netral secara leksikal",
    "bapak": "Entitas domain (Peran Orang Tua), netral secara leksikal",
    "bapa": "Entitas domain (Peran Orang Tua), netral secara leksikal",
    "ayah": "Entitas domain (Peran Orang Tua), netral secara leksikal",
    "mama": "Entitas domain (Peran Orang Tua), netral secara leksikal",
    "mamak": "Entitas domain (Peran Orang Tua), netral secara leksikal",
    "mamah": "Entitas domain (Peran Orang Tua), netral secara leksikal",
    "bunda": "Entitas domain (Peran Orang Tua), netral secara leksikal",
    "keluarga": "Entitas domain (Peran Orang Tua), netral secara leksikal",
    "rumah": "Entitas lokasi/keluarga, netral secara leksikal",
    "kakek": "Entitas anggota keluarga, netral secara leksikal",
    "awas": "Kata dasar 'mengawasi/pengawasan', netral secara leksikal",
    "pantau": "Kata dasar 'memantau/pemantauan', netral secara leksikal",
    "damping": "Kata dasar 'mendampingi/pendampingan', netral secara leksikal",
    "asuh": "Kata dasar 'mengasuh/pengasuhan', netral secara leksikal",
    "kawal": "Kata dasar 'mengawal', netral secara leksikal",
    "pemerintah": "Entitas domain (Peran Pemerintah/Komdigi), netral secara leksikal",
    "perintah": "Hasil stem Sastrawi untuk 'pemerintah', netral secara leksikal",
    "presiden": "Entitas pejabat negara, netral secara leksikal",
    "menteri": "Entitas pejabat negara, netral secara leksikal",
    "jabat": "Hasil stem Sastrawi untuk 'pejabat', netral secara leksikal",
    "aparat": "Entitas negara, netral secara leksikal",
    "negara": "Entitas domain kebijakan, netral secara leksikal",
    "indo": "Singkatan Indonesia, netral secara leksikal",
    "politik": "Entitas bidang, netral secara leksikal",
    "hukum": "Entitas regulasi/hukum, netral secara leksikal",
    "pasal": "Entitas regulasi, netral secara leksikal",
    "ayat": "Entitas regulasi, netral secara leksikal",
    "undang": "Bagian frasa 'undang-undang', netral secara leksikal",
    "aturan": "Entitas kebijakan, netral secara leksikal",
    "atur": "Kata dasar 'aturan/peraturan/mengatur', netral secara leksikal",
    "bijak": "Hasil stem Sastrawi untuk 'kebijakan', netral secara leksikal",
    "wacana": "Istilah kebijakan netral",
    "prosedur": "Istilah teknis kebijakan netral",
    "metode": "Istilah teknis netral",
    "sistem": "Istilah teknis sistem verifikasi/platform, netral secara leksikal",
    "proses": "Istilah teknis netral",
    "operasi": "Istilah teknis netral",
    "uji": "Bagian frasa 'uji coba', netral secara leksikal",
    "tahap": "Bagian frasa 'bertahap' (mekanisme implementasi), netral secara leksikal",
    "implementasi": "Istilah teknis penerapan kebijakan, netral secara leksikal",
    "terap": "Kata dasar 'penerapan/menerapkan', netral secara leksikal",
    "efek": "Kata benda netral (dampak/efek)",
    "dampak": "Kata benda netral (dampak positif/negatif ditentukan kata sifatnya)",
    "imbas": "Kata benda netral",
    "akibat": "Kata hubung/benda kausalitas netral",
    "sebab": "Kata hubung/benda kausalitas netral",
    "faktor": "Kata benda netral",
    "risiko": "Kata benda netral",
    "konsekuensi": "Kata benda netral",
    "sosial": "Bagian frasa 'media sosial', netral secara leksikal",
    "aplikasi": "Entitas platform digital, netral secara leksikal",
    "game": "Entitas permainan digital, netral secara leksikal",
    "main": "Aktivitas 'bermain/main medsos', netral secara leksikal",
    "maen": "Variasi slang 'main', netral secara leksikal",
    "mainin": "Variasi slang 'memainkan', netral secara leksikal",
    "akses": "Istilah teknis mengakses platform, netral secara leksikal",
    "aktif": "Bagian istilah 'akun aktif/nonaktif', netral secara leksikal",
    "aktivasi": "Istilah teknis netral",
    "daftar": "Aktivitas mendaftar akun, netral secara leksikal",
    "masuk": "Aktivitas login/masuk akun, netral secara leksikal",
    "buka": "Aktivitas membuka aplikasi, netral secara leksikal",
    "tutup": "Aktivitas menutup aplikasi, netral secara leksikal",
    "hapus": "Tindakan teknis menghapus/menonaktifkan akun, netral secara leksikal",
    "blokir": "Tindakan teknis pemblokiran pada topik kebijakan, netral tanpa opini/negasi",
    "block": "Istilah teknis pemblokiran, netral tanpa opini/negasi",
    "batas": "Tindakan teknis pembatasan usia medsos, netral tanpa opini/negasi",
    "larang": "Tindakan regulasi pelarangan usia <16 tahun, netral tanpa opini/negasi",
    "henti": "Tindakan menghentikan, netral secara leksikal",
    "renti": "Variasi 'berhenti', netral secara leksikal",
    "stop": "Tindakan menghentikan, netral secara leksikal",
    "cabut": "Tindakan mencabut, netral secara leksikal",
    "suspend": "Istilah teknis penangguhan akun, netral secara leksikal",
    "tonton": "Aktivitas menonton video, netral secara leksikal",
    "tayang": "Kata dasar 'tayangan', netral secara leksikal",
    "live": "Fitur siaran langsung medsos, netral secara leksikal",
    "chat": "Fitur pesan medsos, netral secara leksikal",
    "pesan": "Fitur pesan/komunikasi, netral secara leksikal",
    "komunikasi": "Aktivitas komunikasi, netral secara leksikal",
    "interaksi": "Aktivitas interaksi sosial, netral secara leksikal",
    "informasi": "Kata benda informasi, netral secara leksikal",
    "info": "Singkatan informasi, netral secara leksikal",
    "berita": "Kata benda berita, netral secara leksikal",
    "artikel": "Kata benda artikel, netral secara leksikal",
    "data": "Kata benda data/identitas, netral secara leksikal",
    "fakta": "Kata benda fakta, netral secara leksikal",
    "bukti": "Kata benda bukti, netral secara leksikal",
    "foto": "Kata benda foto, netral secara leksikal",
    "gambar": "Kata benda gambar, netral secara leksikal",
    "rekam": "Kata kerja merekam, netral secara leksikal",
    "suara": "Kata benda suara, netral secara leksikal",
    "bahasa": "Kata benda bahasa, netral secara leksikal",
    "teks": "Kata benda teks, netral secara leksikal",
    "judul": "Kata benda judul, netral secara leksikal",
    "karya": "Kata benda karya, netral secara leksikal",
    "desain": "Kata benda desain, netral secara leksikal",
    "musik": "Kata benda musik, netral secara leksikal",
    "lagu": "Kata benda lagu, netral secara leksikal",
    "novel": "Kata benda novel, netral secara leksikal",
    "cerita": "Kata benda cerita, netral secara leksikal",

    # Kata kerja/benda/fungsi sehari-hari yang netral namun memiliki bobot di InSet Twitter
    "sama": "Kata depan/hubung 'sama dengan/bersama', netral",
    "lagi": "Adverbia 'sedang/kembali', netral",
    "semua": "Numeralia kolektif, netral",
    "esok": "Keterangan waktu 'besok', netral",
    "pagi": "Keterangan waktu, netral",
    "malam": "Keterangan waktu, netral",
    "kini": "Keterangan waktu, netral",
    "nanti": "Keterangan waktu, netral",
    "sebentar": "Keterangan waktu, netral",
    "sementara": "Keterangan waktu, netral",
    "lalu": "Keterangan waktu, netral",
    "awal": "Keterangan waktu/urutan, netral",
    "akhir": "Keterangan waktu/urutan, netral",
    "final": "Keterangan urutan, netral",
    "lama": "Keterangan waktu/durasi, netral",
    "cepat": "Keterangan kecepatan netral",
    "cepet": "Keterangan kecepatan netral",
    "segera": "Keterangan waktu netral",
    "langsung": "Adverbia cara netral",
    "sering": "Adverbia frekuensi netral",
    "kadang": "Adverbia frekuensi netral",
    "terkadang": "Adverbia frekuensi netral",
    "selalu": "Adverbia frekuensi netral",
    "pernah": "Adverbia frekuensi netral",
    "tetap": "Adverbia kesinambungan netral",
    "masih": "Adverbia kesinambungan netral",
    "sudah": "Adverbia aspek selesai netral",
    "baru": "Adverbia waktu netral",
    "mulai": "Kata kerja memulai netral",
    "lanjut": "Kata kerja melanjutkan netral",
    "selesai": "Kata kerja selesai netral",
    "kelar": "Kata slang selesai netral",
    "habis": "Kata keterangan selesai/habis netral",
    "abis": "Kata slang habis/setelah netral",
    "tamat": "Kata keterangan selesai netral",
    "biar": "Konjungsi tujuan 'agar/supaya', netral",
    "supaya": "Konjungsi tujuan, netral",
    "karena": "Konjungsi sebab, netral",
    "gara": "Bagian frasa 'gara-gara', netral",
    "gegara": "Slang 'gara-gara', netral",
    "meski": "Konjungsi konsesif, netral",
    "walau": "Konjungsi konsesif, netral",
    "cuma": "Adverbia pembatas 'hanya', netral",
    "doang": "Adverbia pembatas slang 'hanya', netral",
    "only": "Kata bahasa Inggris 'hanya', netral",
    "memang": "Adverbia penegas netral",
    "pasti": "Adverbia kepastian netral",
    "mungkin": "Adverbia kemungkinan netral",
    "tentu": "Adverbia kepastian netral",
    "sebenarnya": "Adverbia netral",
    "sebenernya": "Adverbia netral",
    "mending": "Kata pembanding preferensi ('lebih baik/mending'), netral tanpa opini",
    "kayak": "Kata pembanding 'seperti', netral",
    "kaya": "Kata pembanding 'seperti' di korpus medsos, netral",
    "mirip": "Kata pembanding, netral",
    "seperti": "Kata pembanding, netral",
    "beda": "Kata keterangan berbeda, netral",
    "banding": "Kata kerja membandingkan, netral",
    "lain": "Kata ganti/keterangan 'yang lain', netral",
    "sendiri": "Kata keterangan reflektif, netral",
    "diri": "Kata benda reflektif, netral",
    "anda": "Pronomina persona kedua, netral",
    "aku": "Pronomina persona pertama, netral",
    "dia": "Pronomina persona ketiga, netral",
    "elu": "Pronomina persona kedua slang, netral",
    "lau": "Pronomina persona kedua slang, netral",
    "siapa": "Pronomina tanya, netral",
    "apa": "Pronomina tanya, netral",
    "itu": "Pronomina demonstrativa, netral",
    "begini": "Pronomina demonstrativa, netral",
    "segitu": "Pronomina demonstrativa, netral",
    "segitunya": "Pronomina demonstrativa, netral",
    "ada": "Kata kerja eksistensial, netral",
    "punya": "Kata kerja kepemilikan, netral",
    "milik": "Kata kerja kepemilikan, netral",
    "jadi": "Kata kerja/hubung 'menjadi', netral",
    "buat": "Kata kerja/depan 'membuat/untuk', netral",
    "bikin": "Kata kerja 'membuat', netral",
    "guna": "Kata dasar 'menggunakan/pengguna', netral (kecuali 'tidak guna')",
    "pakai": "Kata kerja menggunakan, netral",
    "cari": "Kata kerja mencari, netral",
    "nyari": "Kata kerja mencari slang, netral",
    "nemu": "Kata kerja menemukan slang, netral",
    "temu": "Kata dasar bertemu/menemukan, netral",
    "ketemu": "Kata kerja bertemu, netral",
    "lihat": "Kata kerja melihat, netral",
    "liat": "Kata kerja melihat slang, netral",
    "baca": "Kata kerja membaca, netral",
    "dengar": "Kata kerja mendengar, netral",
    "denger": "Kata kerja mendengar slang, netral",
    "dengerin": "Kata kerja mendengarkan slang, netral",
    "kata": "Kata benda/kerja berkata, netral",
    "bilang": "Kata kerja mengatakan, netral",
    "omong": "Kata kerja berbicara, netral",
    "nyebut": "Kata kerja menyebut, netral",
    "tanya": "Kata kerja bertanya, netral",
    "nanya": "Kata kerja bertanya slang, netral",
    "jawab": "Kata kerja menjawab, netral",
    "minta": "Kata kerja meminta, netral",
    "kasih": "Kata kerja memberi ('kasih tahu/kasih hp'), netral",
    "kasi": "Variasi 'kasih', netral",
    "ngasih": "Variasi 'memberi', netral",
    "beri": "Kata kerja memberi, netral",
    "ambil": "Kata kerja mengambil, netral",
    "bawa": "Kata kerja membawa, netral",
    "kirim": "Kata kerja mengirim, netral",
    "beli": "Kata kerja membeli ('beli hp'), netral",
    "beliin": "Kata kerja membelikan, netral",
    "belikan": "Kata kerja membelikan, netral",
    "dibeliin": "Kata kerja dibelikan, netral",
    "jual": "Kata kerja menjual, netral",
    "bayar": "Kata kerja membayar, netral",
    "harga": "Kata benda harga, netral",
    "uang": "Kata benda uang, netral",
    "duit": "Kata benda uang, netral",
    "dana": "Kata benda dana/aplikasi DANA, netral",
    "saldo": "Kata benda saldo, netral",
    "tabung": "Kata dasar menabung, netral",
    "pulsa": "Kata benda pulsa, netral",
    "kuota": "Kata benda kuota internet, netral",
    "paket": "Kata benda paket data, netral",
    "kerja": "Kata kerja bekerja, netral",
    "usaha": "Kata benda/kerja usaha, netral",
    "bisnis": "Kata benda bisnis, netral",
    "produk": "Kata benda produk, netral",
    "ganti": "Kata kerja mengganti ('ganti umur/akun'), netral",
    "ubah": "Kata kerja mengubah, netral",
    "pindah": "Kata kerja pindah, netral",
    "masuk": "Kata kerja masuk, netral",
    "keluar": "Kata kerja keluar, netral",
    "naik": "Kata kerja naik, netral",
    "turun": "Kata kerja turun, netral",
    "tinggal": "Kata kerja tinggal ('tinggal ganti'), netral",
    "ikut": "Kata kerja ikut, netral",
    "turut": "Kata kerja menurut/ikut, netral",
    "suruh": "Kata kerja menyuruh, netral",
    "nyuruh": "Kata kerja menyuruh slang, netral",
    "pegang": "Kata kerja memegang ('pegang hp'), netral",
    "pilih": "Kata kerja memilih, netral",
    "milih": "Kata kerja memilih slang, netral",
    "coba": "Kata kerja mencoba, netral",
    "cek": "Kata kerja mengecek, netral",
    "check": "Kata kerja mengecek, netral",
    "periksa": "Kata kerja memeriksa, netral",
    "tunggu": "Kata kerja menunggu, netral",
    "nunggu": "Kata kerja menunggu slang, netral",
    "nungguin": "Kata kerja menunggu slang, netral",
    "hadir": "Kata kerja hadir, netral",
    "muncul": "Kata kerja muncul, netral",
    "nongol": "Kata kerja muncul slang, netral",
    "datang": "Kata kerja datang, netral",
    "dateng": "Kata kerja datang slang, netral",
    "balik": "Kata kerja kembali, netral",
    "pulang": "Kata kerja pulang, netral",
    "pergi": "Kata kerja pergi, netral",
    "lari": "Kata kerja lari, netral",
    "tidur": "Kata kerja tidur, netral",
    "makan": "Kata kerja makan, netral",
    "minum": "Kata kerja minum, netral",
    "masak": "Kata kerja memasak, netral",
    "mandi": "Kata kerja mandi, netral",
    "tahu": "Kata kerja mengetahui ('tidak tahu'), netral",
    "tau": "Variasi 'tahu', netral",
    "tauk": "Variasi 'tahu', netral",
    "gatau": "Slang 'tidak tahu', netral",
    "erti": "Kata dasar 'mengerti', netral",
    "ngerti": "Slang 'mengerti', netral",
    "pikir": "Kata dasar 'berpikir', netral",
    "mikir": "Slang 'berpikir', netral",
    "mikirin": "Slang 'memikirkan', netral",
    "ingat": "Kata kerja mengingat, netral",
    "kenal": "Kata kerja mengenal, netral",
    "rasa": "Kata kerja merasa, netral",
    "ngerasa": "Slang 'merasa', netral",
    "kira": "Kata kerja mengira, netral",
    "anggap": "Kata kerja menganggap, netral",
    "maksud": "Kata benda maksud, netral",
    "arti": "Kata benda arti, netral",
    "soal": "Kata benda soal/masalah topik, netral",
    "tentang": "Preposisi perihal, netral",
    "bahas": "Kata dasar membahas, netral",
    "fokus": "Kata kerja/sifat fokus, netral",
    "butuh": "Kata kerja membutuhkan, netral",
    "perlu": "Kata kerja memerlukan, netral",
    "harus": "Modalitas keharusan, netral",
    "wajib": "Modalitas kewajiban, netral",
    "mesti": "Modalitas keharusan, netral",
    "boleh": "Modalitas izin, netral",
    "izin": "Kata benda izin, netral",
    "bisa": "Modalitas kemampuan, netral",
    "mampu": "Modalitas kemampuan, netral",
    "sanggup": "Modalitas kemampuan, netral",
    "mau": "Modalitas keinginan, netral",
    "ingin": "Modalitas keinginan, netral",
    "cukup": "Adverbia kecukupan, netral",
    "banyak": "Kuantifier banyak, netral",
    "sedikit": "Kuantifier sedikit, netral",
    "dikit": "Kuantifier sedikit slang, netral",
    "minim": "Kuantifier minim, netral",
    "penuh": "Kuantifier penuh, netral",
    "kosong": "Kuantifier kosong, netral",
    "satu": "Numeralia, netral",
    "dua": "Numeralia, netral",
    "nol": "Numeralia, netral",
    "juta": "Numeralia, netral",
    "persen": "Satuan persentase, netral",
    "rata": "Kata keterangan rata-rata, netral",
    "tinggi": "Ukuran/tingkat netral",
    "rendah": "Ukuran/tingkat netral",
    "panjang": "Ukuran netral",
    "pendek": "Ukuran netral",
    "luas": "Ukuran netral",
    "dekat": "Jarak netral",
    "jauh": "Jarak netral",
    "dalam": "Preposisi/ukuran netral",
    "atas": "Preposisi posisi netral",
    "tengah": "Preposisi posisi netral",
    "ujung": "Posisi netral",
    "tempat": "Kata benda lokasi netral",
    "ruang": "Kata benda ruang netral",
    "kota": "Kata benda lokasi netral",
    "pelosok": "Kata benda lokasi netral",
    "dunia": "Kata benda dunia netral",
    "alam": "Kata benda alam netral",
    "manusia": "Kata benda manusia netral",
    "orang": "Kata benda orang netral",
    "warga": "Kata benda warga netral",
    "rakyat": "Kata benda rakyat netral",
    "masyarakat": "Kata benda masyarakat netral",
    "kelompok": "Kata benda kelompok netral",
    "komunitas": "Kata benda komunitas netral",
    "forum": "Kata benda forum netral",
    "tim": "Kata benda tim netral",
    "pihak": "Kata benda pihak netral",
    "teman": "Kata benda teman netral",
    "temenan": "Kata kerja berteman netral",
    "friends": "Kata benda teman netral",
    "pacar": "Kata benda pacar netral",
    "mantan": "Kata benda mantan netral",
    "artis": "Kata benda artis netral",
    "fans": "Kata benda penggemar netral",
    "bos": "Kata panggilan bos netral",
    "guys": "Kata sapaan netral",
    "neng": "Kata sapaan netral",
    "man": "Kata sapaan netral",
    "hai": "Kata sapaan netral",
    "halo": "Kata sapaan netral",
    "hello": "Kata sapaan netral",
    "hy": "Kata sapaan netral",
    "hi": "Kata sapaan netral",
    "assalamualaikum": "Salam pembuka, netral",
    "bismillah": "Ungkapan pembuka, netral",
    "allah": "Nama Tuhan (sering dipakai di kalimat positif maupun keluhan), netral",
    "tuhan": "Nama Tuhan, netral",
    "islam": "Nama agama, netral",
    "agama": "Kata benda agama, netral",
    "ibadah": "Kata benda ibadah, netral",
    "sholat": "Kata benda ibadah, netral",
    "puasa": "Kata benda ibadah, netral",
    "adzan": "Kata benda adzan, netral",
    "nabi": "Kata benda nabi, netral",
    "doa": "Kata benda doa, netral",
    "amin": "Ungkapan doa netral",
    "aamiin": "Ungkapan doa netral",
    "moga": "Kata harapan 'semoga' (dipakai di harapan positif maupun protes), netral",
    "semoga": "Kata harapan 'semoga', netral",
    "harap": "Kata kerja berharap, netral",
    "tolong": "Kata permohonan ('tolong blokir' / 'tolong jangan blokir'), netral",
    "tlg": "Singkatan 'tolong', netral",
    "mohon": "Kata permohonan, netral",
    "maaf": "Kata permohonan maaf pembuka opini, netral",
    "sorry": "Kata permohonan maaf, netral",
    "ampun": "Kata seruan netral",
    "sumpah": "Kata penegas slang, netral",
    "asli": "Kata penegas slang, netral",
    "real": "Kata penegas slang, netral",
    "riil": "Kata penegas slang, netral",
    "nyata": "Kata sifat nyata, netral",
    "beneran": "Kata penegas 'sungguh', netral",
    "sungguh": "Kata penegas, netral",
    "serius": "Kata sifat serius, netral",
    "jujur": "Kata pembuka opini ('jujur aku...'), netral",
    "biasa": "Kata sifat biasa, netral",
    "normal": "Kata sifat normal, netral",
    "umum": "Kata sifat umum, netral",
    "resmi": "Kata sifat resmi, netral",
    "otomatis": "Kata sifat otomatis, netral",
    "permanen": "Kata sifat permanen, netral",
    "ketat": "Kata sifat ketat (verifikasi ketat), netral",
    "bebas": "Kata sifat bebas, netral",
    "penuh": "Kata sifat penuh, netral",
    "sederhana": "Kata sifat sederhana, netral",
    "simple": "Kata sifat sederhana, netral",
    "khusus": "Kata sifat khusus, netral",
    "eksklusif": "Kata sifat eksklusif, netral",
    "gratis": "Kata sifat gratis, netral",
    "premium": "Kata benda/sifat akun premium, netral",
    "mahal": "Kata sifat harga buku/kuota, netral terhadap kebijakan",
    "murah": "Kata sifat harga, netral",
    "hidup": "Kata kerja/benda hidup, netral",
    "mati": "Kata hiperbola medsos ('kalo roblox mati'), netral",
    "otak": "Kata benda otak, netral",
    "mata": "Kata benda mata, netral",
    "mulut": "Kata benda mulut, netral",
    "tangan": "Kata benda tangan, netral",
    "tubuh": "Kata benda tubuh, netral",
    "hati": "Kata benda hati, netral",
    "darah": "Kata benda darah, netral",
    "kembang": "Bagian frasa 'tumbuh kembang', netral",
    "tumbuh": "Bagian frasa 'tumbuh kembang', netral",
    "sikap": "Kata benda sikap, netral",
    "pola": "Bagian frasa 'pola asuh', netral",
    "langkah": "Kata benda langkah kebijakan, netral",
    "rencana": "Kata benda rencana, netral",
    "usul": "Kata benda usul, netral",
    "saran": "Kata benda saran, netral",
    "argumen": "Kata benda argumen, netral",
    "kritik": "Kata benda kritik, netral",
    "debat": "Kata benda debat, netral",
    "diskusi": "Kata benda diskusi, netral",
    "musyawarah": "Kata benda musyawarah, netral",
    "lapor": "Kata kerja melapor, netral",
    "tuntut": "Kata kerja menuntut, netral",
    "protes": "Kata kerja protes, netral",
    "demo": "Kata benda/kerja demonstrasi, netral",
    "aksi": "Kata benda aksi, netral",
    "juang": "Kata dasar berjuang, netral",
    "lawan": "Kata kerja melawan, netral",
    "tahan": "Kata kerja menahan, netral",
    "jaga": "Kata kerja menjaga, netral",
    "rawat": "Kata kerja merawat, netral",
    "pelihara": "Kata kerja memelihara, netral",
    "simpan": "Kata kerja menyimpan, netral",
    "bantu": "Kata kerja membantu ('youtube bantu belajar' vs 'bantu orang tua'), netral",
    "bagi": "Preposisi/kata kerja membagi, netral",
    "pas": "Preposisi slang 'saat/ketika', netral",
    "sampai": "Preposisi batas, netral",
    "smpai": "Variasi 'sampai', netral",
    "lewat": "Preposisi 'melalui', netral",
    "lanjut": "Kata kerja lanjut, netral",
    "isi": "Kata benda/kerja isi, netral",
    "terima": "Bagian frasa 'terima kasih' atau 'tidak terima' (ditangani via bigram/resolusi)",
    "kasihan": "Kata empati ('kasihan anak/penjual'), netral terhadap kebijakan",
    "sayang": "Kata ambigu ('sayang anak' vs 'sayang sekali'), netral",
    "kangen": "Kata perasaan rindu, netral",
    "mimpi": "Kata benda mimpi/cita-cita, netral",
    "cita": "Kata benda cita-cita, netral",
    "potensi": "Kata benda potensi, netral",
    "bakat": "Kata benda bakat, netral",
    "minat": "Kata benda minat, netral",
    "kreatif": "Kata sifat kreatif (sering pada keluhan kreator kecil), netral",
    "kreativitas": "Kata benda kreativitas, netral",
    "inspirasi": "Kata benda inspirasi, netral",
    "motivasi": "Kata benda motivasi, netral",
    "semangat": "Kata seruan semangat, netral",
    "fokus": "Kata sifat fokus, netral",
    "sibuk": "Kata sifat sibuk, netral",
    "santai": "Kata sifat santai, netral",
    "tenang": "Kata sifat tenang, netral",
    "diam": "Kata kerja diam, netral",
    "diem": "Variasi 'diam', netral",
    "anteng": "Kata sifat diam/tenang, netral",
    "rame": "Kata sifat ramai, netral",
    "ramai": "Kata sifat ramai, netral",
    "sepi": "Kata sifat sepi, netral",
    "sunyi": "Kata sifat sunyi, netral",
    "wkwk": "Ekspresi tawa medsos (dipakai di sindiran maupun candaan), netral",
    "wkwkwk": "Ekspresi tawa medsos, netral",
    "ngakak": "Ekspresi tawa medsos, netral",
    "lucu": "Kata sifat lucu, netral",
    "kaget": "Ekspresi terkejut, netral",
    "heran": "Ekspresi heran, netral",
    "wah": "Interjeksi, netral",
    "ayo": "Interjeksi ajakan, netral",
    "yuk": "Interjeksi ajakan, netral",
    "mari": "Interjeksi ajakan, netral",
    "hayo": "Interjeksi, netral",
    "hayoloh": "Interjeksi, netral",
    "dah": "Partikel penegas slang, netral",
    "yaudah": "Partikel slang, netral",
    "oke": "Kata persetujuan ringan/transisi, netral",
    "ok": "Kata transisi, netral",
    "okay": "Kata transisi, netral",
    "yes": "Kata seruan umum, netral",
    "entuk": "Bentuk tidak baku, netral",
    "bentuk": "Kata benda bentuk, netral",
    "nilai": "Kata benda nilai sekolah, netral",
    "lulus": "Kata kerja lulus kelas, netral",
    "absen": "Kata kerja absen komentar, netral",
    "follback": "Istilah medsos follow back, netral",
    "comeback": "Istilah medsos, netral",
    "join": "Istilah medsos, netral",
    "gas": "Kata slang, netral",
    "mulu": "Adverbia frekuensi slang 'terus-menerus', netral",
    "bareng": "Adverbia 'bersama', netral",
    "duluan": "Adverbia 'lebih dulu', netral",
    "serta": "Konjungsi, netral",
    "kabar": "Kata benda kabar, netral",
    "selamat": "Sering dipakai dalam arti 'lolos dari blokir (aku selamat)', netral",
    "untung": "Sering dipakai dalam arti 'untung umurku sudah 17+', netral",
    "aman": "Sering dipakai dalam arti 'akun ku aman gak keblokir', netral (kecuali bigram 'aman bagi anak')",
    "kena": "Istilah teknis 'akun kena blokir/pembatasan', netral tanpa opini",
    "lupa": "Sering dipakai dalam klausa 'jangan lupa urus judol', netral",
    "jelas": "Dalam frasa tanya 'tolong jelaskan prosedurnya' netral; 'tidak jelas' ditangani via bigram negatif",
    "negatif": "Sering muncul pada klausa 'melindungi dari konten negatif' (tujuan positif perlindungan anak), netral",
    "kaci": "Slang plesetan 'kasihan (kacian)', netral terhadap kebijakan",
}

# 2. Resolusi kata ganda (overlap antara positive.tsv dan negative.tsv di InSet asli)
#    serta penyesuaian bobot kata opini yang relevan di korpus ABSA kebijakan Komdigi:
RESOLVED_AND_ADDED_INSET_WORDS: dict[str, tuple[int, str]] = {
    # --- KATA OPINI POSITIF (Bobot +2 s.d. +5) ---
    "setuju": (5, "Resolusi kata ganda InSet (+4/-3) -> Positif kuat (+5) untuk persetujuan kebijakan"),
    "tuju": (5, "Resolusi stem Sastrawi 'setuju' di InSet (+3/-4) -> Positif kuat (+5)"),
    "stuju": (5, "Penambahan variasi ejaan 'setuju' (+5)"),
    "se7": (5, "Penambahan slang 'setuju' (+5)"),
    "dukung": (4, "Konfirmasi bobot InSet asli (+4) untuk dukungan kebijakan"),
    "mendukung": (4, "Penambahan bentuk berimbuhan 'dukung' (+4)"),
    "alhamdulillah": (5, "Konfirmasi bobot InSet asli (+5) ungkapan syukur mendukung kebijakan"),
    "syukur": (4, "Konfirmasi bobot InSet asli (+5 -> +4)"),
    "bagus": (4, "Resolusi kata ganda InSet (+2/-4) -> Positif (+4)"),
    "bgs": (4, "Penambahan singkatan 'bagus' (+4)"),
    "baik": (3, "Resolusi kata ganda InSet (+3/-1) -> Positif (+3)"),
    "terbaik": (5, "Penambahan superlatif 'baik' (+5)"),
    "mantap": (5, "Konfirmasi bobot InSet asli (+5)"),
    "keren": (4, "Penambahan kata opini positif (+4)"),
    "hebat": (5, "Penambahan kata opini positif (+5)"),
    "top": (4, "Konfirmasi bobot InSet asli (+5 -> +4)"),
    "good": (4, "Konfirmasi bobot InSet asli (+5 -> +4)"),
    "nice": (4, "Konfirmasi bobot InSet asli (+4)"),
    "senang": (4, "Resolusi kata ganda InSet (+5/-4) -> Positif (+4)"),
    "bahagia": (4, "Resolusi kata ganda InSet (+4/-1) -> Positif (+4)"),
    "suka": (3, "Resolusi kata ganda InSet (+3/-1) -> Positif (+3)"),
    "cinta": (4, "Penambahan kata opini positif (+4)"),
    "lindung": (4, "Konfirmasi bobot InSet asli (+4) untuk perlindungan anak"),
    "lindungi": (4, "Penambahan bentuk berimbuhan 'lindung' (+4)"),
    "melindungi": (4, "Penambahan bentuk berimbuhan 'lindung' (+4)"),
    "terlindung": (4, "Penambahan bentuk berimbuhan 'lindung' (+4)"),
    "aman bagi anak": (4, "Penambahan frasa spesifik perlindungan anak (+4)"),
    "aman untuk anak": (4, "Penambahan frasa spesifik perlindungan anak (+4)"),
    "efektif": (4, "Konfirmasi bobot InSet asli (+5 -> +4)"),
    "efisien": (4, "Penambahan kata opini efektivitas (+4)"),
    "berhasil": (4, "Penambahan kata opini keberhasilan kebijakan (+4)"),
    "sukses": (4, "Konfirmasi bobot InSet asli (+3 -> +4)"),
    "manfaat": (4, "Konfirmasi bobot InSet asli (+5 -> +4)"),
    "bermanfaat": (4, "Penambahan bentuk berimbuhan 'manfaat' (+4)"),
    "berguna": (4, "Penambahan kata opini positif (+4)"),
    "solusi": (3, "Konfirmasi bobot InSet asli (+3)"),
    "solutif": (4, "Penambahan kata opini positif (+4)"),
    "positif": (3, "Konfirmasi bobot InSet asli (+4 -> +3)"),
    "sehat": (4, "Resolusi kata ganda InSet (+4/-1) -> Positif (+4)"),
    "cerdas": (4, "Penambahan kata opini positif (+4)"),
    "pintar": (4, "Penambahan kata opini positif (+4)"),
    "pinter": (4, "Resolusi kata ganda InSet (+5/-1) -> Positif (+4)"),
    "cerdik": (3, "Konfirmasi bobot InSet asli (+2 -> +3)"),
    "pandai": (4, "Resolusi kata ganda InSet (+4/-4) -> Positif (+4)"),
    "tepat": (4, "Penambahan kata opini ketepatan kebijakan (+4)"),
    "cocok": (3, "Resolusi kata ganda InSet (+4/-3) -> Positif (+3)"),
    "sesuai": (3, "Konfirmasi bobot InSet asli (+3)"),
    "layak": (3, "Penambahan kata opini positif (+3)"),
    "adil": (4, "Resolusi kata ganda InSet (+4/-3) -> Positif (+4)"),
    "tegas": (3, "Penambahan kata opini positif ketegasan regulasi (+3)"),
    "peduli": (4, "Resolusi kata ganda InSet (+4/-1) -> Positif (+4)"),
    "duli": (4, "Resolusi stem Sastrawi 'peduli' di InSet (+4/-3) -> Positif (+4)"),
    "nyaman": (4, "Konfirmasi bobot InSet asli (+4)"),
    "tertib": (4, "Resolusi kata ganda InSet (+4/-2) -> Positif (+4)"),
    "patuh": (4, "Konfirmasi bobot InSet asli (+4)"),
    "taat": (3, "Resolusi kata ganda InSet (+3/-1) -> Positif (+3)"),
    "benar": (3, "Resolusi kata ganda InSet (+2/-4) -> Positif (+3)"),
    "betul": (3, "Resolusi kata ganda InSet (+2/-4) -> Positif (+3)"),
    "penting": (3, "Resolusi kata ganda InSet (+3/-4) -> Positif (+3)"),
    "realistis": (3, "Konfirmasi bobot InSet asli (+4 -> +3)"),
    "masuk akal": (3, "Penambahan bigram positif (+3)"),
    "terima kasih": (4, "Penambahan bigram apresiasi (+4)"),
    "makasih": (4, "Konfirmasi bobot InSet asli (+4)"),
    "trimakasih": (4, "Konfirmasi bobot InSet asli (+4)"),
    "mksh": (3, "Konfirmasi bobot InSet asli (+2 -> +3)"),
    "sabar": (3, "Konfirmasi bobot InSet asli (+5 -> +3)"),
    "amanah": (4, "Konfirmasi bobot InSet asli (+5 -> +4)"),
    "adab": (3, "Konfirmasi bobot InSet asli (+3)"),
    "akhlak": (4, "Konfirmasi bobot InSet asli (+4)"),
    "moral": (3, "Konfirmasi bobot InSet asli (+4 -> +3)"),
    "bangga": (4, "Konfirmasi bobot InSet asli (+3 -> +4)"),
    "puas": (4, "Konfirmasi bobot InSet asli (+3 -> +4)"),
    "ramah": (4, "Resolusi kata ganda InSet (+5/-3) -> Positif (+4)"),
    "indah": (3, "Resolusi kata ganda InSet (+3/-2) -> Positif (+3)"),
    "lancar": (3, "Resolusi kata ganda InSet (+4/-2) -> Positif (+3)"),
    "stabil": (3, "Konfirmasi bobot InSet asli (+3)"),
    "cemerlang": (4, "Resolusi kata ganda InSet (+4/-1) -> Positif (+4)"),
    "optimal": (4, "Penambahan kata opini positif (+4)"),
    "maksimal": (3, "Konfirmasi bobot InSet asli (+3)"),
    "mudah": (3, "Resolusi kata ganda InSet (+4/-1) -> Positif (+3)"),
    "gampang": (2, "Resolusi kata ganda InSet (+4/-3) -> Positif ringan (+2)"),
    "asik": (3, "Konfirmasi bobot InSet asli (+4 -> +3)"),
    "seru": (3, "Resolusi kata ganda InSet (+5/-5) -> Positif (+3)"),
    "enak": (3, "Resolusi kata ganda InSet (+5/-4) -> Positif (+3)"),
    "emoji_suka": (4, "Penambahan tokenisasi emoji positif (+4)"),
    "emoji suka": (4, "Penambahan tokenisasi emoji positif (+4)"),
    "emoji_setuju": (4, "Penambahan tokenisasi emoji setuju (+4)"),
    "emoji tuju": (4, "Penambahan tokenisasi emoji setuju setelah stemming (+4)"),
    "emoji_keren": (4, "Penambahan tokenisasi emoji keren (+4)"),
    "emoji keren": (4, "Penambahan tokenisasi emoji keren (+4)"),

    # --- KATA & FRASA OPINI NEGATIF (Bobot -2 s.d. -5) ---
    "tidak tuju": (-5, "Penambahan bigram negasi 'tidak setuju' (-5)"),
    "tidak setuju": (-5, "Penambahan bigram negasi 'tidak setuju' (-5)"),
    "kurang tuju": (-4, "Penambahan bigram negasi 'kurang setuju' (-4)"),
    "kurang setuju": (-4, "Penambahan bigram negasi 'kurang setuju' (-4)"),
    "tidak efektif": (-5, "Penambahan bigram negasi 'tidak efektif' (-5)"),
    "tidak adil": (-4, "Penambahan bigram negasi 'tidak adil' (-4)"),
    "tidak jelas": (-4, "Penambahan bigram negasi 'tidak jelas' (-4)"),
    "tidak guna": (-4, "Penambahan bigram negasi 'tidak berguna' (-4)"),
    "tidak manfaat": (-4, "Penambahan bigram negasi 'tidak bermanfaat' (-4)"),
    "tidak suka": (-4, "Penambahan bigram negasi 'tidak suka' (-4)"),
    "tidak aman": (-4, "Penambahan bigram negasi 'tidak aman' (-4)"),
    "tidak terima": (-4, "Penambahan bigram negasi 'tidak terima' (-4)"),
    "tidak masuk akal": (-4, "Penambahan frasa negasi (-4)"),
    "jangan blokir": (-4, "Penambahan bigram penolakan pemblokiran (-4)"),
    "jangan blok": (-4, "Penambahan bigram penolakan pemblokiran (-4)"),
    "jangan batas": (-4, "Penambahan bigram penolakan pembatasan (-4)"),
    "jangan hapus": (-4, "Penambahan bigram penolakan penghapusan (-4)"),
    "jangan larang": (-4, "Penambahan bigram penolakan pelarangan (-4)"),
    "jangan tutup": (-4, "Penambahan bigram penolakan penutupan (-4)"),
    "tolak": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "menolak": (-4, "Penambahan bentuk berimbuhan 'tolak' (-4)"),
    "gagal": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "percuma": (-4, "Resolusi kata ganda InSet (+4/-4) -> Negatif (-4)"),
    "sia": (-5, "Konfirmasi bobot InSet asli (-5) untuk 'sia-sia'"),
    "sia sia": (-5, "Penambahan bentuk reduplikasi 'sia-sia' (-5)"),
    "mustahil": (-4, "Resolusi kata ganda InSet (+3/-2) -> Negatif (-4)"),
    "buruk": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "jelek": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "parah": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "kacau": (-4, "Penambahan kata opini negatif (-4)"),
    "rusak": (-4, "Resolusi kata ganda InSet (+3/-4) -> Negatif (-4)"),
    "merusak": (-5, "Penambahan bentuk berimbuhan 'rusak' (-5)"),
    "usak": (-4, "Konfirmasi stem Sastrawi tidak baku untuk 'merusak' (-3 -> -4)"),
    "hancur": (-5, "Penambahan kata opini negatif (-5)"),
    "runtuh": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "bahaya": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "berbahaya": (-5, "Penambahan bentuk berimbuhan 'bahaya' (-5)"),
    "ancam": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "ancaman": (-4, "Penambahan bentuk berimbuhan 'ancam' (-4)"),
    "rugi": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "merugikan": (-5, "Penambahan bentuk berimbuhan 'rugi' (-5)"),
    "beban": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "masalah": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "problem": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "salah": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "nyalahin": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "keliru": (-4, "Penambahan kata opini negatif (-4)"),
    "sesat": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "bohong": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "boong": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "palsu": (-4, "Penambahan kata opini negatif manipulasi usia (-4)"),
    "tipu": (-4, "Penambahan kata opini negatif penipuan (-4)"),
    "hoax": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "hoaks": (-5, "Penambahan variasi ejaan 'hoax' (-5)"),
    "fitnah": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "curang": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "langgar": (-4, "Penambahan kata negatif pelanggaran (-4)"),
    "bobol": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "bocor": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "tembus": (-3, "Penambahan kata negatif celah sistem (-3)"),
    "ngakali": (-3, "Penambahan kata negatif mengakali aturan (-3)"),
    "bypass": (-3, "Penambahan kata negatif mengakali sistem (-3)"),
    "bypas": (-3, "Penambahan variasi ejaan 'bypass' (-3)"),
    "celah": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "manipulasi": (-4, "Penambahan kata negatif manipulasi data/usia (-4)"),
    "bodoh": (-5, "Penambahan kata opini negatif (-5)"),
    "bodo": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "bego": (-5, "Penambahan kata opini negatif (-5)"),
    "goblok": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "tolol": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "dongok": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "idiot": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "oon": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "gila": (-4, "Resolusi kata ganda InSet (+3/-5) -> Negatif (-4)"),
    "gelo": (-3, "Konfirmasi bobot InSet asli (-2 -> -3)"),
    "waras": (-3, "Resolusi kata ganda InSet (+2/-3) -> Negatif (-3)"),
    "kejam": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "keji": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "jahat": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "tega": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "egois": (-4, "Penambahan kata opini negatif (-4)"),
    "kolot": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "purba": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "jadul": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "susah": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "sulit": (-3, "Penambahan kata opini negatif (-3)"),
    "ribet": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "repot": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "belit": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "hambat": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "halang": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "kekang": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "ekang": (-4, "Konfirmasi stem Sastrawi 'mengekang' (-4)"),
    "bungkam": (-4, "Resolusi kata ganda InSet (+1/-4) -> Negatif (-4)"),
    "paksa": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "maksa": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "lambat": (-3, "Resolusi bobot InSet (+1) -> Negatif (-3) untuk kelambatan kebijakan"),
    "lemot": (-3, "Penambahan kata opini negatif (-3)"),
    "telat": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "kasep": (-4, "Konfirmasi bobot InSet asli (-4) bahasa daerah 'terlambat'"),
    "sedih": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "sad": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "nangis": (-4, "Penambahan kata ekspresi kesedihan (-4)"),
    "mewek": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "kecewa": (-4, "Penambahan kata opini negatif (-4)"),
    "nyesel": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "miris": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "prihatin": (-3, "Resolusi kata ganda InSet (+2/-3) -> Negatif (-3)"),
    "marah": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "marahin": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "amuk": (-5, "Resolusi kata ganda InSet (+1/-5) -> Negatif (-5)"),
    "kesal": (-4, "Resolusi kata ganda InSet (+4/-5) -> Negatif (-4)"),
    "kesel": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "ngeselin": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "nyebelin": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "benci": (-5, "Resolusi kata ganda InSet (+4/-5) -> Negatif (-5)"),
    "muak": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "najis": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "resah": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "gelisah": (-4, "Penambahan kata opini negatif (-4)"),
    "stres": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "stress": (-4, "Penambahan variasi ejaan 'stres' (-4)"),
    "depresi": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "trauma": (-4, "Resolusi kata ganda InSet (+4/-5) -> Negatif (-4)"),
    "sakit": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "pusing": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "panik": (-3, "Konfirmasi bobot InSet asli (-4 -> -3)"),
    "takut": (-3, "Resolusi kata ganda InSet (+1/-4) -> Negatif (-3)"),
    "cemas": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "khawatir": (-3, "Resolusi kata ganda InSet (+1/-5) -> Negatif (-3)"),
    "capek": (-3, "Resolusi kata ganda InSet (+4/-4) -> Negatif (-3)"),
    "cape": (-3, "Konfirmasi bobot InSet asli (-4 -> -3)"),
    "lelah": (-3, "Penambahan kata opini negatif (-3)"),
    "bosan": (-3, "Penambahan kata opini negatif (-3)"),
    "bosen": (-3, "Konfirmasi bobot InSet asli (-4 -> -3)"),
    "gabut": (-3, "Konfirmasi bobot InSet asli (-5 -> -3)"),
    "malas": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "males": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "mls": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "malu": (-3, "Resolusi kata ganda InSet (+2/-4) -> Negatif (-3)"),
    "candu": (-4, "Resolusi kata ganda InSet (+3/-3) -> Negatif (-4) untuk kecanduan gadget/medsos"),
    "kecanduan": (-4, "Penambahan bentuk berimbuhan 'candu' (-4)"),
    "ketagihan": (-4, "Penambahan kata negatif adiksi (-4)"),
    "ketergantungan": (-4, "Penambahan kata negatif adiksi (-4)"),
    "gantung": (-3, "Konfirmasi stem Sastrawi 'ketergantungan' (-3)"),
    "judol": (-4, "Penambahan istilah domain 'judi online' (-4)"),
    "judi": (-4, "Penambahan istilah domain perjudian (-4)"),
    "pinjol": (-4, "Penambahan istilah domain pinjaman online ilegal (-4)"),
    "korupsi": (-5, "Konfirmasi bobot InSet asli (-4 -> -5)"),
    "koruptor": (-5, "Penambahan istilah domain korupsi (-5)"),
    "korup": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "suap": (-4, "Resolusi kata ganda InSet (+1/-3) -> Negatif (-4)"),
    "porno": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "pornografi": (-5, "Penambahan istilah domain pornografi (-5)"),
    "bokep": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "cabul": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "senonoh": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "leceh": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "pelecehan": (-5, "Penambahan bentuk berimbuhan 'leceh' (-5)"),
    "predator": (-5, "Penambahan istilah bahaya predator anak (-5)"),
    "eksploitasi": (-5, "Penambahan istilah eksploitasi anak (-5)"),
    "rundung": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "perundungan": (-4, "Penambahan bentuk berimbuhan 'rundung' (-4)"),
    "bully": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "dibully": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "bullying": (-5, "Penambahan istilah cyberbullying (-5)"),
    "cyberbullying": (-5, "Penambahan istilah cyberbullying (-5)"),
    "hujat": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "ejek": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "lalai": (-4, "Penambahan kata negatif kelalaian pengawasan orang tua (-4)"),
    "abai": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "lepas tangan": (-4, "Penambahan bigram negatif kelalaian orang tua (-4)"),
    "manja": (-3, "Resolusi kata ganda InSet (+5/-3) -> Negatif (-3)"),
    "ceroboh": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "becus": (-4, "Resolusi kata ganda InSet (+1/-4) -> Negatif (-4)"),
    "aneh": (-3, "Resolusi kata ganda InSet (+4/-4) -> Negatif (-3)"),
    "absurd": (-3, "Resolusi kata ganda InSet (+3/-2) -> Negatif (-3)"),
    "gaje": (-3, "Konfirmasi bobot InSet asli (-3) slang 'gak jelas'"),
    "kocak": (-3, "Konfirmasi bobot InSet asli (-2 -> -3) sindiran kebijakan"),
    "lawak": (-3, "Penambahan kata sindiran negatif (-3)"),
    "lebay": (-3, "Penambahan kata opini negatif (-3)"),
    "norak": (-3, "Penambahan kata opini negatif (-3)"),
    "sampah": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "ampas": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "receh": (-3, "Konfirmasi bobot InSet asli (-4 -> -3)"),
    "ganggu": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "mengganggu": (-4, "Penambahan bentuk berimbuhan 'ganggu' (-4)"),
    "rusuh": (-4, "Resolusi kata ganda InSet (+1/-4) -> Negatif (-4)"),
    "ribut": (-3, "Resolusi kata ganda InSet (+4/-4) -> Negatif (-3)"),
    "tengkar": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "korban": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "sengsara": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "melarat": (-4, "Konfirmasi bobot InSet asli (-4)"),
    "miskin": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "bangkrut": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "bencana": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "darurat": (-4, "Resolusi kata ganda InSet (+1/-4) -> Negatif (-4)"),
    "gawat": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "parah": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "anjir": (-3, "Resolusi kata ganda InSet (+1/-3) -> Negatif (-3)"),
    "njir": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "jir": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "anj": (-3, "Penambahan umpatan slang negatif (-3)"),
    "anjg": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "anjay": (-2, "Konfirmasi bobot InSet asli (-3 -> -2)"),
    "cok": (-3, "Penambahan umpatan slang negatif (-3)"),
    "bacot": (-4, "Resolusi kata ganda InSet (+3/-4) -> Negatif (-4)"),
    "nyocot": (-4, "Penambahan umpatan slang negatif (-4)"),
    "taik": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "tai": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "eek": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "sial": (-4, "Konfirmasi bobot InSet asli (-5 -> -4)"),
    "sialan": (-4, "Penambahan bentuk berimbuhan 'sial' (-4)"),
    "laknat": (-5, "Konfirmasi bobot InSet asli (-5)"),
    "hadeh": (-3, "Konfirmasi bobot InSet asli (-4 -> -3)"),
    "halah": (-3, "Konfirmasi bobot InSet asli (-5 -> -3)"),
    "alah": (-3, "Konfirmasi bobot InSet asli (-4 -> -3)"),
    "aduh": (-2, "Resolusi kata ganda InSet (+1/-3) -> Negatif ringan (-2)"),
    "duh": (-2, "Konfirmasi bobot InSet asli (-4 -> -2)"),
    "waduh": (-2, "Resolusi kata ganda InSet (+3/-3) -> Negatif ringan (-2)"),
    "astaghfirullah": (-3, "Konfirmasi bobot InSet asli (-3)"),
    "astaga": (-2, "Resolusi kata ganda InSet (+3/-4) -> Negatif ringan (-2)"),
    "emoji_sedih": (-4, "Penambahan tokenisasi emoji sedih (-4)"),
    "emoji sedih": (-4, "Penambahan tokenisasi emoji sedih (-4)"),
    "emoji_marah": (-5, "Penambahan tokenisasi emoji marah (-5)"),
    "emoji marah": (-5, "Penambahan tokenisasi emoji marah (-5)"),
}


def build_inset_lexicon_from_original() -> tuple[dict[str, int], dict[str, int], list[dict]]:
    """
    Membentuk kamus `inset_positive` dan `inset_negative` langsung dari kamus InSet asli
    (`inset_original_positive.tsv` & `inset_original_negative.tsv`), lalu menerapkan:
    1. Penetralan kata operator negasi/intensifier dan kata benda/fungsi domain (`NEUTRALIZED_INSET_WORDS`)
    2. Resolusi otomatis kata ganda (overlap) pada InSet asli yang belum diatur manual:
       - Jika bobot positif > |bobot negatif|, masuk ke positif dengan selisih/bobot dominan
       - Jika |bobot negatif| > bobot positif, masuk ke negatif dengan bobot dominan
       - Jika sama kuat (mis. +3 dan -3), dinetralkan (0) karena ambigu
    3. Penyesuaian dan penambahan kata/frasa domain & stem Sastrawi (`RESOLVED_AND_ADDED_INSET_WORDS`)
    Seluruh perubahan dan tambahan dicatat ke dalam daftar `modification_records`.
    """
    pos_lex: dict[str, int] = {}
    neg_lex: dict[str, int] = {}
    mod_records: list[dict] = []

    all_orig_words = sorted(set(inset_original_positive.keys()) | set(inset_original_negative.keys()))

    for w in all_orig_words:
        orig_p = inset_original_positive.get(w)
        orig_n = inset_original_negative.get(w)

        # 1. Cek apakah kata dinetralkan secara eksplisit
        if w in NEUTRALIZED_INSET_WORDS:
            mod_records.append({
                "kata": w,
                "bobot_asli_positif": orig_p if orig_p is not None else "",
                "bobot_asli_negatif": orig_n if orig_n is not None else "",
                "bobot_ akhir": 0,
                "kategori_perubahan": "Dinetralkan (Kata Fungsi / Operator / Entitas Domain)",
                "alasan": NEUTRALIZED_INSET_WORDS[w],
            })
            continue

        # 2. Cek apakah kata ada di daftar resolusi/penyesuaian eksplisit
        if w in RESOLVED_AND_ADDED_INSET_WORDS:
            new_w, reason = RESOLVED_AND_ADDED_INSET_WORDS[w]
            if new_w > 0:
                pos_lex[w] = new_w
            elif new_w < 0:
                neg_lex[w] = new_w
            if orig_p is not None and orig_n is not None:
                cat = "Resolusi Kata Ganda (Overlap InSet Asli)"
            elif (orig_p is not None and orig_p != new_w) or (orig_n is not None and orig_n != new_w):
                cat = "Penyesuaian Bobot Kata InSet Asli"
            else:
                cat = "Konfirmasi Bobot InSet Asli"
            mod_records.append({
                "kata": w,
                "bobot_asli_positif": orig_p if orig_p is not None else "",
                "bobot_asli_negatif": orig_n if orig_n is not None else "",
                "bobot_ akhir": new_w,
                "kategori_perubahan": cat,
                "alasan": reason,
            })
            continue

        # 3. Jika kata muncul di KEDUA kamus asli (overlap 1.142 kata) dan belum ada di daftar manual:
        if orig_p is not None and orig_n is not None:
            if orig_p > abs(orig_n):
                pos_lex[w] = orig_p
            elif abs(orig_n) > orig_p:
                neg_lex[w] = orig_n
            # Jika orig_p == abs(orig_n), kata ambigu seimbang -> tidak dimasukkan
            continue

        # 4. Kata murni 1 polaritas dari kamus InSet asli tetap dipertahankan utuh!
        if orig_p is not None:
            pos_lex[w] = orig_p
        elif orig_n is not None:
            neg_lex[w] = orig_n

    # 5. Tambahkan kata/frasa baru (stem Sastrawi, slang medsos, bigram negasi, emoji)
    for w, (new_w, reason) in sorted(RESOLVED_AND_ADDED_INSET_WORDS.items()):
        if w not in inset_original_positive and w not in inset_original_negative:
            if new_w > 0:
                pos_lex[w] = new_w
            elif new_w < 0:
                neg_lex[w] = new_w
            mod_records.append({
                "kata": w,
                "bobot_asli_positif": "",
                "bobot_asli_negatif": "",
                "bobot_ akhir": new_w,
                "kategori_perubahan": "Penambahan Kata / Bigram Domain & Stem Sastrawi",
                "alasan": reason,
            })

    return pos_lex, neg_lex, mod_records


inset_positive, inset_negative, INSET_MODIFICATION_LOG = build_inset_lexicon_from_original()

# Kompatibilitas pemanggilan modul
positive_words = dict(inset_positive)
negative_words = {k: abs(v) for k, v in inset_negative.items()}

intensifiers = {
    "sangat": 1.5,
    "banget": 1.5,
    "sekali": 1.5,
    "amat": 1.5,
    "paling": 1.5,
}

negation_words = {
    "tidak",
    "bukan",
    "jangan",
    "belum",
    "tanpa",
    "kurang",
    "tak",
    "tdk",
    "gak",
    "ga",
    "gk",
    "nggak",
    "enggak",
}

contrastive_conjunctions = {
    "tapi",
    "tetapi",
    "namun",
    "sedangkan",
    "padahal",
    "meski",
    "meskipun",
    "walaupun",
    "walau",
    "cuma",
    "cuman",
    "mending",
    "daripada",
    "kecuali",
    "akan tetapi",
    "sebaliknya",
}


# ==============================================================================
# 2. PEMETAAN 4 KATEGORI ASPEK DENGAN ATURAN KONTEKS TERKETAT (POIN 4 REVISI)
# ==============================================================================
# Acuan Revisi Poin 4:
# "Perketat aturan pemetaan aspek. Kata seperti 'hp', 'beli hp', atau 'youtube'
#  jangan langsung menentukan kategori tanpa mempertimbangkan konteks.
#  Tetap gunakan empat aspek sesuai rancangan."
#
# Solusi Arsitektur:
# Setiap aspek dibagi menjadi:
#   A. KATA KUNCI INTI (CORE KEYWORDS): Frasa/kata spesifik yang secara langsung dan
#      tidak ambigu merujuk pada aspek tersebut sesuai definisi Bab I & Bab III.
#   B. KATA KUNCI KONTEKSTUAL (CONTEXTUAL KEYWORDS): Kata umum/ambigu seperti
#      "hp", "gadget", "beli hp", "youtube", "tiktok", "roblox", "game", "sosmed",
#      "akun", "sekolah", "ibu", "bapak", "umur", "usia" yang HANYA memicu aspek
#      apabila muncul BERSAMA kata konteks pendukung (supporting context) yang relevan!

ASPECT_CATEGORIES = [
    "Perlindungan Anak",
    "Peran Orang Tua",
    "Peran Pemerintah/Komdigi",
    "Efektivitas Kebijakan",
]

# A. Kata Kunci Inti (Unambiguous Core Keywords) per Aspek
ASPECT_CORE_KEYWORDS: dict[str, list[str]] = {
    "Perlindungan Anak": [
        # Subjek anak & kelompok usia anak di bawah umur
        "anak", "anak anak", "anak kecil", "anak bangsa", "anak dibawah umur",
        "anak di bawah umur", "bocil", "bocah", "remaja", "generasi", "generasi muda",
        "bawah umur", "dibawah umur", "batas usia", "batas umur", "usia anak", "umur anak",
        "usia dini", "16 tahun", "enam belas tahun", "dibawah 16", "di bawah 16",
        "adek", "adik", "pelajar", "siswa", "anak sd", "anak smp",
        # Perlindungan, keamanan, psikologis, mental, moral, & bahaya bagi anak
        "perlindungan", "melindungi", "lindungi", "terlindung", "proteksi", "lindung",
        "jaga anak", "menjaga anak", "keselamatan anak", "keamanan anak",
        "aman bagi anak", "aman untuk anak", "kesehatan mental", "sehat mental",
        "mental anak", "mental", "psikologis", "psikis", "tumbuh kembang",
        "perkembangan anak", "moral anak", "moral", "akhlak", "karakter anak",
        "kecanduan", "candu", "ketagihan", "ketergantungan",
        "konten dewasa", "konten negatif", "konten berbahaya", "konten porno", "konten sensitif",
        "tontonan anak", "pornografi", "porno", "bokep", "seksual", "kekerasan",
        "cyberbullying", "bullying", "perundungan", "rundung", "dibully", "bully",
        "predator", "pelecehan", "leceh", "eksploitasi", "korban",
        "bahaya medsos", "bahaya sosmed", "bahaya bagi anak", "merusak anak", "merusak mental",
        "merusak generasi", "masa depan anak", "depan anak",
    ],
    "Peran Orang Tua": [
        # Subjek orang tua & pengasuhan/pengawasan keluarga
        "orang tua", "orangtua", "ortu", "ortunya", "peran orang tua", "peran ortu",
        "pola asuh", "pengasuhan", "mengasuh", "asuh",
        "pengawasan orang tua", "pengawasan ortu", "pengawasan", "mengawasi", "awasi", "diawasi", "ngawasin",
        "pendampingan", "mendampingi", "dampingi", "didampingi", "damping",
        "pemantauan", "memantau", "pantau", "dipantau",
        "membimbing anak", "bimbingan orang tua", "bimbing anak", "membina anak",
        "didik anak", "mendidik anak", "pendidikan keluarga", "didikan orang tua",
        "tanggung jawab orang tua", "tanggung jawab ortu", "urusan ortu", "urusan orang tua",
        "izin orang tua", "izin ortu", "kontrol orang tua", "parental control", "family link",
        "orang tua lalai", "ortu lalai", "lepas tangan", "orang tua peduli", "ortu peduli",
    ],
    "Peran Pemerintah/Komdigi": [
        # Lembaga pemerintah, kementerian, pejabat, & regulasi resmi
        "pemerintah", "perintah", "komdigi", "menkomdigi", "kominfo", "menkominfo",
        "menteri", "bu menteri", "pak menteri", "mutia", "meutya", "kementerian",
        "presiden", "pejabat", "dpr", "aparat", "lembaga", "regulator",
        "kebijakan pemerintah", "kebijakan komdigi", "kebijakan", "regulasi",
        "peraturan", "aturan pemerintah", "aturan", "undang undang", "uu", "pp tunas",
        "hukum", "sanksi", "menindak", "tindak tegas",
        "pemblokiran", "memblokir", "diblokir", "keblokir", "blokir",
        "pelarangan", "melarang", "dilarang", "larangan",
        "pembatasan", "membatasi", "dibatasi", "dibatasin", "batasin",
        "verifikasi ktp", "verifikasi nik", "verifikasi wajah", "ktp", "nik",
        "memberantas", "berantas", "tutup judol", "blokir judol", "urus judol",
        "judi online", "judol", "pinjol", "korupsi", "koruptor",
    ],
    "Efektivitas Kebijakan": [
        # Evaluasi efektivitas, keberhasilan/kegagalan, dampak, & celah teknis implementasi
        "efektif", "efektivitas", "tidak efektif", "kurang efektif", "efisien",
        "berhasil", "sukses", "gagal", "sia sia", "percuma", "mustahil",
        "solutif", "solusi", "jalan keluar", "dampak kebijakan", "dampak",
        "penerapan", "menerapkan", "diterapkan", "terap", "implementasi",
        "realistis", "masuk akal", "berguna", "bermanfaat", "tidak guna",
        "tetap bisa", "tetap saja", "masih bisa",
        # Celah teknis / bypass / manipulasi umur pada kebijakan pembatasan
        "vpn", "memalsukan", "palsukan", "dipalsukan", "palsu", "manipulasi",
        "mengakali", "ngakali", "akalan", "bobol", "tembus", "lolos",
        "bypass", "bypas", "ganti tahun", "ganti umur", "ubah umur", "ubah usia",
        "ganti usia", "tuakan", "tanggal lahir", "tahun lahir",
        "akun", "nonaktif", "dinonaktifkan", "bertahap", "tahap",
        "verifikasi umur", "verifikasi usia",
    ],
}

# B. Aturan Kontekstual untuk Kata Ambigu (Contextual Rules - Poin 4 Revisi)
# Setiap tuple berisi: (daftar_kata_pemicu, daftar_kata_konteks_wajib)
# Kata pemicu HANYA akan memetakan komentar ke aspek terkait apabila di dalam komentar
# juga terdapat minimal satu kata dari `daftar_kata_konteks_wajib`.
ASPECT_CONTEXTUAL_RULES: dict[str, list[tuple[list[str], list[str]]]] = {
    "Perlindungan Anak": [
        (
            # Kata pemicu ambigu: perangkat ("hp", "gadget", "gawai")
            ["hp", "gadget", "gawai"],
            # Wajib didampingi konteks anak / dampak / bahaya / usia
            [
                "anak", "bocil", "bocah", "remaja", "adik", "adek", "siswa", "pelajar", "sekolah",
                "kecil", "umur", "usia", "16", "mental", "psikis", "otak", "mata", "candu",
                "kecanduan", "ketagihan", "bahaya", "rusak", "merusak", "dampak", "lindung",
                "dewasa", "porno", "pornografi", "bully", "belajar",
            ],
        ),
        (
            # Kata pemicu ambigu: "konten", "tonton", "tontonan", "dewasa"
            ["konten", "tonton", "tontonan", "dewasa"],
            [
                "anak", "bocil", "bocah", "remaja", "umur", "usia", "16", "bahaya", "rusak",
                "porno", "pornografi", "seksual", "sensitif", "negatif", "positif", "edukasi",
                "belajar", "mental", "moral", "akhlak", "lindung", "aman", "filter", "sensor",
            ],
        ),
        (
            # Kata pemicu ambigu: "usia", "umur" (agar "untung umur gw 20" tidak masuk Perlindungan Anak)
            ["usia", "umur"],
            [
                "anak", "bocil", "bocah", "remaja", "kecil", "dini", "bawah", "batas", "16",
                "17", "18", "sekolah", "sd", "smp", "mental", "bahaya", "lindung", "belum cukup",
                "cukup umur", "sesuai usia", "perkembangan",
            ],
        ),
        (
            # Kata pemicu ambigu: "sekolah", "belajar", "ajar" (agar "kurang ajar" atau "esok sekolah" tidak masuk)
            ["sekolah", "belajar", "ajar"],
            [
                "anak", "bocil", "siswa", "pelajar", "tugas", "materi", "edukasi", "guru",
                "buku", "fokus", "ilmu", "pendidikan", "pintar", "pinter", "cerdas",
                "medsos", "sosmed", "youtube", "yt", "tiktok", "hp", "dampak", "ganggu",
            ],
        ),
    ],
    "Peran Orang Tua": [
        (
            # Kata pemicu ambigu: "beli hp", "belikan hp", "kasih hp", "pegang hp"
            # (Agar "aku nabung beli hp sendiri kok lu yg ngatur" TIDAK masuk Peran Orang Tua!)
            ["beli hp", "belikan hp", "beliin hp", "kasih hp", "pegang hp", "pake hp", "pakai hp", "main hp"],
            [
                "orang tua", "orangtua", "ortu", "ibu", "ayah", "bapak", "mama", "papa", "emak",
                "bunda", "keluarga", "anak", "bocil", "bocah", "adik", "adek", "bayi",
                "awasi", "pengawasan", "didik", "asuh", "damping", "pantau", "kontrol",
                "izin", "larang", "tanggung jawab", "lalai",
            ],
        ),
        (
            # Kata pemicu ambigu: "ibu", "bapak", "ayah", "mama", "papa", "emak", "bunda", "keluarga", "rumah", "guru"
            # (Agar "ibu menteri", "bapak yg pake kacamata", "nangis di rumah", typo "kebapa" TIDAK salah masuk Peran Orang Tua)
            ["ibu", "bapak", "ayah", "mama", "papa", "emak", "mamak", "mamah", "bunda", "wali", "keluarga"],
            [
                "orang tua", "ortu", "asuh", "pola asuh", "didik", "mendidik",
                "awasi", "mengawasi", "pengawasan", "pantau", "memantau",
                "damping", "mendampingi", "bimbing", "kontrol", "tanggung jawab", "lalai",
                "sibuk", "peduli", "izin", "kasih hp", "belikan hp", "pegang hp",
                "akun bapak", "akun ibu", "akun emak", "ktp bapak", "ktp ibu", "ktp emak", "hp emak", "hp ibu",
                "anak saya", "anak ku", "anakku", "melawan ibu", "lawan ibu", "lawan ortu",
            ],
        ),
    ],
    "Peran Pemerintah/Komdigi": [
        (
            # Kata pemicu ambigu: "negara", "indonesia", "indo", "rakyat", "masyarakat", "demo", "atur", "urus", "batas", "larang", "bijak"
            ["negara", "indonesia", "indo", "rakyat", "masyarakat", "demo", "atur", "ngatur", "urus", "ngurus", "urusin", "ngurusin", "batas", "larang", "bijak", "blok", "block", "ban", "banned"],
            [
                "pemerintah", "perintah", "komdigi", "kominfo", "menteri", "presiden", "dpr",
                "pejabat", "kebijakan", "peraturan", "aturan", "regulasi", "uu", "undang",
                "hukum", "sanksi", "blokir", "diblokir", "dibatasi", "pembatasan", "dilarang",
                "medsos", "sosmed", "media sosial", "platform", "aplikasi", "akun",
                "youtube", "tiktok", "roblox", "ig", "game", "judol", "judi", "korupsi",
                "ktp", "nik", "verifikasi", "16 tahun", "anak", "bocil",
            ],
        ),
    ],
    "Efektivitas Kebijakan": [
        (
            # Kata pemicu ambigu: nama platform ("youtube", "yt", "tiktok", "ig", "instagram", "facebook",
            # "roblox", "game", "ff", "free fire", "epep", "ml", "mobile legends", "wa", "whatsapp",
            # "sosmed", "medsos", "media sosial", "sosial media", "platform", "aplikasi", "apk", "online", "hapus", "streak")
            [
                "youtube", "youtube kids", "yt", "tiktok", "tik tok", "ig", "instagram",
                "facebook", "fb", "whatsapp", "wa", "roblox", "game", "free fire", "ff",
                "epep", "ml", "mobile legends", "minecraft", "sosmed", "medsos",
                "media sosial", "sosial media", "platform", "aplikasi", "apk", "online",
                "hapus", "streak", "login", "akses", "sistem", "guna", "manfaat", "hasil",
            ],
            # Wajib didampingi konteks kebijakan / pembatasan / pemblokiran / dampak / efektivitas / verifikasi / edukasi
            [
                "blokir", "diblokir", "keblokir", "blok", "block", "ban", "banned",
                "batas", "dibatasi", "dibatasin", "batasin", "pembatasan",
                "larang", "dilarang", "pelarangan", "hapus", "dihapus", "nonaktif", "tutup",
                "aturan", "peraturan", "kebijakan", "regulasi", "pemerintah", "perintah", "komdigi",
                "efektif", "berhasil", "gagal", "percuma", "sia", "solusi", "dampak", "efek",
                "terap", "implementasi", "tahap", "bertahap",
                "verifikasi", "ktp", "nik", "wajah", "umur", "usia", "16", "tahun lahir", "tanggal lahir",
                "palsu", "bohong", "manipulasi", "vpn", "akal", "ngakali", "bobol", "tembus", "lolos", "aman",
                "belajar", "edukasi", "tugas", "sekolah", "materi", "informasi", "hiburan",
                "setuju", "tuju", "tolak", "dukung", "anak", "bocil", "remaja", "ortu", "orang tua",
            ],
        ),
    ],
}

# Gabungan seluruh kata kunci per aspek untuk kompatibilitas pencocokan klausa & highlight
ASPECT_KEYWORDS: dict[str, list[str]] = {
    asp: sorted(
        set(
            ASPECT_CORE_KEYWORDS[asp]
            + [kw for trig_list, _ in ASPECT_CONTEXTUAL_RULES.get(asp, []) for kw in trig_list]
        ),
        key=len,
        reverse=True,
    )
    for asp in ASPECT_CATEGORIES
}

_CORE_PATTERNS = {
    asp: re.compile(
        r"\b(?:"
        + "|".join(re.escape(kw) for kw in sorted(set(kws), key=len, reverse=True))
        + r")\b"
    )
    for asp, kws in ASPECT_CORE_KEYWORDS.items()
}

_CONTEXTUAL_COMPILED = {
    asp: [
        (
            re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(set(trigs), key=len, reverse=True)) + r")\b"),
            re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(set(ctxs), key=len, reverse=True)) + r")\b"),
        )
        for trigs, ctxs in rules
    ]
    for asp, rules in ASPECT_CONTEXTUAL_RULES.items()
}


def match_aspect_in_text(aspect: str, combined_text: str) -> list[str]:
    """
    Memeriksa apakah teks memuat aspek tertentu berdasarkan:
    1. Kata kunci inti (`ASPECT_CORE_KEYWORDS[aspect]`), ATAU
    2. Kata kunci kontekstual (`ASPECT_CONTEXTUAL_RULES[aspect]`) yang WAJIB
       didampingi minimal satu kata konteks pendukung pada teks.
    Mengembalikan daftar kata kunci aspek yang cocok (atau list kosong jika tidak memenuhi syarat).
    """
    text_low = combined_text.lower()

    # Penanganan khusus homograf "ajar" pada ungkapan umpatan "kurang ajar"
    # agar "kurang ajar" tidak memicu aspek pendidikan/Perlindungan Anak
    text_clean = re.sub(r"\bkurang\s+ajar\b", "umpatan_kasar", text_low)
    # Penanganan khusus kata seru Arab "wa syukurillah" agar "wa" tidak dianggap WhatsApp
    text_clean = re.sub(r"\bwa\s+syukur\w*\b", "syukur", text_clean)

    matched = set(_CORE_PATTERNS[aspect].findall(text_clean))

    for trig_pat, ctx_pat in _CONTEXTUAL_COMPILED.get(aspect, []):
        trig_hits = trig_pat.findall(text_clean)
        if trig_hits and ctx_pat.search(text_clean):
            matched.update(trig_hits)

    return sorted(matched, key=len, reverse=True)


def detect_aspects_strict(raw_text: str, norm_text: str, stem_text: str) -> dict[str, list[str]]:
    """
    Mendeteksi kategori aspek (dari 4 aspek) beserta daftar kata kunci pemicunya
    dengan aturan konteks yang diperketat (Revisi Poin 4).
    """
    combined = f"{str(raw_text).lower()} {norm_text} {stem_text}"
    result: dict[str, list[str]] = {}
    for asp in ASPECT_CATEGORIES:
        kws = match_aspect_in_text(asp, combined)
        if kws:
            result[asp] = kws
    return result


def export_inset_tsv(output_dir: Path | None = None):
    """
    Mengekspor:
    1. Kamus InSet aktif (`inset_positive.tsv` & `inset_negative.tsv`)
    2. Tabel dokumentasi modifikasi & tambahan kata/bobot terhadap InSet asli
       (`inset_modifikasi_dan_tambahan.csv` & `inset_modifikasi_dan_tambahan.xlsx`)
    3. Ringkasan metadata sumber asli Kamus InSet (`inset_sumber_asli_metadata.csv`)
    """
    target_dir = Path(output_dir) if output_dir else CURRENT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    pos_path = target_dir / "inset_positive.tsv"
    with open(pos_path, "w", encoding="utf-8") as f:
        f.write("word\tweight\n")
        for w, val in sorted(inset_positive.items()):
            f.write(f"{w}\t{val}\n")

    neg_path = target_dir / "inset_negative.tsv"
    with open(neg_path, "w", encoding="utf-8") as f:
        f.write("word\tweight\n")
        for w, val in sorted(inset_negative.items()):
            f.write(f"{w}\t{val}\n")

    mod_df = pd.DataFrame(INSET_MODIFICATION_LOG)
    mod_csv = target_dir / "inset_modifikasi_dan_tambahan.csv"
    mod_xlsx = target_dir / "inset_modifikasi_dan_tambahan.xlsx"
    mod_df.to_csv(mod_csv, index=False, encoding="utf-8-sig")
    mod_df.to_excel(mod_xlsx, index=False)

    meta_df = pd.DataFrame([
        {
            **INSET_SOURCE_METADATA,
            "jumlah_kata_unik_asli_positif": len(inset_original_positive),
            "jumlah_kata_unik_asli_negatif": len(inset_original_negative),
            "jumlah_kata_overlap_asli": len(set(inset_original_positive) & set(inset_original_negative)),
            "jumlah_entri_aktif_positif": len(inset_positive),
            "jumlah_entri_aktif_negatif": len(inset_negative),
            "jumlah_modifikasi_dan_tambahan_tercatat": len(mod_df),
        }
    ])
    meta_df.to_csv(target_dir / "inset_sumber_asli_metadata.csv", index=False, encoding="utf-8-sig")

    return pos_path, neg_path


if __name__ == "__main__":
    p_file, n_file = export_inset_tsv()
    print(f"Kamus InSet Asli dimuat dari:\n- {ORIGINAL_POS_TSV} ({len(inset_original_positive)} kata unik)\n- {ORIGINAL_NEG_TSV} ({len(inset_original_negative)} kata unik)")
    print(f"Kamus InSet Aktif disimpan di:\n- {p_file} ({len(inset_positive)} entri)\n- {n_file} ({len(inset_negative)} entri)")
    print(f"Catatan modifikasi & tambahan disimpan di:\n- {CURRENT_DIR / 'inset_modifikasi_dan_tambahan.csv'} ({len(INSET_MODIFICATION_LOG)} entri)")
