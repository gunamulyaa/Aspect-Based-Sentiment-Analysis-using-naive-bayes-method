# -*- coding: utf-8 -*-
"""
Kamus InSet (Indonesian Sentiment Lexicon) & Pemetaan 4 Kategori Aspek
Acuan Revisi Bab I-III:
1. Empat kategori aspek: Perlindungan Anak, Peran Orang Tua, Peran Pemerintah/Komdigi, Efektivitas Kebijakan
   (mencakup kata dasar, kata berimbuhan, dan bentuk kata setelah stemming Sastrawi).
2. Kamus InSet (Fajri Koto & Gemala Y. Rahmaningtyas, 2017) dengan bobot terukur (-5 s.d. +5).
3. Mempertahankan kata negasi (tidak, bukan, jangan, belum, tanpa, kurang, tak).
"""

from pathlib import Path

# ==============================================================================
# 1. PEMETAAN 4 KATEGORI ASPEK (BENTUK DASAR, BERIMBUHAN, & HASIL STEMMING)
# ==============================================================================

ASPECT_CATEGORIES = [
    "Perlindungan Anak",
    "Peran Orang Tua",
    "Peran Pemerintah/Komdigi",
    "Efektivitas Kebijakan",
]

ASPECT_KEYWORDS = {
    "Perlindungan Anak": [
        # Bentuk dasar & berimbuhan
        "anak", "anak anak", "bocil", "bocah", "remaja", "generasi",
        "bawah umur", "dibawah umur", "batas usia", "batas umur",
        "usia anak", "umur anak", "usia dini", "16 tahun", "enam belas tahun",
        "anak kecil", "adik", "adek", "pelajar", "siswa", "sekolah", "belajar",
        "perlindungan", "melindungi", "lindungi", "terlindung", "proteksi",
        "jaga anak", "menjaga anak", "keselamatan", "selamat",
        "keamanan anak", "aman bagi anak", "aman untuk anak",
        "mental", "psikologis", "psikis", "kesehatan mental", "sehat mental",
        "tumbuh kembang", "perkembangan", "moral", "akhlak", "karakter", "otak",
        "kecanduan", "candu", "ketagihan", "ketergantungan",
        "konten", "tontonan", "dewasa", "pornografi", "porno", "seksual",
        "cyberbullying", "bullying", "perundungan", "predator", "pelecehan",
        "eksploitasi", "korban", "bahaya", "berbahaya", "rusak", "merusak",
        "masa depan", "depan anak", "anak bangsa", "gadget", "hp", "gawai",
        "usia", "umur",
        # Bentuk kata setelah stemming Sastrawi
        "ajar", "lindung", "tonton", "leceh", "kembang", "tumbuh", "rundung",
        "ank", "cil", "sd", "smp",
    ],
    "Peran Orang Tua": [
        # Bentuk dasar & berimbuhan
        "orang tua", "orangtua", "ortu", "ortunya", "ibu", "ayah", "bapak",
        "mama", "papa", "emak", "bunda", "wali", "keluarga", "rumah", "guru",
        "pengasuhan", "mengasuh", "pola asuh", "mendidik", "pendidikan",
        "pengawasan", "mengawasi", "awasi", "diawasi", "ngawasin",
        "pendampingan", "mendampingi", "dampingi", "didampingi",
        "pemantauan", "memantau", "pantau", "dipantau", "mengontrol", "kontrol",
        "membimbing", "bimbingan", "bimbing", "membina", "bina",
        "tanggung jawab", "peran orang tua", "peran ortu",
        "lepas tangan", "lalai", "orang tua peduli", "ortu peduli",
        "kasih hp", "beli hp", "belikan hp", "pegang hp",
        "izin orang tua", "izin ortu",
        # Bentuk kata setelah stemming Sastrawi
        "asuh", "didik", "damping", "biarin", "dibiarin", "manja",
    ],
    "Peran Pemerintah/Komdigi": [
        # Bentuk dasar & berimbuhan
        "pemerintah", "komdigi", "menkomdigi", "kominfo", "menkominfo",
        "menteri", "kementerian", "presiden", "negara", "pejabat", "dpr",
        "aparat", "lembaga", "regulator", "kebijakan", "regulasi",
        "peraturan", "aturan", "mengatur", "undang undang", "uu", "pp tunas",
        "hukum", "sanksi", "menindak", "tindak",
        "pemblokiran", "memblokir", "diblokir", "keblokir", "blokir",
        "pelarangan", "melarang", "dilarang", "larangan", "larang",
        "pembatasan", "membatasi", "dibatasi", "batas",
        "verifikasi", "ktp", "nik", "fasilitasi",
        "memberantas", "berantas", "mengurus", "urus",
        "judol", "judi online", "judi", "pinjol", "korupsi", "koruptor",
        "demo", "rakyat", "masyarakat", "indonesia", "indo",
        # Bentuk kata setelah stemming Sastrawi & variasi
        "perintah", "bijak", "atur", "ngatur", "blok", "block", "ban", "banned",
        "jabat", "batasin", "dibatasin", "dibatesin", "urusin", "ngurusin",
    ],
    "Efektivitas Kebijakan": [
        # Bentuk dasar & berimbuhan
        "efektif", "efektivitas", "efisien", "berhasil", "sukses", "gagal",
        "sia sia", "percuma", "mustahil", "solutif", "solusi",
        "jalan keluar", "jalan pintas", "dampak", "efek",
        "penerapan", "menerapkan", "implementasi", "realistis", "masuk akal",
        "berguna", "bermanfaat", "tetap saja", "tetap bisa",
        "vpn", "memalsukan", "palsukan", "dipalsukan", "palsu", "bohong",
        "mengakali", "ngakali", "akalan", "akal", "bobol", "tembus", "lolos",
        "bypass", "bypas", "ganti tahun", "ganti umur", "ubah umur",
        "tanggal lahir", "tahun lahir", "akun", "nonaktif", "hapus", "uninstal",
        "akses", "buka", "login", "streak", "roblox", "epep", "free fire",
        "ff", "ml", "mobile legends", "minecraft", "game",
        "sosmed", "medsos", "media sosial", "sosial media",
        "tiktok", "tik tok", "youtube", "youtube kids", "yt",
        "ig", "instagram", "facebook", "whatsapp", "wa",
        "platform", "aplikasi", "apk", "online", "internet", "digital",
        # Bentuk kata setelah stemming Sastrawi
        "hasil", "sia", "terap", "sistem", "tahap", "uji", "permanen",
        "guna", "manfaat",
    ],
}


# ==============================================================================
# 2. KAMUS INSET (INDONESIAN SENTIMENT LEXICON)
# ==============================================================================
# Bobot positif (+1 s.d. +5) dan negatif (-1 s.d. -5) sesuai rancangan InSet,
# mencakup bentuk dasar dan bentuk hasil stemming Sastrawi.

inset_positive = {
    # Bobot +5 (Sangat Positif)
    "setuju": 5,
    "tuju": 5,
    "hebat": 5,
    "terbaik": 5,
    "alhamdulillah": 5,
    # Bobot +4 (Positif Kuat)
    "dukung": 4,
    "mendukung": 4,
    "bagus": 4,
    "baik": 4,
    "mantap": 4,
    "keren": 4,
    "top": 4,
    "syukur": 4,
    "senang": 4,
    "bahagia": 4,
    "cinta": 4,
    "aman": 4,
    "selamat": 4,
    "lindung": 4,
    "lindungi": 4,
    "melindungi": 4,
    "terlindung": 4,
    "efektif": 4,
    "efisien": 4,
    "berhasil": 4,
    "sukses": 4,
    "manfaat": 4,
    "bermanfaat": 4,
    "solutif": 4,
    "positif": 4,
    "sehat": 4,
    "cerdas": 4,
    "pintar": 4,
    "tepat": 4,
    "adil": 4,
    "terima kasih": 4,
    "makasih": 4,
    "emoji suka": 4,
    "emoji_suka": 4,
    "emoji tuju": 4,
    "emoji_setuju": 4,
    "emoji keren": 4,
    "emoji_keren": 4,
    # Bobot +3 (Positif Sedang)
    "suka": 3,
    "solusi": 3,
    "kreatif": 3,
    "cocok": 3,
    "layak": 3,
    "jelas": 3,
    "tegas": 3,
    "peduli": 3,
    "mudah": 3,
    "gampang": 3,
    "nyaman": 3,
    "tenang": 3,
    "tertib": 3,
    "benar": 3,
    "betul": 3,
    "penting": 3,
    "bantu": 3,
    "membantu": 3,
    "jaga": 3,
    "menjaga": 3,
    "seru": 3,
    "asik": 3,
    "enak": 3,
    # Bobot +2 (Positif Ringan)
    "semoga": 2,
    "moga": 2,
}

inset_negative = {
    # Bobot -5 (Sangat Negatif)
    "tidak tuju": -5,
    "tidak setuju": -5,
    "tolak": -5,
    "menolak": -5,
    "tidak efektif": -5,
    "hancur": -5,
    "merusak": -5,
    "gagal": -5,
    "percuma": -5,
    "sia sia": -5,
    "berbahaya": -5,
    "bodoh": -5,
    "bego": -5,
    "tolol": -5,
    "jahat": -5,
    "depresi": -5,
    "korupsi": -5,
    "koruptor": -5,
    "porno": -5,
    "pornografi": -5,
    "leceh": -5,
    "pelecehan": -5,
    "predator": -5,
    "bullying": -5,
    "cyberbullying": -5,
    "eksploitasi": -5,
    # Bobot -4 (Negatif Kuat)
    "jangan blokir": -4,
    "jangan batas": -4,
    "jangan hapus": -4,
    "jangan larang": -4,
    "tidak adil": -4,
    "tidak jelas": -4,
    "tidak guna": -4,
    "tidak terima": -4,
    "tidak suka": -4,
    "tidak aman": -4,
    "buruk": -4,
    "jelek": -4,
    "parah": -4,
    "kacau": -4,
    "rusak": -4,
    "sia": -4,
    "bahaya": -4,
    "negatif": -4,
    "rugi": -4,
    "merugikan": -4,
    "beban": -4,
    "bohong": -4,
    "palsu": -4,
    "tipu": -4,
    "hoax": -4,
    "hoaks": -4,
    "curang": -4,
    "langgar": -4,
    "gila": -4,
    "kejam": -4,
    "susah": -4,
    "sulit": -4,
    "ribet": -4,
    "sedih": -4,
    "nangis": -4,
    "kecewa": -4,
    "marah": -4,
    "kesal": -4,
    "benci": -4,
    "resah": -4,
    "stres": -4,
    "stress": -4,
    "candu": -4,
    "kecanduan": -4,
    "ketagihan": -4,
    "judol": -4,
    "judi": -4,
    "pinjol": -4,
    "rundung": -4,
    "perundungan": -4,
    "lalai": -4,
    "lepas tangan": -4,
    "emoji sedih": -4,
    "emoji_sedih": -4,
    "emoji marah": -4,
    "emoji_marah": -4,
    # Bobot -3 (Negatif Sedang)
    "aneh": -3,
    "kocak": -3,
    "lebay": -3,
    "norak": -3,
    "paksa": -3,
    "repot": -3,
    "lambat": -3,
    "lemot": -3,
    "telat": -3,
    "salah": -3,
    "masalah": -3,
    "ganggu": -3,
    "mengganggu": -3,
    "takut": -3,
    "cemas": -3,
    "capek": -3,
    "cape": -3,
    "bosan": -3,
    "malas": -3,
    "males": -3,
    "malu": -3,
    "manja": -3,
    "bobol": -3,
    "tembus": -3,
    "ngakali": -3,
    "bypass": -3,
    "bypas": -3,
    "anjir": -3,
    "anj": -3,
    "cok": -3,
    "hadeh": -3,
    "halah": -3,
    "kasihan": -3,
    # Bobot -2 (Negatif Ringan)
    "aduh": -2,
}

# Kompatibilitas dengan struktur pemanggilan Dictionary.py
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
}


def export_inset_tsv(output_dir: Path | None = None):
    """Simpan berkas TSV Kamus InSet (inset_positive.tsv & inset_negative.tsv)."""
    target_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
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

    return pos_path, neg_path


if __name__ == "__main__":
    p_file, n_file = export_inset_tsv()
    print(f"Kamus InSet disimpan di:\n- {p_file}\n- {n_file}")
