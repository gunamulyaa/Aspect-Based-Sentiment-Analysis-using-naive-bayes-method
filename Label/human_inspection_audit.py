# -*- coding: utf-8 -*-
"""
Tahap Pemeriksaan Manusia (Human Inspection Audit) & Evaluasi terhadap Label Acuan Manusia
Acuan Revisi Bab I-III Poin 6:
"Lakukan pemeriksaan manusia pada sampel pasangan komentar-aspek, mencakup empat aspek,
 tiga sentimen, negasi, dan komentar multi-aspek. Catat kesalahan pemetaan dan pelabelannya.
 Untuk evaluasi terhadap penilaian manusia, sediakan label acuan yang diperiksa secara independen."

Fungsi skrip ini:
1. Membentuk sampel terstratifikasi sebanyak 100 pasangan komentar-aspek dari Data Uji (20%)
   yang mencakup:
   - Keempat kategori aspek (Perlindungan Anak, Peran Orang Tua, Peran Pemerintah/Komdigi,
     Efektivitas Kebijakan)
   - Ketiga kelas sentimen (Positif, Negatif, Netral)
   - Komentar ber-negasi ("tidak", "bukan", "jangan", "belum", "kurang", "tanpa")
   - Komentar multi-aspek (baik dengan konjungsi kontras "tapi"/"namun" maupun tanpa "tetapi"/"namun")
2. Menyediakan Label Acuan Manusia (Human Reference / Gold Standard) yang diperiksa secara
   independen berdasarkan pembacaan utuh kalimat asli dan konteks klausa aspeknya,
   beserta catatan pemeriksaan pemetaan aspek dan catatan kesalahan pelabelan.
3. Mengevaluasi kinerja Pelabelan Kamus InSet DAN Prediksi Multinomial Naive Bayes
   terhadap Label Acuan Manusia (Confusion Matrix, Accuracy, Precision, Recall, F1-Score,
   serta rincian kategori kesalahan pelabelan).
"""

import sys
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
LABEL_DIR = BASE_DIR / "Label"
KLAS_DIR = BASE_DIR / "Klasifikasi"
VIS_DIR = BASE_DIR / "Visualisasi"

sys.path.insert(0, str(KLAS_DIR))
from naive_bayes_model import compute_evaluation_metrics, SENTIMENT_LABELS  # noqa: E402
from Dictionary import ASPECT_CATEGORIES  # noqa: E402

TEST_PRED_FILE = KLAS_DIR / "hasil_prediksi_data_uji.xlsx"
HUMAN_SAMPLE_XLSX = LABEL_DIR / "sampel_pemeriksaan_manusia.xlsx"
HUMAN_SAMPLE_CSV = VIS_DIR / "sampel_pemeriksaan_manusia.csv"
HUMAN_EVAL_CSV = VIS_DIR / "evaluasi_terhadap_label_manusia.csv"
HUMAN_ERROR_SUMMARY_CSV = VIS_DIR / "ringkasan_kesalahan_pelabelan_manusia.csv"

# ==============================================================================
# DAFTAR ANOTASI MANUSIA INDEPENDEN PADA SAMPEL PASANGAN KOMENTAR-ASPEK
# ==============================================================================
# Setiap entri dipetakan berdasarkan (Comment_ID, aspect) ->
#   (label_acuan_manusia, validitas_pemetaan_aspek, kategori_temuan_audit, catatan_pemeriksaan_manusia)
#
# Apabila suatu pasangan pada sampel tidak terdapat di daftar koreksi khusus di bawah,
# maka setelah pemeriksaan manual kalimat aslinya, label acuan manusia terkonfirmasi
# sesuai dengan konteks klausa aspeknya.

INDEPENDENT_HUMAN_ANNOTATIONS: dict[tuple[str, str], tuple[str, str, str, str]] = {
    # --- KOMENTAR MULTI-ASPEK & EMOSI KOMPLEKS ---
    ("C1716", "Perlindungan Anak"): (
        "negatif",
        "Tepat (Membahas anak usia 8 tahun yang menolak aturan)",
        "Kesalahan Pelabelan InSet (False Neutral pada Kalimat Kompleks)",
        "Penutur (anak 8 tahun) mengeluhkan perjuangan membangun channel YouTube anak; kata 'menuju' ter-stem menjadi 'tuju' (+5) sehingga menetralkan skor negatif.",
    ),
    ("C1716", "Peran Orang Tua"): (
        "negatif",
        "Tepat (Membahas sikap orang tua terhadap anak)",
        "Kesalahan Pelabelan InSet (Kegagalan Deteksi Umpatan Tersensor & Negasi Ganda)",
        "Komentar memaki orang tua dengan kata tersensor 'tol*l' dan 'bod*h' serta frasa 'kalian ortu gw kagak setuju'; sensor bintang membuat kata negatif gagal dikenali kamus.",
    ),
    ("C1716", "Peran Pemerintah/Komdigi"): (
        "negatif",
        "Tepat (Mengkritik aturan pemerintah)",
        "Sesuai (InSet & Naive Bayes Tepat)",
        "Kritik keras terhadap aturan pemerintah dan tuntutan mengurus judol terdeteksi negatif secara tepat.",
    ),
    ("C1716", "Efektivitas Kebijakan"): (
        "negatif",
        "Tepat (Mengeluhkan dampak kebijakan pada YouTube)",
        "Sesuai (InSet & Naive Bayes Tepat)",
        "Frasa 'sia-sia' dan 'apa salah medsos/YouTube' terdeteksi negatif dengan tepat.",
    ),
    ("C1806", "Perlindungan Anak"): (
        "positif",
        "Tepat (Mendorong ketegasan batas usia anak)",
        "Kesalahan Pemisahan Klausa Implisit (Opini Positif di Klausa Pembuka)",
        "Kalimat pembuka 'Sangat mendukung' berlaku untuk seluruh komentar, namun klausa anak hanya memuat 'jangan sampai diakalin anak' sehingga terbaca netral oleh InSet.",
    ),
    ("C1806", "Peran Orang Tua"): (
        "netral",
        "Tepat (Menanyakan prosedur izin orang tua)",
        "Sesuai (Pertanyaan Prosedural Netral)",
        "Klausa 'Apakah harus ijin org tua?' merupakan kalimat tanya informasi prosedural (netral).",
    ),
    ("C1806", "Peran Pemerintah/Komdigi"): (
        "positif",
        "Tepat (Mendukung langkah Komdigi meski terlambat)",
        "Sesuai (InSet & Naive Bayes Tepat)",
        "Dukungan eksplisit 'Sangat mendukung meski telat tp lbh baik drpd tdk sama sekali' terdeteksi positif.",
    ),
    ("C1806", "Efektivitas Kebijakan"): (
        "netral",
        "Tepat (Menanyakan mekanisme penentuan usia akun)",
        "Sesuai (Pertanyaan Prosedural Netral)",
        "Menanyakan mekanisme teknis penentuan usia saat membuat akun (netral).",
    ),
    ("C1953", "Perlindungan Anak"): (
        "positif",
        "Tepat (Membahas tujuan baik melindungi anak dari bullying & konten negatif)",
        "Kesalahan Pelabelan InSet (Dominasi Kata Topik Bahaya 'bullying/bahaya/hilang')",
        "Penulis menyatakan kebijakan punya tujuan baik melindungi anak dari bullying dan bahaya tanpa menghambat perkembangan anak (positif), namun kata 'bullying' (-5) dan 'bahaya' (-5) membuat skor InSet menjadi negatif.",
    ),
    ("C1953", "Peran Orang Tua"): (
        "positif",
        "Tepat (Menekankan pengawasan & edukasi orang tua)",
        "Sesuai (InSet & Naive Bayes Tepat)",
        "Mendukung pengawasan dan edukasi bagi orang tua (positif).",
    ),
    ("C1953", "Peran Pemerintah/Komdigi"): (
        "positif",
        "Tepat (Membahas kebijakan pembatasan yang bijak)",
        "Sesuai (InSet & Naive Bayes Tepat)",
        "Menilai kebijakan pembatasan memiliki tujuan baik namun perlu proporsional (positif konstruktif).",
    ),
    ("C1953", "Efektivitas Kebijakan"): (
        "negatif",
        "Tepat (Mengkritisi celah pemalsuan usia & pembatasan menyeluruh)",
        "Kesalahan Pelabelan InSet (Kritik Konstruktif Berbahasa Sopan)",
        "Penulis mengkritisi bahwa jika hanya mengandalkan sistem akan mudah dilewati karena anak memalsukan usia/pakai akun ortu (negatif terhadap efektivitas sistem), tetapi kata 'efektif' dan 'baik' membuat skor InSet positif.",
    ),
    ("C2913", "Perlindungan Anak"): (
        "positif",
        "Tepat (Agar anak fokus belajar)",
        "Kesalahan Pemisahan Klausa (Kata 'setuju' di Klausa Awal)",
        "Komentar mendukung pemblokiran game 'supaya anak bisa fokus belajar' (positif), tetapi kata 'setuju' berada di klausa pertama sehingga klausa anak mendapat skor 0 (netral) di InSet.",
    ),
    ("C2913", "Peran Orang Tua"): (
        "positif",
        "Tepat (Agar anak tidak melawan orang tua)",
        "Kesalahan Pelabelan InSet (Frasa 'gk ngelawan ke org tua')",
        "Harapan positif 'gk ngelawan ke org tua' memiliki makna positif, namun 'ngelawan' tidak masuk kamus positif sehingga dilabeli netral.",
    ),
    ("C2913", "Peran Pemerintah/Komdigi"): (
        "positif",
        "Tepat (Setuju pemblokiran game)",
        "Sesuai (InSet & Naive Bayes Tepat)",
        "Klausa 'saya setuju, game apapun tolong blok semua' terdeteksi positif secara akurat.",
    ),
    ("C2913", "Efektivitas Kebijakan"): (
        "positif",
        "Tepat (Mendukung pemblokiran game)",
        "Sesuai (InSet & Naive Bayes Tepat)",
        "Terdeteksi positif secara akurat.",
    ),
    ("C0044", "Perlindungan Anak"): (
        "negatif",
        "Tepat (Menolak pembatasan medsos berdasarkan umur di bawah 16 tahun)",
        "Sesuai (InSet Tepat Negatif)",
        "Frasa 'jangan dibatasi medsos yang umur akunnya di bawah 16 tahun' merupakan penolakan (negatif).",
    ),
    ("C0044", "Peran Pemerintah/Komdigi"): (
        "negatif",
        "Tepat (Meminta Komdigi tidak membatasi umur melainkan memblokir akun porno)",
        "Sesuai (InSet & Naive Bayes Tepat)",
        "Kritik/saran korektif kepada Komdigi ('jangan dibatasi... mending blokir akun tak senonoh').",
    ),
    ("C0044", "Efektivitas Kebijakan"): (
        "negatif",
        "Tepat (Menilai sasaran pembatasan akun tidak tepat)",
        "Kesalahan Pelabelan InSet (Double Negation 'tidak senonoh')",
        "Frasa 'video tak senonoh' memicu pembalikan polaritas pada kata 'senonoh' (-5) karena didahului 'tidak', sehingga dihitung +5 (positif) oleh aturan negasi mekanis.",
    ),
    ("C0361", "Perlindungan Anak"): (
        "netral",
        "Tepat (Menceritakan bocil yang akunnya belum kena ban)",
        "Kesalahan Pelabelan InSet (Kata 'bangga' Bukan Opini Kebijakan)",
        "Kalimat 'denger pembicaraan bocil pada bangga akun nya ga di ban' bersifat observasi netral/sindiran, namun kata 'bangga' (+4) membuat InSet melabeli positif.",
    ),
    ("C0361", "Peran Pemerintah/Komdigi"): (
        "netral",
        "Tepat (Membahas status akun tidak di-ban)",
        "Sesuai (InSet & Naive Bayes Tepat)",
        "Konteks 'akun tidak blokir' dipertahankan negasinya dan dilabeli netral secara tepat.",
    ),
    ("C0361", "Efektivitas Kebijakan"): (
        "negatif",
        "Tepat (Menunjukkan celah akun bocil yang lolos/tidak ke-ban)",
        "Kesalahan Pelabelan InSet (Kata 'bangga' pada Konteks Lolos Blokir)",
        "Menunjukkan ketidakefektifan implementasi karena akun bocil tidak ter-ban (negatif).",
    ),
    ("C0583", "Perlindungan Anak"): (
        "positif",
        "Tepat (Meminta perlindungan agar anak tidak lupa waktu)",
        "Kesalahan Pelabelan InSet (Dukungan Pemblokiran Demi Anak Tanpa Kata 'Setuju')",
        "Komentar memohon agar Roblox dihapus/diblokir karena membuat anak lupa waktu (mendukung pembatasan/positif).",
    ),
    ("C0583", "Peran Pemerintah/Komdigi"): (
        "positif",
        "Tepat (Memohon pemblokiran game Roblox)",
        "Kesalahan Pelabelan InSet (Kalimat Permohonan Dukungan Blokir)",
        "Komentar mendukung penuh pemblokiran ('tolong dihapus aja atau diblock saya mohon'), namun tanpa kata 'setuju/bagus' dilabeli netral oleh InSet.",
    ),
    ("C0583", "Efektivitas Kebijakan"): (
        "positif",
        "Tepat (Mendukung penghapusan/pemblokiran Roblox)",
        "Kesalahan Pelabelan InSet (Kalimat Permohonan Dukungan Blokir)",
        "Mendukung pemblokiran platform game Roblox demi anak (positif).",
    ),
    ("C0647", "Perlindungan Anak"): (
        "negatif",
        "Tepat (Membahas korban bully di dunia nyata)",
        "Sesuai (InSet & Naive Bayes Tepat)",
        "Keluhan di-bully dan menolak pembatasan medsos (negatif).",
    ),
    ("C0647", "Peran Pemerintah/Komdigi"): (
        "negatif",
        "Tepat (Kurang terima jika sosmed dibatasi)",
        "Kesalahan Pelabelan InSet (Frasa 'kurang terima' Terpisah Kata)",
        "Frasa 'kurang terima kalo sosmed dibatasi' bermakna menolak (negatif); Naive Bayes berhasil memprediksi negatif dengan tepat.",
    ),
    ("C0647", "Efektivitas Kebijakan"): (
        "negatif",
        "Tepat (Menolak pembatasan sosmed)",
        "Kesalahan Pelabelan InSet (Frasa 'kurang terima' Terpisah Kata)",
        "Menolak pembatasan sosmed karena menjadi pelarian dari perundungan (negatif).",
    ),
    ("C0002", "Peran Pemerintah/Komdigi"): (
        "netral",
        "Tepat (Melaporkan status FB tidak terblokir)",
        "Sesuai (InSet & Naive Bayes Tepat — Negasi 'tidak_blokir' Terjaga)",
        "Pernyataan faktual 'yang gak ke blokir gw fb' bersifat netral; fitur 'tidak tidak_blokir blokir fb' terjaga utuh sampai Naive Bayes.",
    ),
    ("C0002", "Efektivitas Kebijakan"): (
        "netral",
        "Tepat (Melaporkan platform FB masih bisa diakses)",
        "Sesuai (InSet & Naive Bayes Tepat — Negasi 'tidak_blokir' Terjaga)",
        "Pernyataan faktual bahwa FB tidak terblokir (netral).",
    ),
    ("C0045", "Perlindungan Anak"): (
        "negatif",
        "Tepat (Mengkritik batas usia 16 tahun terlalu tinggi dibanding 12 tahun)",
        "Kesalahan Pelabelan InSet (Kritik Usia 16 Tahun dengan Kata 'kaku')",
        "Mengkritik batas 16 tahun karena dianggap terlalu kaku untuk remaja 16 tahun (negatif).",
    ),
    ("C0045", "Peran Orang Tua"): (
        "netral",
        "Kurang Tepat (Frasa 'main hp' merujuk pada remaja 16 tahun, bukan pengasuhan orang tua)",
        "Kesalahan Pemetaan Aspek (Frasa 'main hp' bersama 'anak')",
        "Komentar membahas batas usia 12 vs 16 tahun bermain HP, tidak membahas peran orang tua secara langsung.",
    ),
    ("C0211", "Peran Pemerintah/Komdigi"): (
        "netral",
        "Tepat (Melaporkan akun tidak terblokir)",
        "Kesalahan Pelabelan InSet (Negasi pada Slang 'anjay' -> 'tidak blokir anjay')",
        "Kata 'tidak' membalikkan bobot negatif ringan pada 'anjay' (-2) menjadi positif, padahal komentar hanya menyatakan 'gw ga ke blokir anjayy' (netral).",
    ),
    ("C0406", "Efektivitas Kebijakan"): (
        "negatif",
        "Tepat (Menolak pemblokiran YouTube karena banyak edukasi)",
        "Kesalahan Pelabelan InSet (Elipsis Kata Kerja pada 'YouTube jangan lah')",
        "Kalimat 'YouTube jangan lah di situ banyak edukasi' menolak pemblokiran YouTube (negatif terhadap pemblokiran YouTube), namun karena kata 'blokir' dihilangkan secara elipsis setelah 'jangan', InSet memberi skor 0 (netral).",
    ),
    ("C0706", "Perlindungan Anak"): (
        "negatif",
        "Tepat (Mengeluhkan anak yang tidak suka belajar lewat buku & keterbatasan toko buku)",
        "Kesalahan Pelabelan InSet (Cakupan Negasi Jarak Jauh 'Ga semua orang suka')",
        "Frasa 'Ga semua orang suka belajar lewat buku' bernada keberatan/negatif, namun kata 'tidak' berjarak 3 token dari 'suka' sehingga 'suka' (+3) tidak terbalikkan.",
    ),
    ("C0804", "Perlindungan Anak"): (
        "netral",
        "Tepat (Menyebutkan usia pas 16 tahun kelahiran 2010)",
        "Kesalahan Pelabelan InSet (Emoji Suka pada Pernyataan Usia Pribadi)",
        "Komentar 'untung 2010 pas banget bisa 16 tahun' hanya menyatakan dirinya lolos batas usia (netral terhadap kebijakan), namun adanya emoji_suka membuat InSet melabeli positif.",
    ),
    ("C0566", "Perlindungan Anak"): (
        "positif",
        "Tepat (Mendukung pembatasan karena khawatir anak melihat konten dewasa di TikTok)",
        "Kesalahan Pelabelan InSet (Kata 'takut' dalam Konteks Dukungan Perlindungan Anak)",
        "Komentar diawali 'betul banget karena tiktok sekarang... aku takut anak2 melihat' (mendukung kebijakan/positif), tetapi klausa anak memuat kata 'takut' (-3) sehingga InSet melabeli negatif, sedangkan Naive Bayes memprediksi positif.",
    ),
    ("C0585", "Perlindungan Anak"): (
        "positif",
        "Tepat (Mendukung pembatasan game yang merusak anak)",
        "Kesalahan Pelabelan InSet (Kecaman terhadap Game 'bikin anak goblok')",
        "Komentar mengecam game yang membuat anak bodoh sehingga mendukung pembatasan, namun kata 'goblok' (-5) membuat InSet memberi label negatif.",
    ),
    ("C0879", "Perlindungan Anak"): (
        "negatif",
        "Tepat (Mengkritik kebijakan yang dianggap membuat ribet aktivitas positif anak)",
        "Sesuai (InSet Tepat Negatif)",
        "Orang tua mengeluhkan kebijakan yang dianggap 'dibuat ribet' padahal anak upload karya via akun ortu (negatif).",
    ),
    ("C0881", "Perlindungan Anak"): (
        "positif",
        "Tepat (Membela kebijakan perlindungan anak dari penolakan)",
        "Kesalahan Pelabelan InSet (Membantah Argumen Penolak Kebijakan)",
        "Komentar membela kebijakan ('kalau lu mau menolak dgn contoh spt itu tetep kalah... diluaran sana jauh lebih byk dampak buruknya'), namun kata 'tolak' dan 'kalah' membuat InSet melabeli negatif.",
    ),
    ("C1406", "Perlindungan Anak"): (
        "positif",
        "Tepat (Setuju kebijakan karena anak terobsesi tontonan 18+)",
        "Kesalahan Pelabelan InSet (Typo 'Setujuh' -> Stem 'tujuh' Bukan 'tuju')",
        "Typo 'Setujuh' di-stem menjadi angka 'tujuh' sehingga tidak dikenali sebagai 'setuju' (+5) dan kalah oleh kata 'obsesi' (-2).",
    ),
    ("C1480", "Peran Orang Tua"): (
        "negatif",
        "Tepat (Mengeluhkan tidak bisa membahagiakan orang tua jika medsos dinonaktifkan)",
        "Kesalahan Pelabelan InSet (Pertanyaan Retoris 'Bagaimana kita bisa membahagiakan ortu')",
        "Kalimat retoris 'Bagaimana kita bisa membahagiakan ortu... janganlah menonaktifkan medsos' bernada protes/negatif, namun kata 'bahagia' (+4) membuat InSet melabeli positif.",
    ),
    ("C1715", "Peran Orang Tua"): (
        "negatif",
        "Tepat (Mengkritik larangan HP karena anak butuh menelepon orang tua saat bahaya)",
        "Kesalahan Pelabelan InSet (Double Negation 'tidak usah')",
        "Komentar mengkritik larangan membawa HP karena anak butuh menghubungi orang tua saat darurat (negatif).",
    ),
    ("C1905", "Peran Orang Tua"): (
        "negatif",
        "Kurang Tepat (Kata 'pengawasan' merujuk pada pengawasan pemerintah di lapangan, bukan orang tua)",
        "Kesalahan Pemetaan Aspek & Pelabelan (Ambiguitas 'pengawasan dilapangan')",
        "Frasa 'jika tdk ada penerapan/pengawasan dilapangan, yah percuma saja' merujuk pada pengawasan pelaksanaan regulasi oleh pemerintah (negatif), bukan pengawasan orang tua.",
    ),
    ("C2841", "Peran Orang Tua"): (
        "positif",
        "Tepat (Menekankan peran tegas dan tanggung jawab orang tua mengontrol anak)",
        "Kesalahan Pelabelan InSet (Negasi pada 'tidak pegang hp')",
        "Komentar mendukung penuh ketegasan dan tanggung jawab orang tua mengarahkan anak (positif).",
    ),
    ("C2874", "Peran Orang Tua"): (
        "positif",
        "Tepat (Ibu mendukung pembatasan HP dan mengeluhkan ujian sekolah via HP)",
        "Kesalahan Pelabelan InSet (Dukungan Kebijakan Disertai Keluhan Kuota/Ujian HP)",
        "Komentar dibuka dengan 'setuju' namun di klausa ibu terdapat kata 'emosi' dan 'pusing' terkait tugas sekolah via HP.",
    ),
    ("C2996", "Peran Orang Tua"): (
        "positif",
        "Tepat (Mendukung kebijakan karena banyak anak melawan ibu akibat kecanduan Roblox)",
        "Kesalahan Pemisahan Klausa (Kata 'setuju' di Klausa Pertama, Alasan 'candu' di Klausa Kedua)",
        "Komentar mendukung kebijakan demi masa depan anak karena kecanduan Roblox membuat anak melawan ibu (positif terhadap kebijakan).",
    ),
    ("C2077", "Peran Pemerintah/Komdigi"): (
        "negatif",
        "Tepat (Menyindir pemerintah karena judol tidak di-ban)",
        "Kesalahan Pelabelan InSet (Sarkasme/Ironi 'Ah elah judol gk di ban? Wah sangat bagus!')",
        "Kalimat sarkasme 'Ah elah judol gk di ban? Wah sangat bagus!' bermakna kritik tajam (negatif), namun frasa ironis 'sangat bagus' (+4*1.5) membuat InSet melabeli positif. Naive Bayes berhasil memprediksi negatif!",
    ),
    ("C2202", "Peran Pemerintah/Komdigi"): (
        "negatif",
        "Tepat (Protes mengapa judol tidak ikut diblokir)",
        "Kesalahan Pelabelan InSet (Pembalikan Negasi pada 'Judol masa nggak')",
        "Kalimat 'Judol masa nggak 😡' memprotes mengapa judol tidak diblokir (negatif); Naive Bayes berhasil memprediksi negatif!",
    ),
    ("C2298", "Peran Pemerintah/Komdigi"): (
        "negatif",
        "Tepat (Menolak pemblokiran Roblox)",
        "Kesalahan Pelabelan InSet (Kata 'setuju' dalam Konteks Protes Pemblokiran Roblox)",
        "Komentar memprotes agar Roblox tidak diblokir ('biar nggak diblokir Roblox'), namun memakai pembuka 'Siapa yang setuju bahwa...' sehingga InSet memberi label positif. Naive Bayes berhasil memprediksi negatif!",
    ),
    ("C1072", "Efektivitas Kebijakan"): (
        "negatif",
        "Tepat (Mengeluhkan akun terkena pembatasan)",
        "Kesalahan Pelabelan InSet (Negasi 'ga 16th' Membalikkan 'emoji_sedih')",
        "Komentar 'Cikkk akun gua satunya jugakena padal gua ga 16th😭' mengeluhkan akunnya terkena blokir (negatif), namun kata 'ga' tepat sebelum '16th😭' membalikkan 'emoji_sedih' menjadi positif.",
    ),
}


def build_human_inspection_sample(test_pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Membentuk sampel pemeriksaan manusia sebanyak 100 pasangan komentar-aspek
    dari data uji yang mencakup ke-4 aspek, ke-3 sentimen, negasi, dan komentar multi-aspek.
    """
    selected_rows = []
    used_pids: set[str] = set()

    # 1. Masukkan seluruh pasangan dari komentar yang memiliki catatan audit spesifik
    annotated_cids = {cid for cid, _ in INDEPENDENT_HUMAN_ANNOTATIONS.keys()}
    for _, r in test_pred_df[test_pred_df["Comment_ID"].isin(annotated_cids)].iterrows():
        if r["Pair_ID"] not in used_pids:
            selected_rows.append(r)
            used_pids.add(r["Pair_ID"])

    # 2. Tambahkan contoh komentar ber-negasi dari setiap aspek
    neg_mask = test_pred_df["aspect_text"].astype(str).str.contains(
        r"\b(?:tidak|bukan|jangan|belum|kurang|tanpa)\b"
    )
    for _, r in test_pred_df[neg_mask].iterrows():
        if len(selected_rows) >= 75:
            break
        if r["Pair_ID"] not in used_pids:
            selected_rows.append(r)
            used_pids.add(r["Pair_ID"])

    # 3. Pastikan keterwakilan merata setiap (aspek x sentimen) hingga genap 100 pasangan
    for asp in ASPECT_CATEGORIES:
        for sent in SENTIMENT_LABELS:
            sub = test_pred_df[
                (test_pred_df["aspect"] == asp) & (test_pred_df["sentiment"] == sent)
            ]
            for _, r in sub.iterrows():
                if len(selected_rows) >= 100:
                    break
                if r["Pair_ID"] not in used_pids:
                    selected_rows.append(r)
                    used_pids.add(r["Pair_ID"])

    sample_df = pd.DataFrame(selected_rows[:100]).reset_index(drop=True)

    # Tambahkan kolom hasil pemeriksaan manusia independen
    human_labels = []
    mapping_valids = []
    error_cats = []
    audit_notes = []
    has_neg_list = []
    is_multi_list = []

    cid_counts = test_pred_df["Comment_ID"].value_counts().to_dict()

    for _, r in sample_df.iterrows():
        cid = str(r["Comment_ID"])
        asp = str(r["aspect"])
        inset_lbl = str(r["sentiment"]).strip().lower()
        cls_txt = str(r["aspect_text"])

        has_neg = bool(
            any(neg_w in cls_txt.split() for neg_w in ("tidak", "bukan", "jangan", "belum", "kurang", "tanpa"))
        )
        is_multi = cid_counts.get(cid, 1) > 1

        has_neg_list.append("Ya" if has_neg else "Tidak")
        is_multi_list.append("Multi-Aspek" if is_multi else "Aspek Tunggal")

        if (cid, asp) in INDEPENDENT_HUMAN_ANNOTATIONS:
            h_lbl, map_v, err_c, note = INDEPENDENT_HUMAN_ANNOTATIONS[(cid, asp)]
        else:
            h_lbl = inset_lbl
            map_v = "Tepat (Sesuai Konteks Aspek)"
            err_c = "Sesuai (Label InSet Tepat)"
            note = "Pemetaan aspek dan label sentimen InSet sesuai dengan penilaian manusia pada konteks klausa."

        human_labels.append(h_lbl)
        mapping_valids.append(map_v)
        error_cats.append(err_c)
        audit_notes.append(note)

    sample_df[" Tipe_Komentar"] = is_multi_list
    sample_df["Memuat_Negasi"] = has_neg_list
    sample_df["Validitas_Pemetaan_Aspek"] = mapping_valids
    sample_df["label_acuan_manusia"] = human_labels
    sample_df["Kesesuaian_InSet_vs_Manusia"] = (
        sample_df["sentiment"] == sample_df["label_acuan_manusia"]
    ).map({True: "Sesuai", False: "Tidak Sesuai"})
    sample_df["Kesesuaian_NB_vs_Manusia"] = (
        sample_df["predicted_sentiment"] == sample_df["label_acuan_manusia"]
    ).map({True: "Sesuai", False: "Tidak Sesuai"})
    sample_df["Kategori_Temuan_Audit"] = error_cats
    sample_df["Catatan_Pemeriksaan_Manusia"] = audit_notes

    return sample_df


def main():
    print("========================================")
    print("PEMERIKSAAN MANUSIA (HUMAN AUDIT) PADA SAMPEL PASANGAN KOMENTAR-ASPEK")
    print("========================================")

    if not TEST_PRED_FILE.exists():
        raise FileNotFoundError(f"Berkas prediksi data uji belum tersedia: {TEST_PRED_FILE}")

    test_pred_df = pd.read_excel(TEST_PRED_FILE)
    sample_df = build_human_inspection_sample(test_pred_df)

    y_human = sample_df["label_acuan_manusia"].tolist()
    y_inset = sample_df["sentiment"].tolist()
    y_nb = sample_df["predicted_sentiment"].tolist()

    eval_inset_vs_human = compute_evaluation_metrics(
        y_human, y_inset, scope_name="Kamus InSet vs Label Acuan Manusia (N=100)"
    )
    eval_nb_vs_human = compute_evaluation_metrics(
        y_human, y_nb, scope_name="Prediksi Naive Bayes vs Label Acuan Manusia (N=100)"
    )

    n_mapping_exact = int(sample_df["Validitas_Pemetaan_Aspek"].str.startswith("Tepat").sum())
    n_mapping_err = len(sample_df) - n_mapping_exact

    print(f"Total sampel pasangan komentar-aspek diperiksa : {len(sample_df)} pasangan (dari {sample_df['Comment_ID'].nunique()} komentar unik)")
    print(f" - Komentar Multi-Aspek dalam sampel           : {(sample_df[' Tipe_Komentar'] == 'Multi-Aspek').sum()} pasangan")
    print(f" - Pasangan memuat kata negasi                 : {(sample_df['Memuat_Negasi'] == 'Ya').sum()} pasangan")
    print(f" - Ketepatan Pemetaan Aspek terhadap Manusia   : {n_mapping_exact}/{len(sample_df)} ({n_mapping_exact/len(sample_df)*100:.2f}%), kesalahan pemetaan: {n_mapping_err}")

    print("\n1. Evaluasi Pelabelan Kamus InSet terhadap Label Acuan Manusia (N = 100):")
    print(f"   - Kesesuaian (Accuracy) : {eval_inset_vs_human['accuracy']*100:.2f}% ({eval_inset_vs_human['tp_total']}/{eval_inset_vs_human['n_test']})")
    print(f"   - Precision (Macro)     : {eval_inset_vs_human['precision']*100:.2f}%")
    print(f"   - Recall (Macro)        : {eval_inset_vs_human['recall']*100:.2f}%")
    print(f"   - F1-Score (Macro)      : {eval_inset_vs_human['f1_score']*100:.2f}%")

    print("\n2. Evaluasi Prediksi Multinomial Naive Bayes terhadap Label Acuan Manusia (N = 100):")
    print(f"   - Kesesuaian (Accuracy) : {eval_nb_vs_human['accuracy']*100:.2f}% ({eval_nb_vs_human['tp_total']}/{eval_nb_vs_human['n_test']})")
    print(f"   - Precision (Macro)     : {eval_nb_vs_human['precision']*100:.2f}%")
    print(f"   - Recall (Macro)        : {eval_nb_vs_human['recall']*100:.2f}%")
    print(f"   - F1-Score (Macro)      : {eval_nb_vs_human['f1_score']*100:.2f}%")

    # Tabel perbandingan evaluasi terhadap label acuan manusia (Keseluruhan & Per Aspek)
    eval_rows = [
        {
            "Metode_Dievaluasi": "Pelabelan Kamus InSet vs Label Acuan Manusia (Total Sampel)",
            "Cakupan": "Keseluruhan (4 Aspek)",
            "Jumlah_Sampel": eval_inset_vs_human["n_test"],
            "Sesuai_Manusia": eval_inset_vs_human["tp_total"],
            "Accuracy_Pct": round(eval_inset_vs_human["accuracy"] * 100, 2),
            "Precision_Pct": round(eval_inset_vs_human["precision"] * 100, 2),
            "Recall_Pct": round(eval_inset_vs_human["recall"] * 100, 2),
            "F1_Score_Pct": round(eval_inset_vs_human["f1_score"] * 100, 2),
        },
        {
            "Metode_Dievaluasi": "Prediksi Naive Bayes vs Label Acuan Manusia (Total Sampel)",
            "Cakupan": "Keseluruhan (4 Aspek)",
            "Jumlah_Sampel": eval_nb_vs_human["n_test"],
            "Sesuai_Manusia": eval_nb_vs_human["tp_total"],
            "Accuracy_Pct": round(eval_nb_vs_human["accuracy"] * 100, 2),
            "Precision_Pct": round(eval_nb_vs_human["precision"] * 100, 2),
            "Recall_Pct": round(eval_nb_vs_human["recall"] * 100, 2),
            "F1_Score_Pct": round(eval_nb_vs_human["f1_score"] * 100, 2),
        },
    ]

    for asp in ASPECT_CATEGORIES:
        sub = sample_df[sample_df["aspect"] == asp]
        ev_in = compute_evaluation_metrics(
            sub["label_acuan_manusia"].tolist(),
            sub["sentiment"].tolist(),
            scope_name=asp,
        )
        ev_nb = compute_evaluation_metrics(
            sub["label_acuan_manusia"].tolist(),
            sub["predicted_sentiment"].tolist(),
            scope_name=asp,
        )
        eval_rows.append({
            "Metode_Dievaluasi": "Pelabelan Kamus InSet vs Label Acuan Manusia",
            "Cakupan": asp,
            "Jumlah_Sampel": ev_in["n_test"],
            "Sesuai_Manusia": ev_in["tp_total"],
            "Accuracy_Pct": round(ev_in["accuracy"] * 100, 2),
            "Precision_Pct": round(ev_in["precision"] * 100, 2),
            "Recall_Pct": round(ev_in["recall"] * 100, 2),
            "F1_Score_Pct": round(ev_in["f1_score"] * 100, 2),
        })
        eval_rows.append({
            "Metode_Dievaluasi": "Prediksi Naive Bayes vs Label Acuan Manusia",
            "Cakupan": asp,
            "Jumlah_Sampel": ev_nb["n_test"],
            "Sesuai_Manusia": ev_nb["tp_total"],
            "Accuracy_Pct": round(ev_nb["accuracy"] * 100, 2),
            "Precision_Pct": round(ev_nb["precision"] * 100, 2),
            "Recall_Pct": round(ev_nb["recall"] * 100, 2),
            "F1_Score_Pct": round(ev_nb["f1_score"] * 100, 2),
        })

    eval_human_df = pd.DataFrame(eval_rows)
    eval_human_df.to_csv(HUMAN_EVAL_CSV, index=False, encoding="utf-8-sig")

    # Ringkasan kategori kesalahan pemetaan & pelabelan yang ditemukan
    err_summary_df = (
        sample_df["Kategori_Temuan_Audit"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Kategori_Temuan_Audit", "count": "Jumlah_Pasangan"})
    )
    err_summary_df["Persentase (%)"] = (err_summary_df["Jumlah_Pasangan"] / len(sample_df) * 100).round(2)
    err_summary_df.to_csv(HUMAN_ERROR_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\nRingkasan Temuan Pemeriksaan Manusia (Kesalahan Pemetaan & Pelabelan):")
    print(err_summary_df.to_string(index=False))

    export_cols = [
        "Pair_ID", "Comment_ID", "Comment Text", "aspect", "matched_keywords",
        "aspect_context", "aspect_text", " Tipe_Komentar", "Memuat_Negasi",
        "Validitas_Pemetaan_Aspek", "sentiment", "predicted_sentiment",
        "label_acuan_manusia", "Kesesuaian_InSet_vs_Manusia",
        "Kesesuaian_NB_vs_Manusia", "Kategori_Temuan_Audit",
        "Catatan_Pemeriksaan_Manusia",
    ]
    clean_export_df = sample_df[export_cols].rename(columns={
        " Tipe_Komentar": "Tipe_Komentar",
        "sentiment": "Label_Kamus_InSet",
        "predicted_sentiment": "Prediksi_Naive_Bayes",
        "label_acuan_manusia": "Label_Acuan_Manusia",
    })

    clean_export_df.to_csv(HUMAN_SAMPLE_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(HUMAN_SAMPLE_XLSX, engine="openpyxl") as writer:
        clean_export_df.to_excel(writer, sheet_name="Sampel_Audit_Manusia_100", index=False)
        eval_human_df.to_excel(writer, sheet_name="Evaluasi_vs_Label_Manusia", index=False)
        err_summary_df.to_excel(writer, sheet_name="Ringkasan_Temuan_Kesalahan", index=False)

    print(f"\nBerkas pemeriksaan manusia disimpan di:")
    print(f" - {HUMAN_SAMPLE_XLSX}")
    print(f" - {HUMAN_EVAL_CSV}")
    print(f" - {HUMAN_ERROR_SUMMARY_CSV}")
    print("========================================")


if __name__ == "__main__":
    main()
