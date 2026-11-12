# -*- coding: utf-8 -*-
import glob
import os
import sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Lokasi direktori tempat script Python ini berada
folder = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(folder, ".."))
results_dir = os.path.join(project_root, "Hasil_Merge")
os.makedirs(results_dir, exist_ok=True)

# Cari semua file Excel di folder Dataset (urutkan agar deterministik)
files = sorted(glob.glob(os.path.join(folder, "*.xlsx")))

# Jangan ikut membaca file hasil gabungan jika script dijalankan ulang
output_file = os.path.join(results_dir, "combined_dataset.xlsx")
files = [file for file in files if os.path.basename(file) != os.path.basename(output_file)]

print("========================================")
print("TAHAP 1: PENGGABUNGAN DATASET (MERGE)")
print("========================================")
print(f"Jumlah berkas dataset ditemukan: {len(files)}")

all_data = []
per_file_stats = []

for file in files:
    fname = os.path.basename(file)
    # Struktur original: header pada baris ke-15
    df = pd.read_excel(file, header=14)
    raw_rows = len(df)

    # Hapus baris yang benar-benar kosong di seluruh kolom
    df = df.dropna(how="all").copy()
    non_empty_rows = len(df)

    # Tambahkan informasi asal file
    df["Source_File"] = fname
    all_data.append(df)
    per_file_stats.append({
        "Source_File": fname,
        "Raw_Rows": raw_rows,
        "Valid_Rows": non_empty_rows,
    })
    print(f" - {fname}: {non_empty_rows} baris")

# Gabungkan semua dataset berdasarkan kolom
combined_df = pd.concat(all_data, ignore_index=True)

# Simpan hasil gabungan dan ringkasan per berkas
combined_df.to_excel(output_file, index=False)
pd.DataFrame(per_file_stats).to_csv(
    os.path.join(results_dir, "merge_file_summary.csv"),
    index=False,
    encoding="utf-8-sig",
)

missing_comment = int(combined_df["Comment Text"].isna().sum()) if "Comment Text" in combined_df.columns else 0

print("\n========================================")
print("PENGGABUNGAN DATASET SELESAI")
print("========================================")
print(f"Total baris gabungan           : {len(combined_df)}")
print(f"Baris tanpa teks komentar      : {missing_comment}")
print(f"Baris berisi teks komentar     : {len(combined_df) - missing_comment}")
print(f"Total kolom                    : {len(combined_df.columns)}")
print(f"Output                         : {output_file}")
print("========================================")
