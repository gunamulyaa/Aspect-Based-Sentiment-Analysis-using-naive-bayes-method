import glob
import os
import pandas as pd

# Lokasi direktori tempat script Python ini berada
folder = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(folder, ".."))
results_dir = os.path.join(project_root, "Hasil_Merge")
os.makedirs(results_dir, exist_ok=True)

# Cari semua file Excel di folder Dataset
files = glob.glob(os.path.join(folder, "*.xlsx"))

# Jangan ikut membaca file hasil gabungan jika script dijalankan ulang
output_file = os.path.join(results_dir, "combined_dataset.xlsx")
files = [file for file in files if os.path.basename(file) != os.path.basename(output_file)]

print(f"Jumlah dataset ditemukan: {len(files)}")

all_data = []

for file in files:
    print(f"Membaca: {os.path.basename(file)}")

    # Struktur original: header pada baris ke-15
    df = pd.read_excel(file, header=14)

    # Hapus baris yang benar-benar kosong
    df = df.dropna(how="all")

    # Tambahkan informasi asal file (opsional)
    df["Source_File"] = os.path.basename(file)
    all_data.append(df)

# Gabungkan semua dataset berdasarkan kolom
combined_df = pd.concat(all_data, ignore_index=True)

# Simpan hasil
combined_df.to_excel(output_file, index=False)

print("\n===================================")
print("Dataset berhasil digabungkan!")
print(f"Total baris : {len(combined_df)}")
print(f"Total kolom: {len(combined_df.columns)}")
print(f"Output     : {output_file}")
print("===================================")