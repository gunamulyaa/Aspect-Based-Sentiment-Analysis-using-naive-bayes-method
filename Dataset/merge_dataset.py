import pandas as pd
import glob
import os

# Folder tempat dataset Excel berada
folder = os.path.dirname(os.path.abspath(__file__))

# Cari semua file Excel
files = glob.glob(os.path.join(folder, "*.xlsx"))

# Jangan ikut membaca file hasil gabungan jika script dijalankan ulang
output_file = os.path.join(folder, "combined_dataset.xlsx")
files = [file for file in files if os.path.basename(file) != output_file]

print(f"Jumlah dataset ditemukan: {len(files)}")

all_data = []

for file in files:
    print(f"Membaca: {os.path.basename(file)}")

    # Header dataset berada di baris ke-15
    # Excel row 15 = pandas header=14
    df = pd.read_excel(
        file,
        header=14
    )

    # Hapus baris yang benar-benar kosong
    df = df.dropna(how="all")

    # Tambahkan informasi asal file (opsional)
    # Kalau benar-benar hanya ingin isi dataset,
    # bagian ini bisa dihapus.
    df["Source_File"] = os.path.basename(file)

    all_data.append(df)

# Gabungkan semua dataset berdasarkan kolom
combined_df = pd.concat(
    all_data,
    ignore_index=True
)

# Simpan hasil
combined_df.to_excel(
    output_file,
    index=False
)

print("\n===================================")
print("Dataset berhasil digabungkan!")
print(f"Total baris : {len(combined_df)}")
print(f"Total kolom: {len(combined_df.columns)}")
print(f"Output     : {output_file}")
print("===================================")