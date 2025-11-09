import pandas as pd

print("PROJECT 2: DATA SISWA")
print("="*60)

# Step 1: Buat data sebagai dictionary
data = {
    'Nama': ['Sherin', 'Deny', 'Anggi', 'Wawan', 'Evan', 'Gunawan'],
    'Umur': [17, 39, 32, 22, 20, 23],
    'Nilai_MTK': [22, 33, 12, 31, 98, 23],
    'Nilai_IPA': [29, 10, 32, 21, 85, 49]
}

# Step 2: Convert dictionary jadi DataFrame
df = pd.DataFrame(data)
print("Tabel Data Siswa:")
print(df)

print("\n" + "="*60)

# Step 3: Lihat informasi basic
print(f"\nJumlah baris: {len(df)}")
print(f"Jumlah kolom: {len(df.columns)}")
print(f"Nama kolom: {list(df.columns)}")

print("\n" + "="*60)

# Step 4: Akses satu kolom
print("\nKolom Nama:")
print(df['Nama'])

print("\nKolom Nilai_MTK:")
print(df['Nilai_MTK'])

print("\nKolom Nilai_IPA:")
print(df['Nilai_IPA'])

print("\n" + "="*60)

# Step 5: Akses satu baris
print("\nData siswa pertama (Sherin):")
print(df.iloc[0])

print("\nData siswa kedua (Deny):")
print(df.iloc[1])

print("\nData siswa ketiga (Anggi):")
print(df.iloc[2])

print("\n" + "="*60)

# Step 6: Hitung statistik
print(f"\nRata-rata Umur: {df['Umur'].mean():.2f}")
print(f"Rata-rata Nilai MTK: {df['Nilai_MTK'].mean():.2f}")
print(f"Rata-rata Nilai IPA: {df['Nilai_IPA'].mean():.2f}")

print(f"\nUmur tertinggi: {df['Umur'].max()}")
print(f"Umur terendah: {df['Umur'].min()}")

print("\n" + "="*60)