import pandas as pd

print("PROJECT 3: FILTERING DATA")
print("="*60)

# Step 1: Buat data
data = {
    'Nama': ['Sherin', 'Deny', 'Anggi', 'Wawan', 'Evan', 'Gunawan'],
    'Umur': [17, 39, 32, 22, 20, 23],
    'Nilai_MTK': [85, 90, 78, 92, 88, 75],
    'Nilai_IPA': [80, 88, 82, 95, 85, 79]
}

df = pd.DataFrame(data)
print("Data asli:")
print(df)

print("\n" + "="*60)

# FILTERING 1: Ambil siswa dengan umur > 25
print("\nFiltering 1: Siswa dengan Umur < 25")
filter1 = df[df['Umur'] > 25]
print(filter1)

print("\n" + "="*60)

# FILTERING 2: Ambil siswa dengan Nilai_MTK >= 85
print("\nFiltering 2: Siswa dengan Nilai MTK >= 85")
filter2 = df[df['Nilai_MTK'] >= 85]
print(filter2)

print("\n" + "="*60)

# FILTERING 3: Ambil siswa dengan Nilai_IPA < 85
print("\nFiltering 3: Siswa dengan Nilai IPA > 85")
filter3 = df[df['Nilai_IPA'] < 85]
print(filter3)

print("\n" + "="*60)

# FILTERING 4: Ambil siswa bernama 'Evan'
print("\nFiltering 4: Siswa bernama 'Evan'")
filter4 = df[df['Nama'] == 'Evan']
print(filter4)

print("\n" + "="*60)

# FILTERING 5: Multiple condition - Umur > 20 DAN Nilai_MTK >= 85
print("\nFiltering 5: Umur > 20 DAN Nilai_MTK >= 85")
filter5 = df[(df['Umur'] > 20) & (df['Nilai_MTK'] >= 85)]
print(filter5)

print("\n" + "="*60)

# FILTERING 6: Multiple condition - Nilai_MTK > 80 ATAU Nilai_IPA > 85
print("\nFiltering 6: Nilai_MTK > 80 ATAU Nilai_IPA > 85")
filter6 = df[(df['Nilai_MTK'] > 80) | (df['Nilai_IPA'] > 85)]
print(filter6)

print("\n" + "="*60)

# FILTERING 7: Ambil kolom tertentu saja
print("\nFiltering 7: Ambil hanya kolom Nama dan Nilai_MTK")
filter7 = df[['Nama', 'Nilai_MTK']]
print(filter7)

print("\n" + "="*60)

# FILTERING 8: Kombinasi - Ambil nama dan umur dari siswa dengan Nilai_MTK >= 85
print("\nFiltering 8: Nama & Umur siswa dengan Nilai_MTK >= 85")
filter8 = df[df['Nilai_MTK'] >= 85][['Nama', 'Umur']]
print(filter8)

#filtering 9: ambill niulai siswa yang nil;ai matematikanya <80
print("\nfiltering 9: nama & umur siswa dengan nilai mtk <= 80 ")
filter9 = df[df["Nilai_MTK"] <= 80][['Nama','Umur','Nilai_MTK']]
print(filter9)


print("\n" + "="*60)
print("\nEXPERIMEN:")
print("1. Ubah kondisi di filtering (misal > jadi <)")
print("2. Tambah kondisi baru")
print("3. Lihat hasilnya berubah")
print("="*60)