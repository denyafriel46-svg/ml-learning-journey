import numpy as np

# ===== PROJECT 1: HELLO NUMPY =====
# Tujuan: Belajar array dan basic operations

print("PROJECT 1: NILAI SISWA")
print("="*50)

# Step 1: Buat array nilai siswa
nilai = np.array([70, 90, 75, 95, 85])
print(f"Nilai siswa: {nilai}")

# Step 2: Hitung rata-rata
rata_rata = np.mean(nilai)
print(f"Rata-rata: {rata_rata}")

# Step 3: Hitung nilai 
tertinggi = np.max(nilai)
print(f"Tertinggi: {tertinggi}")

# Step 4: Hitung nilai terendah
terendah = np.min(nilai)
print(f"Terendah: {terendah}")

# Step 5: Berapa banyak siswa?
jumlah_siswa = len(nilai)
print(f"Jumlah siswa: {jumlah_siswa}")

print("="*50)
print("\nEXPERIMEN:")
print("Coba ubah array 'nilai' dengan angka lain")
print("Jalankan ulang dan lihat hasilnya berubah")~