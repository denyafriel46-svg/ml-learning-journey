import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("PROJECT 5: DATA VISUALIZATION")
print("="*60)

# Step 1: Buat data
np.random.seed(42)
data = {
    'siswa': ['deny', 'sherin', 'anggi', 'wawan', 'ridwan', 'hendrik', 'arif'],
    'nilai_mtk': [70, 100, 93, 43, 67, 88, 89],
    'nilai_penjas': [98, 67, 55, 88, 87, 88, 55]
}

df = pd.DataFrame(data)
print("Data:")
print(df)

print("\n" + "="*60)

# Buat figure dengan 6 subplot
plt.figure(figsize=(15, 10))

# ===== PLOT 1: LINE CHART =====
plt.subplot(2, 3, 1)
plt.plot(df['siswa'], df['nilai_mtk'], marker='s', color='red', linewidth=4)
plt.title('nilai matematika siswa ', fontsize=12, fontweight='bold')
plt.xlabel('siswa')
plt.ylabel('nilai matematika')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.10)

# ===== PLOT 2: BAR CHART =====
plt.subplot(2, 3, 2)
plt.bar(df['siswa'], df['nilai_penjas'], color='black', alpha=0.7)
plt.title('nilai penjas siswa', fontsize=12, fontweight='bold')
plt.xlabel('siswa')
plt.ylabel('Nilai')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# ===== PLOT 3: MULTIPLE LINE CHART =====
plt.subplot(2, 3, 3)
plt.plot(df['siswa'], df['nilai_mtk'], marker='o', label='nilai_mtk', color='green')
plt.plot(df['siswa'], df['nilai_penjas'], marker='s', label='nilai penjas(÷5)', color='purple')
plt.title('nilai_mtk vs nilai_penjas', fontsize=12, fontweight='bold')
plt.xlabel('siswa')
plt.ylabel('nilai')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# ===== PLOT 4: HISTOGRAM =====
plt.subplot(2, 3, 4)
random_data = np.random.normal(100, 20, 1000)
plt.hist(random_data, bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.title('Distribusi Data Random', fontsize=12, fontweight='bold')
plt.xlabel('Nilai')
plt.ylabel('Frekuensi')
plt.grid(True, alpha=0.3, axis='y')

# ===== PLOT 5: SCATTER PLOT =====
plt.subplot(2, 3, 5)
x_scatter = np.random.rand(100) * 100
y_scatter = x_scatter + np.random.normal(0, 10, 100)
plt.scatter(x_scatter, y_scatter, alpha=0.6, color='orange', s=50)
plt.title('Scatter Plot (X vs Y)', fontsize=12, fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)

# ===== PLOT 6: PIE CHART =====
# ===== PLOT 6: PIE CHART =====
plt.subplot(2, 3, 6)
plt.pie(df['nilai_mtk'], labels=df['siswa'], autopct='%1.1f%%', startangle=90)
plt.title('Proporsi Nilai MTK per Siswa', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_output.png', dpi=100)
print("\n✓ Visualization saved as 'visualization_output.png'")
print("\nUntuk lihat plot, uncomment line di bawah:")
print("# plt.show()")
plt.show()

# print("\n" + "="*60)
# print("\nEXPERIMEN:")
# print("1. Ubah warna plot (color='red', 'green', dll)")
# print("2. Ubah jenis marker (marker='o', 's', '^', dll)")
# print("3. Tambah title & label yang berbeda")
# print("4. Coba plot dengan data yang berbeda")
# print("="*60)