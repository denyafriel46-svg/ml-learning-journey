import pandas as pd

print("PROJECT 4: SORTING & GROUPING DATA")
print("="*60)

# Step 1: Buat data
data = {
    'Nama': ['Sherin', 'Deny', 'Anggi', 'Wawan', 'Evan', 'Gunawan', 'Farah', 'Hendra'],
    'Umur': [17, 39, 32, 22, 20, 23, 25, 28],
    'Departemen': ['IT', 'HR', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR'],
    'Gaji': [5000000, 4500000, 6000000, 5500000, 4800000, 5200000, 5300000, 4700000]
}

df = pd.DataFrame(data)
print("Data asli:")
print(df)

print("\n" + "="*60)

# SORTING 1: Sort berdasarkan Umur (ascending = dari kecil ke besar)
print("\nSORTING 1: Sort by Umur (Ascending)")
sort1 = df.sort_values('Umur')
print(sort1)

print("\n" + "="*60)

# SORTING 2: Sort berdasarkan Gaji (descending = dari besar ke kecil)
print("\nSORTING 2: Sort by Gaji (Descending)")
sort2 = df.sort_values('Gaji', ascending=False)
print(sort2)

print("\n" + "="*60)

# SORTING 3: Sort berdasarkan Umur, terus Gaji
print("\nSORTING 3: Sort by Umur, then by Gaji")
sort3 = df.sort_values(['Umur', 'Gaji'], ascending=[True, False])
print(sort3)

print("\n" + "="*60)

# GROUPING 1: Group by Departemen, hitung berapa banyak per departemen
print("\nGROUPING 1: Count per Departemen")
group1 = df.groupby('Departemen').size()
print(group1)

print("\n" + "="*60)

# GROUPING 2: Group by Departemen, hitung rata-rata Gaji per departemen
print("\nGROUPING 2: Average Gaji per Departemen")
group2 = df.groupby('Departemen')['Gaji'].mean()
print(group2)

print("\n" + "="*60)

# GROUPING 3: Group by Departemen, hitung multiple aggregate
print("\nGROUPING 3: Multiple Aggregate per Departemen")
group3 = df.groupby('Departemen').agg({
    'Gaji': ['mean', 'sum', 'min', 'max'],
    'Umur': 'mean'
})
print(group3)

print("\n" + "="*60)

# GROUPING 4: Group by Departemen, hitung count dan avg Gaji
print("\nGROUPING 4: Count & Avg Gaji per Departemen (nicer format)")
group4 = df.groupby('Departemen').agg({
    'Nama': 'count',
    'Gaji': 'mean'
}).round(0)
group4.columns = ['Jumlah_Orang', 'Avg_Gaji']
print(group4)

print("\n" + "="*60)

# COMBINING: Sort + Group
print("\nCOMBINING: Sort Departemen by Avg Gaji (Descending)")
combine = df.groupby('Departemen')['Gaji'].mean().sort_values(ascending=False)
print(combine)

print("\n" + "="*60)

# ADVANCED: Group by Departemen, ambil nama & gaji, sort by gaji
print("\nADVANCED: Nama & Gaji per Departemen (sorted by gaji)")
for dept in df['Departemen'].unique():
    print(f"\n{dept}:")
    dept_data = df[df['Departemen'] == dept].sort_values('Gaji', ascending=False)[['Nama', 'Gaji']]
    print(dept_data.to_string(index=False))

print("\n" + "="*60)
print("\nEXPERIMEN:")
print("1. Sort by kolom yang berbeda")
print("2. Group by Departemen dengan aggregate yang berbeda")
print("3. Kombinasikan sort + group")
print("="*60)