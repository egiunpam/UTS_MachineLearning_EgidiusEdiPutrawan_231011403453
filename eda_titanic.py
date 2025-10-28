# eda_titanic.py
# Exploratory Data Analysis (EDA) untuk dataset Titanic

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. Baca dataset
# ----------------------------
df = pd.read_csv("titanic.csv")

# ----------------------------
# 2. Tampilkan informasi dasar
# ----------------------------
print("\n===== INFORMASI DATASET =====")
print(df.info())

print("\n===== 5 DATA PERTAMA =====")
print(df.head())

print("\n===== DESKRIPSI STATISTIK =====")
print(df.describe(include='all'))

# ----------------------------
# 3. Cek data kosong
# ----------------------------
print("\n===== JUMLAH DATA KOSONG =====")
print(df.isnull().sum())

# ----------------------------
# 4. Distribusi penumpang berdasarkan jenis kelamin
# ----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df, palette='pastel')
plt.title("Distribusi Penumpang berdasarkan Jenis Kelamin")
plt.xlabel("Jenis Kelamin")
plt.ylabel("Jumlah Penumpang")
plt.show()

# ----------------------------
# 5. Tingkat kelangsungan hidup berdasarkan jenis kelamin
# ----------------------------
plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df, palette='coolwarm')
plt.title("Rata-rata Kelangsungan Hidup berdasarkan Jenis Kelamin")
plt.xlabel("Jenis Kelamin")
plt.ylabel("Persentase Kelangsungan Hidup")
plt.show()

# ----------------------------
# 6. Distribusi umur penumpang
# ----------------------------
plt.figure(figsize=(8,4))
sns.histplot(df['Age'].dropna(), bins=20, kde=True, color='skyblue')
plt.title("Distribusi Umur Penumpang")
plt.xlabel("Umur")
plt.ylabel("Jumlah Penumpang")
plt.show()

# ----------------------------
# 7. Hubungan antara Kelas dan Kelangsungan Hidup
# ----------------------------
plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df, palette='mako')
plt.title("Kelangsungan Hidup berdasarkan Kelas Penumpang")
plt.xlabel("Kelas Penumpang (1 = Atas, 3 = Bawah)")
plt.ylabel("Persentase Kelangsungan Hidup")
plt.show()

# ----------------------------
# 8. Korelasi antar variabel numerik
# ----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Heatmap Korelasi Variabel Numerik")
plt.show()
