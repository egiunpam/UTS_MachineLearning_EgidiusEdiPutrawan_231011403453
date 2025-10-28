# process_titanic.py
# Tahap preprocessing untuk dataset Titanic

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------
# 1. Baca dataset
# ----------------------------
df = pd.read_csv("titanic.csv")

print("===== DATA AWAL =====")
print(df.head())

# ----------------------------
# 2. Hapus kolom yang tidak relevan
# ----------------------------
# Kolom seperti Lname, Name, Ticket, Cabin kurang berguna untuk model
df = df.drop(columns=['Lname', 'Name', 'Ticket', 'Cabin'])

# ----------------------------
# 3. Tangani nilai kosong (missing values)
# ----------------------------
# Isi umur (Age) dengan nilai median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Isi Embarked (pelabuhan keberangkatan) dengan modus (nilai paling sering)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# ----------------------------
# 4. Ubah data kategorikal menjadi numerik
# ----------------------------
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])        # male=1, female=0
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  # S, C, Q → angka

# ----------------------------
# 5. Pisahkan fitur (X) dan target (y)
# ----------------------------
X = df.drop(columns=['Survived'])
y = df['Survived']

# ----------------------------
# 6. Standarisasi data numerik
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 7. Split data menjadi train & test
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------
# 8. Tampilkan hasil akhir
# ----------------------------
print("\n===== DATASET HASIL PROCESSING =====")
print("Jumlah data training :", X_train.shape[0])
print("Jumlah data testing  :", X_test.shape[0])
print("\nContoh data setelah scaling:")
print(pd.DataFrame(X_scaled, columns=X.columns).head())

# ----------------------------
# 9. Simpan hasil processing ke file baru (opsional)
# ----------------------------
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['Survived'] = y.values
processed_df.to_csv("titanic_processed.csv", index=False)

print("\nFile hasil processing tersimpan sebagai: titanic_processed.csv ✅")
