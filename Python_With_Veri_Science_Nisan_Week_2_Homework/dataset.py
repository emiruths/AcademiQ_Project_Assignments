import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 1.Veri Setini Yükleyin ve İnceleyin
# Veri Setini Yükle
df = pd.read_csv("train.csv")  # Dosya yolu senin bilgisayarına göre ayarlanmalı

# Veri Yapısı ve Eksik Veri Kontrolü
print("Veri Boyutu:", df.shape)
print("Eksik Veriler:\n", df.isnull().sum())

# 2.Veriyi Temizleyin ve Hazırlayın
# Gereksiz Sütunları Kaldır
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Eksik Verileri Doldur
df['Age'] = SimpleImputer(strategy='mean').fit_transform(df[['Age']])
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Kategorik Verileri Sayısala Çevir
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])         # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])  # S, C, Q → 0,1,2 olabilir

# Giriş ve Çıkış Değişkenlerini Ayır
X = df.drop('Survived', axis=1)
y = df['Survived']

# Eğitim ve Test Verisi Ayırma (%80 - %20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3.Model Seçin ve Eğitin
# Model Seçimi ve Eğitimi (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4.Modeli Değerlendirin
# Tahmin ve Değerlendirme
y_pred = model.predict(X_test)

print(" Doğruluk Oranı: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\n Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print(" Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

# 5.Sonuçları Yorumlayın
# Özellik Önemleri
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n Özellik Önem Sıralaması:\n", feature_importance)

