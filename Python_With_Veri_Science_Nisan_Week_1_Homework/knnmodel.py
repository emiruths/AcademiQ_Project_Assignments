# Gerekli kütüphaneler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Veri kümesini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Eğitim ve test verisi olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN modelini oluştur ve eğit (k = 3 olarak aldık)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Tahmin yap
y_pred = knn.predict(X_test)

# Sonuçlar
print("KNN Sınıflandırma Modeli (k=3)")
print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("\n Sınıflandırma Raporu:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
