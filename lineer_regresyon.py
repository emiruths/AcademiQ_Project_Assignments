import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Veri kümesini yükler
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# DataFrame oluşturur
df = pd.DataFrame(X, columns=diabetes.feature_names)
df["target"] = y

### Basit Lineer Regresyon
X_bmi = df[["bmi"]]
y = df["target"]

X_train_bmi, X_test_bmi, y_train_bmi, y_test_bmi = train_test_split(X_bmi, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train_bmi, y_train_bmi)
y_pred_simple = model_simple.predict(X_test_bmi)

# Basit model metrikleri
r2_simple = r2_score(y_test_bmi, y_pred_simple)
mae_simple = mean_absolute_error(y_test_bmi, y_pred_simple)
mse_simple = mean_squared_error(y_test_bmi, y_pred_simple)

print("Basit Lineer Regresyon (BMI):")
print("R² Skoru:", r2_simple)
print("MAE:", mae_simple)
print("MSE:", mse_simple)
print()

### Çoklu Lineer Regresyon 
X_all = df.drop(columns="target")
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_all, y_train_all)
y_pred_multi = model_multi.predict(X_test_all)

# Çoklu model metrikleri
r2_multi = r2_score(y_test_all, y_pred_multi)
mae_multi = mean_absolute_error(y_test_all, y_pred_multi)
mse_multi = mean_squared_error(y_test_all, y_pred_multi)

print("Çoklu Lineer Regresyon (Tüm Özellikler):")
print("R² Skoru:", r2_multi)
print("MAE:", mae_multi)
print("MSE:", mse_multi)
print()

### Yorum
if r2_multi > r2_simple:
    print("Çoklu model daha başarılı çünkü daha yüksek R² skoruna sahip.")
else:
    print("Basit model daha iyi çıktı, bu durum overfitting/underfitting kaynaklı olabilir.")

print("R² skoru, modelin verinin varyansını ne kadar açıkladığını gösterir. 1'e yaklaştıkça model daha başarılıdır.")
