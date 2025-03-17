import numpy as np
import matplotlib.pyplot as plt

# BÖLÜM 3: Ürün Satış Analizi
# Görev 4: Satış Verileri
# 5 farklı ürün ve 30 günlük rastgele satış miktarları
urunler = ["Telefon", "Bilgisayar", "Kulaklık", "Saat", "Tablet"]
satislar = np.random.randint(10, 101, (5, 30))

# her ürün için toplam ve ortalama satış miktarlarını hesaplar
urun_toplamlari = np.sum(satislar, axis=1)
urun_ortalama = np.mean(satislar, axis=1)

# Sonuçları ekrana yazdırır
for i, urun in enumerate(urunler):
    print(f"{urun}: Toplam Satış = {urun_toplamlari[i]}, Ortalama Günlük Satış = {urun_ortalama[i]:.2f}")

# Ürün bazında çubuk grafiği çizer
plt.bar(urunler, urun_toplamlari, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.title('Ürün Satış Dağılımı')
plt.xlabel('Ürünler')
plt.ylabel('Toplam Satış')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

