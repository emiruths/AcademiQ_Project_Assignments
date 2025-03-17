import numpy as np
import matplotlib.pyplot as plt

# BÖLÜM 1: Şirket Maaş Analizi
# Görev 1: Maaşların Simülasyonu
# 500 çalışanın maaşlarını rastgele oluşturur
maaslar = np.random.randint(3000, 15001, 500)

# Ortalama maksimum ve minimum maaşları hesaplar
ortalama_maas = np.mean(maaslar)
max_maas = np.max(maaslar)
min_maas = np.min(maaslar)

# Hesaplanan değerleri konsola yazdırır
print(f"Ortalama Maaş: {ortalama_maas:.2f} TL")
print(f"Maksimum Maaş: {max_maas} TL")
print(f"Minimum Maaş: {min_maas} TL")

# Maaş dağılımını hrafik ile görselleştirir
plt.hist(maaslar, bins=20, color='blue', edgecolor='black')
plt.title('Maaş Dağılımı')
plt.xlabel('Maaş (TL)')
plt.ylabel('Çalışan Sayısı')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Görev 2: Çalışanların Departmanlara Dağılımı
# Çalışanları rastgele 3 departmana atar
departmanlar = np.random.choice([1, 2, 3], 500)

# Her departmandaki ortalama maaşı hesaplar
ortalama_maas_muh = np.mean(maaslar[departmanlar == 1])
ortalama_maas_muhasebe = np.mean(maaslar[departmanlar == 2])
ortalama_maas_pazarlama = np.mean(maaslar[departmanlar == 3])

# Sonuçları ekrana yazdırır
print(f"Mühendislik Ortalama Maaş: {ortalama_maas_muh:.2f} TL")
print(f"Muhasebe Ortalama Maaş: {ortalama_maas_muhasebe:.2f} TL")
print(f"Pazarlama Ortalama Maaş: {ortalama_maas_pazarlama:.2f} TL")