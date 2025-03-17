import numpy as np
import matplotlib.pyplot as plt

# BÖLÜM 2: Hava Durumu Verileri
# Görev 3: Günlük Sıcaklık Değerleri
# 365 günlük sıcaklık değerlerini rastgele üretşir
temperatures = np.random.uniform(-10, 40, 365)

# Ortalama en yüksek ve en düşük sıcaklıkları hesapla
avg_temp = np.mean(temperatures)
max_temp = np.max(temperatures)
min_temp = np.min(temperatures)

# Sonuçları ekrana yazdırır
print(f"Ortalama Sıcaklık: {avg_temp:.2f}°C")
print(f"En Yüksek Sıcaklık: {max_temp:.2f}°C")
print(f"En Düşük Sıcaklık: {min_temp:.2f}°C")

# Sıcaklık değişimini çizgi grafiği ile gösterir
plt.plot(temperatures, color='red', linewidth=1)
plt.title('Yıllık Sıcaklık Değişimi')
plt.xlabel('Gün')
plt.ylabel('Sıcaklık (°C)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()