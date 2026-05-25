# FLO Müşteri Segmentasyonu

FLO'nun OmniChannel müşteri verisini kullanarak K-Means ve Hierarchical Clustering ile müşteri segmentasyonu yaptım. Amaç, farklı davranış profillerine sahip müşteri gruplarını ortaya çıkarmak ve her segment için pazarlama stratejisi önerisinde bulunmak.

## Veri Seti

`flo_data_20k.csv` — 19.945 müşteri, 12 değişken. 2020-2021 yıllarında hem online hem offline alışveriş yapmış (OmniChannel) müşterilerin geçmiş alışveriş davranışları.

## Yaklaşım

Ham veride recency, frequency, monetary, tenure ve omni_ratio değişkenlerini türettim. Bunları StandardScaler ile ölçeklendirdikten sonra iki farklı yöntemle kümeleme yaptım.

**K-Means:** Elbow ve Silhouette grafiklerine bakarak k=6 seçtim. Her kümenin centroid değerlerini inceleyerek segmentleri isimlendirdim.

**Hierarchical Clustering:** Ward yöntemi ile dendrogram çizdim (20k satır çok ağır olduğu için 2000 örneklemle), orada da 6 küme belirginleşti.

Sonuçta K-Means daha iyi Silhouette skoru verdi (0.278 vs 0.219), bu yüzden asıl segmentasyon olarak onu kullandım.

## Segmentler

| Segment | Müşteri Sayısı | Ortalama Harcama |
|---|---|---|
| Şampiyonlar | 9 | ~25.000 ₺ |
| Sadık Müşteriler | 762 | ~3.000 ₺ |
| Potansiyel Sadık | 5.309 | ~886 ₺ |
| Risk Altındaki | 6.987 | ~534 ₺ |
| Uyku Modundaki | 1.967 | ~807 ₺ |
| Yeni Müşteriler | 4.911 | ~490 ₺ |

## Kurulum

```bash
pip install pandas numpy matplotlib scikit-learn scipy
python flo_segmentation.py
```

Çalıştırmadan önce `outputs/` klasörünü oluşturun:

```bash
mkdir outputs
```

## Çıktılar

```
outputs/
├── 01_kmeans_optimal_k.png       # Elbow + Silhouette grafikleri
├── 02_kmeans_radar.png           # K-Means segment profil radari
├── 03_kmeans_bars.png            # Segment büyüklük ve harcama karşılaştırması
├── 04_dendrogram.png             # Ward dendrogram
├── 05_hc_radar.png               # HC segment profil radari
├── flo_customer_segments.csv     # Müşteri bazlı segment etiketleri
└── kmeans_segment_summary.csv    # Segment bazlı istatistikler
```

## Notlar

- Şampiyonlar segmenti sadece 9 müşteri — aykırı değer gibi duruyor ama harcama profili gerçekten bu kadar farklı, ayrı tutmak mantıklı
- Risk Altındaki segment en kalabalık grup (7k+), buradaki müşterileri geri kazanmak için hedefli kampanya düşünülebilir
- Omni ratio değişkeni online/offline dengesini yakalamak için eklendi, segmentlerde belirgin fark var
