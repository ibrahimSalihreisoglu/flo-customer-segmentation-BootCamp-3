# Önerilen Commit Mesajları

Aşağıdaki sırayla commit atabilirsin. Her biri mantıklı bir çalışma adımını temsil ediyor.

---

### 1. İlk commit — proje iskelet
```
initial commit
```

---

### 2. Veri okuma ve keşif
```
add data loading and initial exploration

veri tiplerini düzeltip eksik değerlere baktım,
shape ve describe çıktıları da eklendi
```

---

### 3. Feature engineering
```
add rfm-based feature engineering

recency, frequency, monetary, tenure ve omni_ratio
türetildi; segmentasyonda bunları kullanacağım
```

---

### 4. K-Means
```
add kmeans clustering with k=6

elbow ve silhouette grafiklerine bakarak k=6 seçtim,
centroid değerlerine göre segmentler isimlendirildi
```

---

### 5. Görselleştirmeler
```
add radar and bar charts for segment visualization
```

---

### 6. Hierarchical Clustering
```
add hierarchical clustering with ward linkage

dendrogram (n=2000 sample) ve agglomerative model eklendi,
kmeans ile silhouette karşılaştırması yapıldı
```

---

### 7. Son çıktılar
```
export segment labels and summary stats to csv
```

---

### 8. README
```
add readme with project overview and segment table
```

---

> **İpucu:** Commit tarihlerini dağıtmak istersen `git commit --date` kullanabilirsin, mesela:
> ```bash
> GIT_AUTHOR_DATE="2024-11-10 14:23:00" GIT_COMMITTER_DATE="2024-11-10 14:23:00" git commit -m "add data loading and initial exploration"
> ```
