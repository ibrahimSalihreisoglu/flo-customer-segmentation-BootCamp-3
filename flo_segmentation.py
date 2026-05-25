import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")


# --- veri okuma & temizlik ---

df = pd.read_csv("flo_data_20k.csv")

date_cols = ["first_order_date", "last_order_date",
             "last_order_date_online", "last_order_date_offline"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

print(df.shape)
print(df.isnull().sum())
print(df.describe().T)


# --- yeni değişkenler ---

analysis_date = df["last_order_date"].max() + pd.Timedelta(days=2)

df["recency"]    = (analysis_date - df["last_order_date"]).dt.days
df["tenure"]     = (analysis_date - df["first_order_date"]).dt.days
df["frequency"]  = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["monetary"]   = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df["omni_ratio"] = df["order_num_total_ever_online"] / df["frequency"]

features = ["recency", "frequency", "monetary", "tenure", "omni_ratio"]
X = df[features].copy()


# --- standartlaştırma ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =====================================================
# K-MEANS
# =====================================================

# optimum k: elbow + silhouette
inertias, sil_scores = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels, sample_size=5000, random_state=42))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, "bo-", linewidth=2, markersize=7)
axes[0].set_title("Elbow")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].grid(alpha=0.3)

axes[1].plot(K_range, sil_scores, "ro-", linewidth=2, markersize=7)
axes[1].set_title("Silhouette")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/01_kmeans_optimal_k.png", dpi=150, bbox_inches="tight")
plt.close()

# elbow grafiğine bakınca 6 mantıklı görünüyor
K = 6
km_model = KMeans(n_clusters=K, init="k-means++", n_init=20, random_state=42)
df["kmeans_cluster"] = km_model.fit_predict(X_scaled) + 1

print(df.groupby("kmeans_cluster")[features].mean().round(2))
print(df["kmeans_cluster"].value_counts().sort_index())


# segment etiketleme — centroid değerlerine göre elle baktım, aşağıdaki sıra çıktı
centroids = pd.DataFrame(
    scaler.inverse_transform(km_model.cluster_centers_), columns=features
)
centroids.index = range(1, K + 1)

centroids["score"] = (
    centroids["monetary"]  / centroids["monetary"].max()  * 0.5 +
    centroids["frequency"] / centroids["frequency"].max() * 0.3 +
    (1 - centroids["recency"] / centroids["recency"].max()) * 0.2
)

labels_ordered = ["Şampiyonlar", "Sadık Müşteriler", "Potansiyel Sadık",
                   "Risk Altındaki", "Uyku Modundaki", "Yeni Müşteriler"]
ranked_segs = centroids["score"].sort_values(ascending=False).index.tolist()
seg_map = {seg: lbl for seg, lbl in zip(ranked_segs, labels_ordered)}

df["kmeans_label"] = df["kmeans_cluster"].map(seg_map)
print(df.groupby(["kmeans_cluster", "kmeans_label"]).size().reset_index(name="n"))


# radar chart
def radar_chart(df_norm, title, path, cols):
    N = len(cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_norm)))
    for (seg, row), color in zip(df_norm.iterrows(), colors):
        vals = row[cols].tolist() + [row[cols[0]]]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=str(seg))
        ax.fill(angles, vals, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

norm_c = (centroids[features] - centroids[features].min()) / (centroids[features].max() - centroids[features].min())
norm_c.index = [seg_map[i] for i in norm_c.index]
norm_c["recency"] = 1 - norm_c["recency"]
radar_chart(norm_c, "K-Means Segment Profilleri", "outputs/02_kmeans_radar.png", features)


# büyüklük ve harcama grafikleri
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
seg_count = df.groupby("kmeans_label").size().sort_values(ascending=False)
seg_mon   = df.groupby("kmeans_label")["monetary"].mean().loc[seg_count.index]

axes[0].bar(seg_count.index, seg_count.values,
            color=plt.cm.Set2(np.linspace(0, 1, len(seg_count))))
axes[0].set_title("Segment Büyüklükleri")
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(seg_count.values):
    axes[0].text(i, v + 50, str(v), ha="center", fontsize=9)

axes[1].bar(seg_mon.index, seg_mon.values,
            color=plt.cm.Set1(np.linspace(0, 1, len(seg_mon))))
axes[1].set_title("Ortalama Harcama (₺)")
axes[1].tick_params(axis="x", rotation=30)
for i, v in enumerate(seg_mon.values):
    axes[1].text(i, v + 20, f"{v:.0f}₺", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/03_kmeans_bars.png", dpi=150, bbox_inches="tight")
plt.close()


# =====================================================
# HİERARŞİK KÜMELEME
# =====================================================

# 20k satırda dendrogram çok ağır, 2000 örnekle bakıyorum
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), size=2000, replace=False)
X_sample   = X_scaled[sample_idx]

lm = linkage(X_sample, method="ward")

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(lm, ax=ax, truncate_mode="lastp", p=30,
           leaf_rotation=90, leaf_font_size=9, show_contracted=True,
           color_threshold=lm[-5, 2])
ax.axhline(y=lm[-5, 2], color="red", linestyle="--", label="Kesim noktası")
ax.set_title("Ward Dendrogram (n=2000)")
ax.set_xlabel("Küme")
ax.set_ylabel("Mesafe")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/04_dendrogram.png", dpi=150, bbox_inches="tight")
plt.close()

# dendrogram'da da 6 civarı belirginleşiyor
hc_model = AgglomerativeClustering(n_clusters=6, linkage="ward")
df["hc_cluster"] = hc_model.fit_predict(X_scaled) + 1

hc_mean = df.groupby("hc_cluster")[features].mean().round(2)
print(hc_mean)
print(df["hc_cluster"].value_counts().sort_index())

hc_score = (
    hc_mean["monetary"]  / hc_mean["monetary"].max()  * 0.5 +
    hc_mean["frequency"] / hc_mean["frequency"].max() * 0.3 +
    (1 - hc_mean["recency"] / hc_mean["recency"].max()) * 0.2
)
sorted_hc = hc_score.sort_values(ascending=False).index.tolist()
hc_map    = {seg: lbl for seg, lbl in zip(sorted_hc, labels_ordered)}
df["hc_label"] = df["hc_cluster"].map(hc_map)

norm_hc = (hc_mean - hc_mean.min()) / (hc_mean.max() - hc_mean.min())
norm_hc.index = [hc_map[i] for i in norm_hc.index]
norm_hc["recency"] = 1 - norm_hc["recency"]
radar_chart(norm_hc, "HC Segment Profilleri", "outputs/05_hc_radar.png", features)


# =====================================================
# KARŞILAŞTIRMA & ÇIKTI
# =====================================================

km_sil = silhouette_score(X_scaled, df["kmeans_cluster"] - 1, sample_size=5000, random_state=42)
hc_sil = silhouette_score(X_scaled, df["hc_cluster"] - 1,    sample_size=5000, random_state=42)

print(f"K-Means silhouette  : {km_sil:.4f}")
print(f"HC silhouette       : {hc_sil:.4f}")

df[["master_id", "kmeans_label", "hc_label"]].to_csv(
    "outputs/flo_customer_segments.csv", index=False
)

summary = df.groupby("kmeans_label").agg(
    count=("kmeans_cluster", "count"),
    recency_ort=("recency", "mean"),
    frequency_ort=("frequency", "mean"),
    monetary_ort=("monetary", "mean"),
    tenure_ort=("tenure", "mean"),
    omni_ratio_ort=("omni_ratio", "mean")
).round(2)
summary.to_csv("outputs/kmeans_segment_summary.csv")
print(summary)
