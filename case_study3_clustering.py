Case Study -3 – Project Code:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# ============================
# Load Dataset
# ============================
data = pd.DataFrame({
    "Jx": [
        0.56181,1.42607,1.09799,0.89799,0.23403,0.23399,0.08713,1.29926,0.90167,1.06211,
        0.03088,1.45487,1.24866,0.31851,0.27274,0.51702,0.82938,0.14945,0.64338,0.21582,
        1.07981,0.75321,0.94696,1.18193,1.11466,0.16641,0.75631,0.12197,0.46436,0.47364,
        0.66355,0.93404,0.47349,0.69295,0.21273,1.07245,0.85358,1.15385,0.05788,0.66927,
        1.05445,1.17023,0.39707,0.83502,1.16127,0.06029,0.26832,1.42261,0.99469,1.10116,
        0.25614,0.17878,0.87424,0.55157,0.48223,0.35199,0.87209,1.13113,0.58924,1.35633,
        0.39573,0.74069,0.78410,0.64131,0.03813,0.16184
    ],
    "Jy": [
        0.04714,0.95462,0.47153,0.76286,1.36135,0.37394,0.61557,1.13333,0.3432,0.11547,
        0.43463,0.24183,1.39455,1.21218,0.95011,1.15091,1.08813,1.18484,0.0191,0.90117,
        1.04301,0.33103,0.34239,0.73568,1.36519,0.32968,1.25498,0.34436,0.72904,0.20596,
        1.14914,0.13073,0.82972,1.04537,0.56405,0.325,0.66444,1.35757,0.07911,0.12565,
        0.68935,1.44088,1.10565,0.65453,1.40197,0.91332,0.33718,1.00796,0.66492,0.99412,
        1.21504,1.11676,0.69384,0.34832,1.15361,1.00636,0.90835,1.29053,0.90532,0.69487,
        0.09846,0.52381,1.08893,1.34566,1.33063,1.16981
    ],
    "Jz": [
        0.60508,-0.78965,-0.59593,1.24639,0.51607,-0.97701,-0.74632,0.65875,-0.98735,-0.59798,
        0.37183,0.72974,0.6299,-0.43933,0.78045,0.84672,-0.64769,-0.55216,0.57594,0.26005,
        0.98505,0.37635,-0.8626,1.21516,0.70924,-0.97941,-0.05303,0.63394,0.61135,-0.51064,
        0.8702,-0.19888,0.0349,0.81619,0.7953,-0.17346,0.45196,1.05015,-0.86464,0.44608,
        0.35035,0.65378,-0.5301,0.02843,1.01891,0.76802,0.70022,0.92287,-0.41652,0.88422,
        -0.63648,-0.12313,0.28768,-0.53932,0.97755,0.06421,0.22789,0.54812,-0.29487,0.70107,
        -0.75678,0.30561,0.92498,-0.46045,0.55723,-0.78663
    ],
    "Phase": [
        "Intermediate","FM","FM","XY","Intermediate","FM","FM","AFM","FM","FM",
        "Ising-like","Intermediate","AFM","FM","Intermediate","XY","FM","FM","Intermediate","Intermediate",
        "AFM","Intermediate","FM","XY","AFM","FM","XY","Ising-like","Intermediate","FM",
        "XY","FM","Intermediate","XY","Intermediate","FM","XY","AFM","FM","Intermediate",
        "XY","AFM","FM","XY","AFM","Intermediate","Intermediate","AFM","FM","XY",
        "FM","FM","XY","FM","XY","Intermediate","XY","AFM","FM","XY",
        "FM","XY","XY","FM","Intermediate","FM"
    ]
})


X = data[["Jx","Jy","Jz"]]
y = data["Phase"]

# Encode phases numerically
label = LabelEncoder()
y_encoded = label.fit_transform(y)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# PCA Projection
# ============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_encoded, cmap="viridis")
plt.title("PCA Projection (Colored by Phase)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()

# ============================
# Best K selection (Elbow + Silhouette)
# ============================
scores = []
for k in range(2,8):
    km = KMeans(n_clusters=k, random_state=0)
    labels_km = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_km)
    scores.append(score)
    print(f"K={k}, Silhouette={score:.4f}")

best_k = np.argmax(scores) + 2
print(f"\nBest K = {best_k}")

# ============================
# Final K-means Clustering
# ============================
kmeans = KMeans(n_clusters=best_k, random_state=0)
clusters = kmeans.fit_predict(X_scaled)
data["Cluster"] = clusters

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="tab10")
plt.title(f"PCA + K-means (K={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# ============================
# t-SNE Projection
# ============================
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_encoded, cmap="viridis")
plt.title("t-SNE Projection (Colored by Phase)")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=clusters, cmap="tab10")
plt.title("t-SNE + K-means Clusters")
plt.show()

# ============================
# Gaussian Mixture Model (GMM)
# ============================
gmm = GaussianMixture(n_components=best_k, random_state=0)
gmm_labels = gmm.fit_predict(X_scaled)
data["GMM_Cluster"] = gmm_labels

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=gmm_labels, cmap="cool")
plt.title("PCA + GMM Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# ============================
# Cluster Purity
# ============================
cross = pd.crosstab(data["Cluster"], data["Phase"])
print("\nCluster vs Phase:")
print(cross)

purity = np.sum(np.max(cross.values, axis=1)) / np.sum(cross.values)
print(f"\nCluster Purity = {purity:.3f}")

