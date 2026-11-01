import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load and pick 2 columns (no y, no split)
df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 2. Scale
X_scaled = StandardScaler().fit_transform(X)

# 3. Try k = 2 to 10
wcss, sil = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, labels))
    print(k, "| WCSS:", round(wcss[-1], 2), "| Silhouette:", round(sil[-1], 4))

# 4. Plot elbow and silhouette
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(range(2, 11), wcss, marker="o"); ax[0].set_title("Elbow (WCSS)")
ax[1].plot(range(2, 11), sil, marker="o"); ax[1].set_title("Silhouette")
plt.show()

# 5. Final model with k = 5
km = KMeans(n_clusters=5, n_init=10, random_state=42)
df["Cluster"] = km.fit_predict(X_scaled)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df["Cluster"])
plt.xlabel("Annual Income"); plt.ylabel("Spending Score")
plt.show()
