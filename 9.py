import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X_moons, _ = make_moons(n_samples=500, noise=0.1, random_state=42)
noise = np.random.RandomState(30).uniform([-2, -1.5], [3, 2], size=(50, 2))
X = np.vstack([X_moons, noise])
print("Total points:", len(X))

db = DBSCAN(eps=0.2, min_samples=5).fit(X)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
n_core = len(db.core_sample_indices_)
n_border = len(X) - n_noise - n_core
print("Clusters:", n_clusters, "| Core:", n_core, "| Border:", n_border, "| Noise:", n_noise)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=10)
plt.title("DBSCAN (noise = -1)")
plt.show()
