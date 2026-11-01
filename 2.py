import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load
X, y = load_wine(return_X_y=True)
print("Shape:", X.shape)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. PCA (no y)
pca = PCA(n_components=2)
Xp_train = pca.fit_transform(X_train)
Xp_test = pca.transform(X_test)
print("Variance kept:", round(pca.explained_variance_ratio_.sum(), 3))

# 5. LDA (needs y)
lda = LinearDiscriminantAnalysis(n_components=2)
Xl_train = lda.fit_transform(X_train, y_train)
Xl_test = lda.transform(X_test)

# 6. Compare accuracy
for name, a, b in [("Original", X_train, X_test), ("PCA", Xp_train, Xp_test), ("LDA", Xl_train, Xl_test)]:
    model = LogisticRegression(max_iter=1000).fit(a, y_train)
    print(name, "| features:", a.shape[1], "| accuracy:", round(accuracy_score(y_test, model.predict(b)), 4))

# 7. Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(Xp_train[:, 0], Xp_train[:, 1], c=y_train); ax[0].set_title("PCA")
ax[1].scatter(Xl_train[:, 0], Xl_train[:, 1], c=y_train); ax[1].set_title("LDA")
plt.show()
