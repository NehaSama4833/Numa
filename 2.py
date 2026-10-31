
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
pca = PCA(n_components=2)
Xp_train = pca.fit_transform(X_train)
Xp_test = pca.transform(X_test)
print("Variance kept:", round(pca.explained_variance_ratio_.sum(), 3))
 
lda = LinearDiscriminantAnalysis(n_components=2)
Xl_train = lda.fit_transform(X_train, y_train)
Xl_test = lda.transform(X_test)
 
for name, a, b in [("Original", X_train, X_test), ("PCA", Xp_train, Xp_test), ("LDA", Xl_train, Xl_test)]:
    model = LogisticRegression(max_iter=1000).fit(a, y_train)
    print(name, round(accuracy_score(y_test, model.predict(b)), 4))
