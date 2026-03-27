import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.minirocket2d import fit_2d, transform_2d


# -------------------------
# Dummy dataset (örnek)
# -------------------------
# Gerçek kullanımda buraya MNIST vs koyacaksın
X = np.random.rand(200, 28, 28).astype(np.float32)
y = np.random.randint(0, 10, 200)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------
# FEATURE EXTRACTION
# -------------------------
dilations, biases = fit_2d(X_train, num_features=5000)

X_train_feat = transform_2d(X_train, dilations, biases)
X_test_feat = transform_2d(X_test, dilations, biases)

# -------------------------
# CLASSIFIER
# -------------------------
clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
clf.fit(X_train_feat, y_train)

y_pred = clf.predict(X_test_feat)

acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# sonucu kaydet
with open("../results/result.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
