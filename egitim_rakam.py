import numpy as np
import joblib
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 1. Sadece Rakamları (MNIST) Çek
print("Rakam uzmanı eğitiliyor (Hızlı işlem)...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist["data"].to_numpy(), mnist["target"].to_numpy()

# 2. HOG Özellikleri (Arayüzle aynı ayarlarda)
def extract_hog(img_array):
    img = img_array.reshape(28, 28).astype(np.uint8)
    return hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2))

# 5.000 veri hızlı sonuç için idealdir
X_feat = [extract_hog(x) for x in X[:5000]]
y_feat = y[:5000]

# 3. Model Eğitimi (Rakamlara özel yüksek hassasiyet)
model_rakam = SVC(kernel='rbf', C=10, probability=True)
model_rakam.fit(X_feat, y_feat)

# 4. Kaydet
joblib.dump(model_rakam, "rakam_uzmani.pkl")
print("Rakam uzmanı hazır! Artık arayüzü açabilirsiniz.")