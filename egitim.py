import numpy as np
import joblib
import time
import sys
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def log(msg):
    print(f"[SİSTEM] {msg}")

# --- 1. VERİ İNDİRME ---
log("1. EMNIST (Harf) Veri Seti İndiriliyor (ID: 41039)...")
print("   (Yüksek doğruluk için geniş veri seti çekiliyor...)")

try:
    # ID: 41039 -> EMNIST Balanced (Doğru Harf Seti)
    emnist = fetch_openml(data_id=41039, parser='auto')
    X, y = emnist["data"], emnist["target"]
except Exception as e:
    log(f"İndirme Hatası: {e}")
    exit()

log(f"2. Veri İndirildi. Toplam havuz: {len(X)} örnek.")

# --- 2. HOG MOTORU (TRANSPOSE DAHİL) ---
def extract_hog_features(img_array):
    img = img_array.reshape(28, 28).astype(np.uint8)
    
    # EMNIST yan yatık olduğu için düzeltiyoruz (.T)
    img = img.T 
    
    features = hog(img, orientations=9, pixels_per_cell=(4, 4),
                   cells_per_block=(2, 2), visualize=False)
    return features

# --- 3. VERİ HAZIRLIĞI (ARTTIRILMIŞ KAPASİTE) ---
log("3. Özellikler Çıkarılıyor ve Veri Hazırlanıyor...")

# Güvenli rakam 50.000 olarak güncellendi.
hedef_sayi = 50000 

X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
y_np = y.to_numpy() if hasattr(y, 'to_numpy') else y

if len(X) > hedef_sayi:
    y_str = y_np.astype(str)
    # Stratify: Her harften eşit oranda al ki model bazı harfleri kayırmasın.
    X_small, _, y_small, _ = train_test_split(X_np, y_str, train_size=hedef_sayi, stratify=y_str, random_state=42)
else:
    X_small, y_small = X_np, y_np

# Özellik çıkarma döngüsü
features_list = []
labels_list = []

total = len(X_small)
rapor_adimi = total // 10 
print("-" * 50)

for i, (img, label) in enumerate(zip(X_small, y_small)):
    try:
        feat = extract_hog_features(img)
        features_list.append(feat)
        labels_list.append(label)
    except:
        continue
    
    if (i + 1) % rapor_adimi == 0:
        yuzde = ((i + 1) / total) * 100
        sys.stdout.write(f"\r   >> İşleniyor: %{yuzde:.0f} tamamlandı.")
        sys.stdout.flush()

print("\n" + "-" * 50)

# --- 4. TRAIN / TEST AYRIMI ---
log("4. Veri Seti Bölünüyor (Eğitim vs Test)...")
# Verinin %80'i ile çalışacak, %20'si ayrılacak 
X_train, X_test, y_train, y_test = train_test_split(features_list, labels_list, test_size=0.2, random_state=42)

log(f"   Eğitim Verisi: {len(X_train)} adet")
log(f"   Test Verisi  : {len(X_test)} adet")

# --- 5. EĞİTİM (MAX PERFORMANS) ---
log("5. Model Eğitiliyor (Bu işlem işlemciyi zorlayabilir)...")
start_time = time.time()

# verbose=True: O akan sayıları görmek için korundu.
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, verbose=True)
# Sadece X_train ile eğitiliyor
model.fit(X_train, y_train)

end_time = time.time()
print(f"\n   >> Eğitim Süresi: {end_time - start_time:.1f} saniye")


# --- 6. KAYIT ---
log("6. Kaydediliyor...")
joblib.dump(model, "ocr_config.pkl")
log("TAMAMLANDI. Yeni modeliniz hazır.")
