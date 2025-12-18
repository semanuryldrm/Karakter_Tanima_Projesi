import joblib
import numpy as np
import sys
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

print("==================================================")
print("   VISIONARY OCR - BİRLEŞİK SİSTEM TESTİ (V2)")
print("   (Düzeltilmiş Doğrulama Mantığı)")
print("==================================================")

# --- 1. AYARLAR VE FONKSİYONLAR ---
def extract_hog_emnist(img_array):
    # EMNIST verisi yan yatıktır, düzeltiyoruz (.T)
    img = img_array.reshape(28, 28).astype(np.uint8).T 
    return hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False)

def get_label_mapping(idx):
    """Sayısal kodu karaktere çevirir (Örn: 10 -> 'A')"""
    mapping = {i: str(i) for i in range(10)}
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghnqrt"
    for i, c in enumerate(chars): mapping[i+10] = c
    
    # Gelen değer string ise int'e çevirip bak, int ise direkt bak
    try:
        return mapping.get(int(idx), "?")
    except:
        return "?"

# --- 2. MODELLERİ YÜKLE ---
print("\n[1/4] Modeller Yükleniyor...")
try:
    model_ana = joblib.load("ocr_config.pkl")
    print("   -> Ana Model (Harf+Rakam) yüklendi.")
    
    model_uzman = joblib.load("rakam_uzmani.pkl")
    print("   -> Rakam Uzmanı (Destek) yüklendi.")
except Exception as e:
    print(f"HATA: Modeller bulunamadı! ({e})")
    sys.exit()

# --- 3. TEST VERİSİNİ HAZIRLA ---
print("[2/4] Test Verisi İndiriliyor (EMNIST)...")
emnist = fetch_openml(data_id=41039, parser='auto')
X, y = emnist["data"], emnist["target"]

# Veri hazırlığı
hedef_sayi = 50000 
X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
y_np = y.to_numpy() if hasattr(y, 'to_numpy') else y

if len(X) > hedef_sayi:
    y_str = y_np.astype(str)
    # Egitim.py ile aynı verileri seçmek için random_state=42 şart
    features_raw, _, labels_raw, _ = train_test_split(X_np, y_str, train_size=hedef_sayi, stratify=y_str, random_state=42)
else:
    features_raw, labels_raw = X_np, y_np

# HOG Çıkarma
print("   -> Özellikler çıkarılıyor (Biraz sürebilir)...")
features_hog = []
labels_true = []

for img, lbl in zip(features_raw, labels_raw):
    features_hog.append(extract_hog_emnist(img))
    labels_true.append(lbl)

# Test Ayrımı (%20)
_, X_test, _, y_test = train_test_split(features_hog, labels_true, test_size=0.2, random_state=42)

print(f"[3/4] {len(X_test)} adet test verisi üzerinde simülasyon başlıyor...")

# --- 4. SİMÜLASYON DÖNGÜSÜ ---
dogru_tahmin = 0
toplam = len(X_test)

for i in range(toplam):
    feat = X_test[i].reshape(1, -1)
    
    # --- KRİTİK DÜZELTME BURADA ---
    # Veri setinden gelen ham etiketi (Örn: '10') karaktere (Örn: 'A') çeviriyoruz
    ham_etiket = y_test[i]
    gercek_karakter = get_label_mapping(ham_etiket) 
    # ------------------------------

    # ADIM 1: Ana Model Tahmini
    idx_ana = model_ana.predict(feat)[0]
    tahmin_ana = get_label_mapping(idx_ana)
    
    # Güven skoru
    try:
        probs = model_ana.predict_proba(feat)[0]
        guven_ana = max(probs)
    except:
        guven_ana = 1.0 

    # ADIM 2: Karar Mekanizması (Hibrit Mantık)
    nihai_tahmin = tahmin_ana
    
    # Eğer ana model rakam dediyse VEYA güveni düşükse Uzmana sor
    if tahmin_ana.isdigit() or guven_ana < 0.6: # Eşik değerini biraz artırdık (0.6)
        try:
            # Uzman sadece rakam bilir (0-9)
            idx_uzman = model_uzman.predict(feat)[0]
            tahmin_uzman = str(idx_uzman)
            
            # Eğer ana model harf dediyse ama güveni düşükse, 
            # Uzman da rakamdan çok eminse değiştirebiliriz.
            # Şimdilik basit kural: Rakam şüphesinde uzman sözü geçer.
            if tahmin_ana.isdigit():
                 nihai_tahmin = tahmin_uzman
        except:
            pass

    # ADIM 3: Karşılaştırma
    if nihai_tahmin == gercek_karakter:
        dogru_tahmin += 1
        
    if (i+1) % 1000 == 0:
        sys.stdout.write(f"\r   >> İşlenen: {i+1}/{toplam}")
        sys.stdout.flush()

print("\n")
basari_orani = (dogru_tahmin / toplam) * 100

print("==================================================")
print(f"   BİRLEŞİK SİSTEM BAŞARISI: %{basari_orani:.2f}")
print("==================================================")