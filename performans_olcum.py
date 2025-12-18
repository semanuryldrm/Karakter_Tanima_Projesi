import joblib
import numpy as np
import sys
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

print("==================================================")
print("   VISIONARY OCR - BİRLEŞİK SİSTEM TESTİ")
print("   (Ana Model + Rakam Uzmanı İşbirliği)")
print("==================================================")

# --- 1. AYARLAR VE FONKSİYONLAR ---
def extract_hog_emnist(img_array):
    """
    EMNIST verisi yan yatıktır. Hem Ana Model hem de Rakam Uzmanı 
    'dik' (upright) karakter bekler. Bu yüzden ikisi için de .T kullanıyoruz.
    """
    img = img_array.reshape(28, 28).astype(np.uint8).T 
    return hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False)

def get_label_mapping(idx):
    """Modelin sayısal çıktısını karaktere çevirir"""
    mapping = {i: str(i) for i in range(10)}
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghnqrt"
    for i, c in enumerate(chars): mapping[i+10] = c
    return mapping.get(int(idx), "?")

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
# Gerçek performans için EMNIST kullanıyoruz (Çünkü arayüzde harf de rakam da gelebilir)
emnist = fetch_openml(data_id=41039, parser='auto')
X, y = emnist["data"], emnist["target"]

# Eğitimde kullanılan mantıkla veriyi ayırıp, TEST kısmını alıyoruz
hedef_sayi = 50000 
X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
y_np = y.to_numpy() if hasattr(y, 'to_numpy') else y

if len(X) > hedef_sayi:
    y_str = y_np.astype(str)
    # random_state=42 ÇOK ÖNEMLİ: egitim.py ile aynı verileri seçmek için
    features_raw, _, labels_raw, _ = train_test_split(X_np, y_str, train_size=hedef_sayi, stratify=y_str, random_state=42)
else:
    features_raw, labels_raw = X_np, y_np

# Şimdi bu 50.000 veriyi HOG'a çevirelim
print("   -> Özellikler çıkarılıyor (Biraz sürebilir)...")
features_hog = []
labels_true = []

for img, lbl in zip(features_raw, labels_raw):
    features_hog.append(extract_hog_emnist(img))
    labels_true.append(lbl)

# Eğitim/Test ayrımı (Yine random_state=42 ile)
# Biz sadece modelin HİÇ GÖRMEDİĞİ %20'lik test kısmını alacağız.
_, X_test, _, y_test = train_test_split(features_hog, labels_true, test_size=0.2, random_state=42)

print(f"[3/4] {len(X_test)} adet test verisi üzerinde simülasyon başlıyor...")

# --- 4. SİMÜLASYON DÖNGÜSÜ (ARAYÜZ MANTIĞI) ---
dogru_tahmin = 0
toplam = len(X_test)

# Toplu tahmin yapmak yerine arayüz mantığını her örnek için tek tek uyguluyoruz
# Bu biraz yavaş olabilir ama en gerçekçi sonucu verir.

for i in range(toplam):
    feat = X_test[i].reshape(1, -1) # Tekil örnek
    gercek = y_test[i]
    
    # ADIM 1: Ana Model Tahmini
    idx_ana = model_ana.predict(feat)[0]
    tahmin_ana = get_label_mapping(idx_ana)
    
    # Güven skorunu al (predict_proba)
    try:
        probs = model_ana.predict_proba(feat)[0]
        guven_ana = max(probs)
    except:
        guven_ana = 1.0 # Prob yoksa tam güven varsay

    # ADIM 2: Karar Mekanizması (Arayüzdeki If Bloğu)
    nihai_tahmin = tahmin_ana
    
    # Eğer ana model rakam dediyse VEYA güveni %50'den düşükse Uzmana sor
    if tahmin_ana.isdigit() or guven_ana < 0.5:
        # Uzman sadece rakam bilir, o yüzden çıkan sonucu direkt rakam kabul ederiz
        idx_uzman = model_uzman.predict(feat)[0]
        tahmin_uzman = str(idx_uzman)
        
        # Sadece ana tahmin zaten rakamsa veya uzman çok eminse değiştirebiliriz
        # Basit senaryo: Rakam şüphesi varsa uzman ne derse odur.
        nihai_tahmin = tahmin_uzman

    # ADIM 3: Kontrol
    if nihai_tahmin == gercek:
        dogru_tahmin += 1
        
    # İlerleme çubuğu
    if (i+1) % 1000 == 0:
        sys.stdout.write(f"\r   >> İşlenen: {i+1}/{toplam}")
        sys.stdout.flush()

print("\n")
# --- 5. SONUÇ ---
basari_orani = (dogru_tahmin / toplam) * 100

print("==================================================")
print(f"   BİRLEŞİK SİSTEM BAŞARISI: %{basari_orani:.2f}")
print("==================================================")
print("Not: Bu oran, kullanıcının arayüzde karşılaşacağı")
print("yaklaşık doğruluk oranıdır.")