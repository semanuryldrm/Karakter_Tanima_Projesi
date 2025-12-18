# Karakter Tanıma Projesi 🧠✍️

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat&logo=python)
![Course](https://img.shields.io/badge/Ders-Örüntü%20Tanıma-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-SVC-green?style=flat)
![Status](https://img.shields.io/badge/Status-Tamamlandı-success)

> **Bu proje, Örüntü Tanıma dersi kapsamında geliştirilmiştir.**
> 
> Kullanıcıların çizim tablası üzerine yazdığı el yazısı karakterleri (harf ve rakam) algılayan, **Görüntü İşleme (HOG)** ve **Makine Öğrenmesi (SVC)** tekniklerini birleştiren hibrit bir karakter tanıma uygulamasıdır.

---

## 📑 İçindekiler
1. [Proje Hakkında](#-proje-hakkında)
2. [Öne Çıkan Özellikler](#-öne-çıkan-özellikler)
3. [Teknik Mimari ve Algoritma](#-teknik-mimari-ve-algoritma)
4. [Veri Setleri](#-veri-setleri)
5. [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
6. [Dosya Yapısı](#-dosya-yapısı)
7. [Gelecek Planları](#-gelecek-planları)

---

## 📖 Proje Hakkında

El yazısı tanıma (OCR), bilgisayarlı görü alanının en temel problemlerinden biridir. Tekil modeller genellikle birbirine yapısal olarak benzeyen karakterleri (Örneğin: `5` ve `S`, `1` ve `I`, `0` ve `O`) ayırt etmekte zorlanır.

Bu proje, bu karışıklığı gidermek amacıyla **"Dual-Model Ensemble" (Çift Modelli Hibrit Yapı)** yaklaşımını benimser. Sistem, genel bir sınıflandırıcı ile özelleşmiş bir alt sınıflandırıcıyı dinamik bir karar ağacı üzerinden yöneterek doğruluk oranını maksimize eder.

---

## ✨ Öne Çıkan Özellikler

* **Hibrit Karar Mekanizması:** Hem harf hem rakam tanıyan "Ana Model" ile sadece rakamlara odaklanan "Uzman Model" birlikte çalışır.
* **Gerçek Zamanlı Çizim Arayüzü:** `Tkinter` ve `PIL` kullanılarak geliştirilen kullanıcı dostu arayüz.
* **Hata Toleransı (Confidence Threshold):** Modelin tahmininden emin olmadığı durumlarda (Güven Skoru < %60) otomatik olarak uzman görüşüne başvurulur.
* **Veri Toplama Modülü:** Kullanıcı, yanlış tahmin durumunda "Doğrusu Bu" diyerek sisteme geri bildirim verebilir. Bu veriler `toplanan_veriler` klasöründe biriktirilir.
* **HOG Özellik Çıkarımı:** Işık değişimlerinden, çizgi kalınlığından ve küçük kaymalardan etkilenmeyen robust (sağlam) özellik çıkarımı.

---

## 🛠️ Teknik Mimari ve Algoritma

Proje, ham piksel verisini işleyip anlamlı sonuçlar üretmek için 3 aşamalı bir boru hattı (pipeline) kullanır:

### 1. Ön İşleme (Preprocessing)
Kullanıcının çizdiği görüntü şu aşamalardan geçer:
* **Grayscale:** Görüntü tek kanallı gri tonlamaya çevrilir.
* **Resize:** 28x28 piksel boyutuna indirgenir.
* **Transposition:** EMNIST veri setinin yapısına uygun olarak görüntü döndürülür (Rotate/Flip).

### 2. Özellik Çıkarımı (Feature Extraction)
Piksel değerlerini doğrudan kullanmak yerine **HOG (Histogram of Oriented Gradients)** yöntemi tercih edilmiştir.
* **Neden HOG?** Nesnenin rengine değil, kenar yönelimlerine ve şekline odaklandığı için el yazısı stillerindeki varyasyonlara karşı daha dayanıklıdır.
* **Parametreler:** `orientations=9`, `pixels_per_cell=(4, 4)`, `cells_per_block=(2, 2)`

### 3. Sınıflandırma (Classification) - SVC
Sınıflandırıcı olarak **Support Vector Classification (SVC)** algoritması kullanılmıştır.
* **Kernel:** `RBF` (Radial Basis Function) - Doğrusal olmayan verileri ayrıştırmak için.
* **C:** `10` - Hata payı ve genelleme arasındaki denge.
* **Probability:** `True` - Sonucun sadece sınıfını değil, % kaç ihtimalle o sınıf olduğunu (Güven Skoru) hesaplamak için.

---

## 📊 Veri Setleri

Projenin eğitiminde iki devasa veri seti kullanılmıştır:

1.  **EMNIST (Extended MNIST) - Balanced:**
    * Ana modelin eğitimi için kullanılmıştır.
    * 47 farklı sınıf (Büyük harf, küçük harf, rakamlar).
    * Toplam ~131.000 örneklem.
2.  **MNIST (Modified NIST):**
    * Sadece "Rakam Uzmanı" modelini eğitmek için kullanılmıştır.
    * 0-9 arası rakamlar.
    * 70.000 örneklem.

---

## 💻 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### Gereksinimler
* Python 3.8 veya üzeri
* Git & Git LFS (Büyük model dosyaları için zorunludur)

### Adım 1: Projeyi Klonlayın
```bash
# Önce Git LFS'yi kurun (Bir kereye mahsus)
git lfs install

# Projeyi indirin
git clone [https://github.com/semanuryldrm/Karakter_Tanima_Projesi.git](https://github.com/semanuryldrm/Karakter_Tanima_Projesi.git)
cd Karakter_Tanima_Projesi
```
### Adım 2: Sanal Ortam Oluşturun (Önerilen)

Projeyi izole bir alanda çalıştırmak ve sisteminizdeki diğer Python kütüphaneleriyle çakışma yaşamamak için sanal ortam (virtual environment) kullanmanız tavsiye edilir.

```bash
# 1. Sanal ortamı oluşturun
python -m venv .venv

# 2. Ortamı aktifleştirin
# Windows için:
.venv\Scripts\activate

# Mac/Linux için:
source .venv/bin/activate
```
### Adım 3: Kütüphaneleri Yükleyin

Projenin sorunsuz çalışabilmesi için gerekli olan görüntü işleme, yapay zeka ve arayüz kütüphanelerini aşağıdaki komutla yükleyin:

```bash
pip install numpy scikit-learn scikit-image opencv-python pillow customtkinter joblib
```
### Adım 4: Uygulamayı Başlatın

Kurulum tamamlandıktan sonra, çizim arayüzünü (GUI) başlatmak ve sistemi test etmek için şu komutu çalıştırın:

```bash
python arayuz.py
```
Bilgi: Proje içerisinde eğitilmiş model dosyaları (ocr_config.pkl ve rakam_uzmani.pkl) hazır olarak gelmektedir. Doğrudan kullanmaya başlayabilirsiniz.

### (Opsiyonel) Modelleri Sıfırdan Eğitmek

Eğer modelleri kendi bilgisayarınızda yeniden eğitmek isterseniz şu komutları kullanabilirsiniz:

```bash
python egitim.py        # Ana modeli (Harf+Rakam) eğitir (~15-20 dk)
python egitim_rakam.py  # Rakam uzmanını eğitir (~1 dk)
```

    ---

## 📂 Dosya Yapısı

| Dosya Adı | Açıklama |
| :--- | :--- |
| `arayuz.py` | 🎨 Kullanıcının çizim yapabileceği, Tkinter tabanlı GUI. Tahmin yapar ve veri toplar. |
| `egitim.py` | 🧠 **Ana Modeli** (EMNIST verisi ile) eğiten script. |
| `egitim_rakam.py` | 🔢 **Rakam Uzmanını** (MNIST verisi ile) eğiten script. |
| `performans_olcum.py` | 📊 Hibrit sistemin (Ana Model + Rakam Uzmanı) birlikte çalıştığı senaryoyu simüle eden ve gerçek başarıyı ölçen test aracı. |
| `toplanan_veriler/` | 💾 Kullanıcının geri bildirimleriyle (Doğru/Yanlış) kaydedilen yeni veri örnekleri. |

    ---

## 🔮 Gelecek Planları (Roadmap)

Bu proje yaşayan bir sistemdir ve geliştirmeler devam etmektedir. Önümüzdeki dönem için hedeflenen temel iyileştirmeler şunlardır:

- [ ] **🔄 Aktif Öğrenme (Active Learning) Entegrasyonu:**
    - Şu an arayüzde bulunan *"Modeli Güncelle"* butonu işlevsel hale getirilecek.
    - Kullanıcının `toplanan_veriler` klasörüne kaydettiği geri bildirimler (yanlış bilinen ve kullanıcının düzelttiği harfler), otomatik bir boru hattı (pipeline) ile modele beslenecek. Böylece model kullanıldıkça akıllanacak.

- [ ] **🧠 Derin Öğrenme (Deep Learning) Dönüşümü:**
    - Mevcut **SVC + HOG** mimarisi, daha karmaşık el yazılarını ve gürültülü görüntüleri işleyebilmek adına **CNN (Convolutional Neural Networks)** mimarisine evrilecek.
    - Hedef: %90 olan başarı oranını %99.5 seviyesine çıkarmak.

- [ ] **📝 Kelime ve Cümle Tanıma (Segmentation):**
    - Şu an sistem tek tek karakterleri tanımaktadır.
    - Görüntü işleme teknikleri (OpenCV) kullanılarak, yan yana yazılan harflerin otomatik olarak ayrıştırılması (Character Segmentation) ve kelime bütünlüğü içinde tanınması sağlanacak.

- [ ] **📱 Mobil ve Web API:**
    - Eğitilen modelin `FastAPI` veya `Flask` ile bir REST API haline getirilmesi.
    - Bu sayede modelin bir mobil uygulama (Flutter/React Native) üzerinden fotoğraf çekerek kullanılabilmesi.

    ---

**Geliştirici:** [Semanur Yıldırım](https://github.com/semanuryldrm)