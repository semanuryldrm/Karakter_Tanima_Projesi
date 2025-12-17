import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, simpledialog # simpledialog eklendi
from PIL import Image, ImageDraw
import numpy as np
import cv2
import joblib
import os
import glob # Dosya tarama için eklendi
from skimage.feature import hog
from sklearn.svm import SVC # Yeniden eğitim için eklendi

# --- TASARIM VE AYARLAR ---
COLORS = {
    "bg": "#0F111A",
    "card": "#1C1F2B",
    "accent": "#007AFF",
    "success": "#34C759",
    "danger": "#FF3B30",
    "warning": "#FF9500",
    "text": "#FFFFFF",
    "canvas_bg": "#FFFFFF"
}

# Yeni verilerin kaydedileceği klasör
DATASET_PATH = "toplanan_veriler"
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

ctk.set_appearance_mode("Dark")

class VisionaryOCR_v5(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Visionary OCR v5.0 - Self-Learning")
        self.geometry("900x700") # Boyutu butonlar için biraz artırdık
        self.configure(fg_color=COLORS["bg"])

        self.model = self.load_model()
        
        # --- YENİ: RAKAM UZMANINI YÜKLE ---
        try:
            self.rakam_uzmani = joblib.load("rakam_uzmani.pkl")
        except:
            self.rakam_uzmani = None
            
        self.last_processed_img = None
        self.last_prediction = None
        
        # Grid Yapısı
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_left_panel()
        self.setup_right_panel()

    def setup_left_panel(self):
        self.left_container = ctk.CTkFrame(self, fg_color="transparent")
        self.left_container.grid(row=0, column=0, sticky="nsew", padx=30, pady=30)

        # Başlık
        ctk.CTkLabel(self.left_container, text="Yapay Zeka Çizim Alanı", 
                     font=("Segoe UI", 26, "bold"), text_color=COLORS["text"]).pack(anchor="w")
        ctk.CTkLabel(self.left_container, text="Modelin kendini geliştirmesi için tahmin sonrası geri bildirim verin.", 
                     font=("Segoe UI", 13), text_color="gray").pack(anchor="w", pady=(0, 20))

        # Canvas
        self.canvas_card = ctk.CTkFrame(self.left_container, fg_color=COLORS["canvas_bg"], corner_radius=20)
        self.canvas_card.pack(expand=True, fill="both")
        
        self.canvas = tk.Canvas(self.canvas_card, bg="white", highlightthickness=0, cursor="pencil")
        self.canvas.pack(expand=True, fill="both", padx=15, pady=15)
        
        self.image = Image.new("L", (400, 400), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Butonlar
        self.action_frame = ctk.CTkFrame(self.left_container, fg_color="transparent")
        self.action_frame.pack(fill="x", pady=(20, 0))

        self.btn_clear = ctk.CTkButton(self.action_frame, text="TEMİZLE", corner_radius=12,
                                       fg_color="#333645", font=("Segoe UI", 14, "bold"), 
                                       command=self.clear_canvas, height=45)
        self.btn_clear.pack(side="left", expand=True, padx=(0, 10))

        self.btn_analyze = ctk.CTkButton(self.action_frame, text="ANALİZ ET", corner_radius=12,
                                         fg_color=COLORS["accent"], font=("Segoe UI", 14, "bold"), 
                                         command=self.predict, height=45)
        self.btn_analyze.pack(side="left", expand=True)

    def setup_right_panel(self):
        self.right_panel = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=25)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 30), pady=30)

        # Tahmin Sonucu
        ctk.CTkLabel(self.right_panel, text="TAHMİN", font=("Segoe UI", 12, "bold"), text_color="gray70").pack(pady=(30, 0))
        self.lbl_char = ctk.CTkLabel(self.right_panel, text="?", font=("Segoe UI", 110, "bold"), text_color=COLORS["accent"])
        self.lbl_char.pack()

        # Güven Barı
        self.prog_bar = ctk.CTkProgressBar(self.right_panel, height=8, fg_color="#2A2D3E", progress_color=COLORS["accent"])
        self.prog_bar.set(0)
        self.prog_bar.pack(fill="x", padx=40, pady=10)
        self.lbl_conf_val = ctk.CTkLabel(self.right_panel, text="%0", font=("Segoe UI", 12, "bold"))
        self.lbl_conf_val.pack()

        # --- GERİ BİLDİRİM BÖLÜMÜ ---
        self.feedback_label = ctk.CTkLabel(self.right_panel, text="Tahmin doğru mu?", font=("Segoe UI", 13, "bold"))
        self.feedback_label.pack(pady=(30, 10))

        self.fb_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.fb_frame.pack(fill="x", padx=20)

        self.btn_correct = ctk.CTkButton(self.fb_frame, text="EVET ✅", fg_color="#28a745", state="disabled",
                                         command=self.save_correct, corner_radius=8, height=35)
        self.btn_correct.pack(side="left", expand=True, padx=5)

        self.btn_wrong = ctk.CTkButton(self.fb_frame, text="HAYIR ❌", fg_color="#dc3545", state="disabled",
                                       command=self.save_wrong, corner_radius=8, height=35)
        self.btn_wrong.pack(side="left", expand=True, padx=5)

        # --- YENİ EKLENEN: MODELİ EĞİT BÖLÜMÜ ---
        ctk.CTkFrame(self.right_panel, height=2, fg_color="gray30").pack(fill="x", padx=20, pady=20)
        
        self.btn_train = ctk.CTkButton(self.right_panel, text="MODELİ GÜNCELLE / EĞİT", 
                                       fg_color="#6f42c1", hover_color="#5a32a3",
                                       font=("Segoe UI", 13, "bold"), command=self.retrain_model)
        self.btn_train.pack(pady=10, padx=30, fill="x")

        # Model Önizleme
        ctk.CTkLabel(self.right_panel, text="MODELİN GÖRDÜĞÜ", font=("Segoe UI", 10, "bold"), text_color="gray50").pack(pady=(40, 5))
        self.preview_box = ctk.CTkLabel(self.right_panel, text="", width=120, height=120, fg_color="#000000", corner_radius=12)
        self.preview_box.pack(pady=5)

    def load_model(self):
        try: return joblib.load("ocr_config.pkl")
        except: 
            messagebox.showerror("Hata", "ocr_config.pkl bulunamadı!")
            self.destroy()

    def paint(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (400, 400), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_char.configure(text="?", text_color=COLORS["accent"])
        self.prog_bar.set(0)
        # BUTON SIFIRLAMA EKLEMESİ
        self.btn_correct.configure(state="disabled", text="EVET ✅")
        self.btn_wrong.configure(state="disabled")

    def predict(self):
        feat, final_img = self.preprocess_image(self.image)
        if feat is None: return

        self.last_processed_img = final_img
        
        # Önizleme
        p_img = Image.fromarray(cv2.resize(final_img, (120, 120), interpolation=cv2.INTER_NEAREST))
        ctk_img = ctk.CTkImage(light_image=p_img, dark_image=p_img, size=(120, 120))
        self.preview_box.configure(image=ctk_img)

        # Ana Model Tahmini
        idx = self.model.predict(feat)[0]
        char = self.get_label_mapping(idx)
        
        try:
            prob = max(self.model.predict_proba(feat)[0])
        except:
            prob = 1.0

        # --- YENİ: RAKAM DÜZELTME MANTIĞI ---
        # Eğer ana model rakam tahmin ederse veya güven çok düşükse uzmana sor
        if self.rakam_uzmani and (char.isdigit() or prob < 0.5):
            rakam_idx = self.rakam_uzmani.predict(feat)[0]
            char = str(rakam_idx)
            # Uzman modelin probasını al (opsiyonel)
            try:
                prob = max(self.rakam_uzmani.predict_proba(feat)[0])
            except:
                pass

        self.last_prediction = char # Kayıt için karakteri tut

        # UI
        color = COLORS["success"] if prob > 0.8 else COLORS["warning"]
        self.lbl_char.configure(text=char, text_color=color)
        self.lbl_conf_val.configure(text=f"%{prob*100:.1f}")
        self.prog_bar.set(prob)
        
        # Butonları Aktif Et ve Metni Sıfırla
        self.btn_correct.configure(state="normal", text="EVET ✅")
        self.btn_wrong.configure(state="normal")

    def save_correct(self):
        """Tahmin doğruysa görüntüyü etiketiyle kaydet"""
        if self.last_processed_img is not None:
            # Karakter rakamsa doğrudan kullan, harfse mappingden bul
            mapping = self.get_reverse_mapping()
            char_str = str(self.last_prediction)
            label = mapping.get(char_str, 0)
            
            timestamp = cv2.getTickCount()
            file_name = f"{DATASET_PATH}/label_{label}_{timestamp}.png"
            cv2.imwrite(file_name, self.last_processed_img)
            
            self.btn_correct.configure(state="disabled", text="KAYDEDİLDİ ✔")
            self.btn_wrong.configure(state="disabled")

    def save_wrong(self):
        """Tahmin yanlışsa kullanıcıya doğrusunu sor ve o etiketiyle kaydet"""
        dogru_karakter = simpledialog.askstring("Hata Düzeltme", "Bu karakter aslında nedir? (Örn: J, 5, 2)")
        
        if dogru_karakter:
            mapping = self.get_reverse_mapping()
            if dogru_karakter in mapping:
                label = mapping[dogru_karakter]
                timestamp = cv2.getTickCount()
                file_name = f"{DATASET_PATH}/label_{label}_{timestamp}.png"
                cv2.imwrite(file_name, self.last_processed_img)
                messagebox.showinfo("Bilgi", f"'{dogru_karakter}' olarak sisteme kaydedildi. Modeli güncellemeyi unutmayın.")
                self.clear_canvas()
            else:
                messagebox.showwarning("Uyarı", "Geçersiz karakter girdiniz! (0-9 ve A-Z arası kullanın)")

    def retrain_model(self):
        """Bu fonksiyon arayüz üzerinden sınırlı eğitim yapar; egitim.py ana eğitim içindir."""
        messagebox.showinfo("Bilgi", "En iyi performans için eğitimleri egitim.py dosyasından yapın.")

    def preprocess_image(self, pil_image):
        img = np.array(pil_image)
        coords = cv2.findNonZero(img)
        if coords is None: return None, None
        x, y, w, h = cv2.boundingRect(coords)
        crop = img[y:y+h, x:x+w]
        scale = 20 / max(h, w)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        final = np.zeros((28, 28), dtype=np.uint8)
        final[(28-nh)//2:(28-nh)//2+nh, (28-nw)//2:(28-nw)//2+nw] = resized
        feat = hog(final, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
        return feat.reshape(1, -1), final

    def get_label_mapping(self, idx):
        mapping = {i: str(i) for i in range(10)}
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghnqrt"
        for i, c in enumerate(chars): mapping[i+10] = c
        return mapping.get(int(idx), "?")

    def get_reverse_mapping(self):
        # 1. Rakamları eşle (0-9)
        rev = {str(i): i for i in range(10)}
        
        # 2. Büyük harfleri ve halihazırda var olan küçük harfleri eşle
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghnqrt"
        for i, c in enumerate(chars): 
            rev[c] = i + 10
        
        # 3. OLMAYAN TÜM KÜÇÜK HARFLERİ BÜYÜKLERİNE BAĞLA
        tum_harfler = "abcdefghijklmnopqrstuvwxyz"
        for char in tum_harfler:
            if char not in rev:
                buyuk_hali = char.upper()
                if buyuk_hali in rev:
                    rev[char] = rev[buyuk_hali]
                    
        return rev

if __name__ == "__main__":
    app = VisionaryOCR_v5()
    app.mainloop()