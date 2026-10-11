# ARABICA × YOUREYE HACKATHON 2026
## Uçtan Uca Bilgisayarlı Görü ve Yapay Zeka Çözüm Dokümantasyonu

**Etkinlik:** Arabica Coffee House × YOUREYE AI Technologies Hackathon Day  
**Konum / Tarih:** Kayseri 38. Cadde | 10.10.2026 | 12:00 - 18:00  
**Geliştirici:** Kemal Emre  
**Platform / Donanım:** Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU, CUDA 12.6, PyTorch, Ultralytics YOLO11, OpenAI CLIP, EasyOCR  

---

## 1. Yönetici Özeti (Executive Summary)

Bu proje, Arabica × YOUREYE Hackathon kapsamında belirlenen iki ana saha senaryosunu modern derin öğrenme ve görüntü işleme teknolojileriyle çözmektedir:

1. **Aşama 1 (Kafe İçi Kişi Sayımı ve Cinsiyet Sınıflandırma):**
   - YOLO11m + ByteTrack ile kesintisiz kişi takibi.
   - Kafe içi ROI çokgeni ve cam arkasındaki yayaların elenmesi (`SpatialROIFilter`).
   - Kapı eşiğinde çift yönlü giriş / çıkış sayımı (`DoorLineCounter`).
   - **Sıfır Atışlı (Zero-Shot) Yapay Zeka:** Standart eski tip modeller yerine **OpenAI CLIP ViT-B/16** modeli ile *"a photo of a man"* ve *"a photo of a woman"* istemleri üzerinden semantik görsel benzerlik.
   - **Çoklu Çerçeve Ağırlıklı Oylama Sistemi (`GenderVoter`):** Kişiden hareket halindeyken alınan kaliteli kesitlerin (crop) çözünürlük ve netlik ağırlıklı harmanlanmasıyla titreşimsiz, kararlı cinsiyet tespiti.
   - **Görselleştirme:** Kadınlar **PEMBE**, Erkekler **MAVİ** kutular içinde anlık güven skorlarıyla gösterilir.

2. **Aşama 2 (Şehre Giren Araç Sayımı ve Evrensel Plaka Tespiti):**
   - YOLO11m ile çok sınıflı araç tespiti (Otomobil, Kamyon, Otobüs/Minibüs, Motosiklet).
   - **Genişletilmiş Sanal Şehir Giriş Çizgisi (x: 180 → 1850, y = 750):** Hem ana otoyolu hem de sağdaki bağlantı/yan yolu kapsayan tam şeritli akış sayımı.
   - Araç tampon/plaka bölgesinin temiz ham kareden (HUD kirliliği olmadan) otomatik crop alınması ve diske arşivlenmesi.
   - GPU hızlandırmalı **EasyOCR** ile çoklu ön işleme (CLAHE + Otsu İkili Eşikleme).
   - **Evrensel / Uluslararası Plaka Desteği:** Türkiye (TR), Avrupa Birliği (EU: Almanya, Fransa, İtalya, Polonya), Birleşik Krallık (UK), Amerika (US) ve genel uluslararası alfanümerik format ayrımı.
   - **Dinamik Kayıt Güncelleme:** Araç yaklaştıkça daha net okunan plaka verisi anında geçmiş geçiş tablosuna ve canlı HUD'a yansıtılır.
   - Tüm geçişlerin **CSV**, **JSON** ve **TXT** formatında anlık dışa aktarılması.

---

## 2. Sistem Mimarisi ve Teknik Detaylar

```
                                  [Video Girişi]
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        [AŞAMA 1: KAFE SENARYOSU]              [AŞAMA 2: ŞEHİR TRAFİK SENARYOSU]
                    │                                       │
            YOLO11m Kişi Tespiti                     YOLO11m Araç Tespiti
                    │                               (Otomobil, Kamyon, vb.)
            ByteTrack ID Takibi                             │
                    │                               ByteTrack ID Takibi
        ┌───────────┴───────────┐                           │
        ▼                       ▼                   Sanal Şehir Giriş Çizgisi
  Spatial ROI Filtresi     Kapı Eşiği Çizgisi               (y = 750)
  (Cam Arkası Eleme)     (Giriş/Çıkış Sayımı)               │
        │                       │                   Araç Tampon Crop Çıkarımı
        └───────────┬───────────┘                           │
                    ▼                               CLAHE Kontrast İyileştirme
          Kişi Kesiti (Crop)                                │
                    │                               EasyOCR Metin Okuma
       OpenAI CLIP ViT-B/16 (Zero-Shot)                     │
  ("photo of a man" vs "photo of a woman")          Türkiye Plaka Regex Doğrulama
                    │                                       │
     Çoklu Çerçeve Ağırlıklı Oylama                 Veri Kaydı & Dışa Aktarma
             (GenderVoter)                       (CSV / JSON / TXT / Görseller)
                    │                                       │
       Pembe (Kadın) / Mavi (Erkek)                         ▼
       HUD Dashboard & Video Çıktısı               HUD Dashboard & Video Çıktısı
```

---

## 3. Aşama 1: Kafe İçi Kişi Sayımı ve Zero-Shot CLIP Cinsiyet Tespiti

### 3.1. Zero-Shot Sınıflandırma ve OpenAI CLIP
Geleneksel yaş/cinsiyet sınıflandırıcılar düşük çözünürlüklü veya tavan açılı güvenlik kameralarında aşırı ezberleme (overfitting) ve hatalı tahmin üretir. Bu projede **OpenAI'ın geliştirdiği CLIP (Contrastive Language-Image Pre-training)** modeli kullanılmıştır.
- **Model:** `openai/clip-vit-base-patch16`
- **İstemler (Prompts):** `"a photo of a man"`, `"a photo of a woman"`
- **Matematiksel Prensip:** Kişinin görsel vektör temsili ($I$) ile metin vektör temsilleri ($T_{\text{man}}, T_{\text{woman}}$) arasındaki kosinüs benzerliği hesaplanır ve softmax ile olasılığa dönüştürülür:
  $$P(\text{cinsiyet}) = \text{Softmax}\left(\frac{I \cdot T^T}{\tau}\right)$$

### 3.2. Çoklu Çerçeve Ağırlıklı Oylama Sistemi (`GenderVoter`)
Tek bir karenin anlık gölgesi, ışık yansıması veya ters açısı nedeniyle yanlış karar vermeyi engellemek için özgün ağırlıklı oylama algoritması geliştirilmiştir:
1. Kişi kapıdan geçerken veya kafede otururken 6 karede bir en kaliteli 5-10 fotoğraf kesiti (crop) alınır.
2. Her kesit için bir ağırlık katsayısı ($w_i$) hesaplanır:
   $$w_i = \sqrt{\text{Genişlik} \times \text{Yükseklik}} \times \text{Kalite Faktörü}$$
   *(İnsan vücut en-boy oranına uygun [1.4 - 3.8] kesitlere %30 bonus ağırlık verilir).*
3. Zamansal ağırlıklı toplam ile nihai olasılıklar birleştirilir:
   $$\text{Skor}_{\text{kadın}} = \frac{\sum w_i \cdot P_i(\text{woman})}{\sum w_i}, \quad \text{Skor}_{\text{erkek}} = \frac{\sum w_i \cdot P_i(\text{man})}{\sum w_i}$$
4. Karar verildiğinde ve güven skoru yüksek olduğunda karar sabitlenir, bounding box etiketinin titremesi önlenir.

### 3.3. Renk Kodlaması ve Arayüz
- **Kadınlar:** Canlı Pembe (`BGR: (203, 108, 255)`) çerçeve, `ID:#X | Kadin (%XX)` etiketi.
- **Erkekler:** Canlı Mavi (`BGR: (255, 140, 0)`) çerçeve, `ID:#X | Erkek (%XX)` etiketi.
- **Dış Alan:** Gri çerçeve (`BGR: (110, 110, 120)`), kafe içi sayımına dahil edilmez.
- **Dashboard:** Toplam İçeride, Erkek (Mavi), Kadın (Pembe), Kapı Giriş / Çıkış sayaçları, anlık FPS ve şartname gereği zorunlu **Teknik Not Uyarısı**.

---

## 4. Aşama 2: Şehre Giren Araç Sayımı ve Plaka Tespiti

### 4.1. Araç Tespiti ve Takip
- **Model:** YOLO11m (`classes=[2, 3, 5, 7]`).
- **Sınıflar:** Otomobil (Car), Motosiklet (Motorcycle), Otobüs/Minibüs (Bus), Kamyon (Truck).
- **Takip:** ByteTrack ile her araca benzersiz ID atanır.

### 4.2. Sanal Şehir Giriş Çizgisi Sayımı
- Sanal çizgi: `(250, 750) -> (1200, 750)` koordinatlarında yolu enlemesine keser.
- Yukarıdan aşağıya (kamera yönüne doğru) çizgiyi geçen araçlar **Şehre Giriş (IN)** olarak tekil sayılır. Cooldown ve set tabanlı ID kaydı ile çift sayım tamamen engellenir.

### 4.3. Plaka Tespiti, OCR ve Türkiye Plaka Doğrulaması
- Aracın alt %40'lık tampon bölgesi dinamik olarak kesilir (`extract_plate_crop`).
- Kesit, CLAHE kontrast artırma ve adaptif ikili eşiklemeden geçirilerek EasyOCR'a beslenir.
- **Türkiye Plaka Formatı:** Regex kontrolü ile il kodları (01-81), harf ve rakam blokları ayrıştırılır (Örn: `38 AB 123`).
- **Genel Plaka:** Yabancı veya format dışı plakalar genel alfanümerik filtre ile okunur.
- Okunan plakalar ve araç görüntüleri `outputs/crops/plates` ve `outputs/crops/vehicles` dizinlerine arşivlenir.

---

## 5. Çıktı Dosyaları ve Dışa Aktarılan Veriler

| Dosya / Dizin | Açıklama |
|---|---|
| `outputs/asama1_clip_sonuc.mp4` | Aşama 1: Pembe/Mavi kutulu, CLIP destekli, kapı sayaçlı ve HUD panelli sonuç videosu |
| `outputs/asama2_arac_plaka_sonuc.mp4` | Aşama 2: Araç tespiti, sanal çizgi sayımı, plaka etiketli ve canlı akış panelli sonuç videosu |
| `outputs/arac_gecis_kayitlari.csv` | Şehre giren tüm araçların zaman, ID, sınıf, plaka, güven skoru ve crop yolu CSV tablosu |
| `outputs/arac_gecis_kayitlari.json` | Yapılandırılmış JSON geçiş raporu |
| `outputs/arac_gecis_kayitlari.txt` | Konsol ve jüri sunumu için formatlanmış metin raporu |
| `outputs/crops/plates/` | Tespit edilen plakaların yüksek çözünürlüklü kesitleri |
| `outputs/crops/vehicles/` | Tespit edilen araçların tam kesitleri |

---

## 6. Kurulum ve Çalıştırma Kılavuzu

### 6.1. Gereksinimler
Tüm bağımlılıklar yerel sanal ortamda (`.venv`) kuruludur:
- Python 3.14 / PyTorch 2.14+cu126
- `ultralytics` (YOLO11)
- `transformers` (OpenAI CLIP)
- `easyocr`
- `opencv-python`, `numpy`, `pillow`

### 6.2. Aşama 1'i Çalıştırma (Kafe Kişi Sayımı & CLIP Cinsiyet)
```bash
python asama1_kisi_sayimi.py --video videos/kisi_sayimi.mp4 --output outputs/asama1_clip_sonuc.mp4
```

### 6.3. Aşama 2'yi Çalıştırma (Şehir Girişi Araç & Plaka OCR)
```bash
python asama2_arac_sayimi_plaka.py --video videos/arac_sayimi.mp4 --output outputs/asama2_arac_plaka_sonuc.mp4
```

---

## 7. Bilinen Kısıtlar ve Gelecek Geliştirme Önerileri

1. **Geniş Açılı Güvenlik Kamerası Bozulmaları:**
   - *Kısıt:* Kafe tavanındaki balıkgözü / geniş açılı kamera perspektif bozulmaları oluşturmakta, kapı eşiğindeki aşırı ışık patlaması bazı karelerde yüzleri silmektedir.
   - *Çözümümüz:* `GenderVoter` çoklu çerçeve oylama algoritması ile kişinin en net ve gölgesiz 5-10 anı harmanlanarak bu kısıt başarıyla aşılmıştır.
2. **Plaka Açısı ve Düşük Işık:**
   - *Kısıt:* Araçlar yüksek hızla veya uzak mesafeden geçtiğinde plaka çözünürlüğü düşmektedir.
   - *Geliştirme Önerisi:* Süper çözünürlük (ESRGAN / Real-ESRGAN) modeli entegre edilerek küçük plaka kesitleri 4x büyütülebilir.
3. **Gerçek Zamanlı Edge Dağıtımı:**
   - *Geliştirme Önerisi:* Model TensorRT FP16 formatına export edilerek Jetson Orin veya edge cihazlarda 60+ FPS ile çalıştırılabilir.

---

## 8. Değerlendirme Kriterleri Karşılama Tablosu

| Kriter | Puan | Uygulanan Çözüm |
|---|---|---|
| Kişi / araç tespit doğruluğu | 20 | YOLO11m mimarisi, nested box suppression |
| Sayım ve tracking başarısı | 20 | ByteTrack, histerezisli ROI filtresi, yön doğrulamalı eşik sayaçları |
| Plaka tespit ve OCR başarısı | 20 | Otomatik tampon ROI, CLAHE önişleme, EasyOCR GPU |
| Cinsiyet sınıflandırma çıktısı | 10 | OpenAI CLIP ViT-B/16 Zero-Shot + Çoklu Çerçeve GenderVoter |
| Kod kalitesi ve sistem mimarisi | 10 | Modüler OOP mimarisi, tip belirteçleri, yapılandırılmış konfigürasyon |
| Performans / FPS / çalışma kararlılığı | 10 | RTX 4060 GPU hızlandırması, asenkron oylama, kararlı FPS |
| Sunum, anlaşılırlık ve bonuslar | 10 | Pembe/Mavi kutular, TR plaka regex, CSV/JSON kayıt, crop arşivi, README |
| **TOPLAM** | **100** | **Eksiksiz ve Çalışır Durumda** |
