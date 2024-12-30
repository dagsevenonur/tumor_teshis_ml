# Tıbbi Görüntü Analiz Sistemi

Bu proje, beyin MR görüntülerinde tümör tespiti ve Alzheimer hastalığı analizi yapabilen web tabanlı bir uygulamadır.

## Geliştiriciler

### Öğrenciler
Hüsniye Özdilek Mesleki ve Teknik Anadolu Lisesi  
Bilişim Teknolojileri - 12/A Sınıfı

### Proje Danışmanı
[@dagsevenonur](https://github.com/dagsevenonur)

## Özellikler

### Beyin Tümörü Tespiti
- MR görüntülerinde tümör varlığını tespit eder
- GradCAM ile tümör bölgesini görselleştirir
- Bounding box ile tümör konumunu işaretler
- Tespit sonuçlarını grafiksel olarak gösterir
- Olasılık dağılımlarını görselleştirir

### Alzheimer Analizi
- Beyin MR görüntülerinde Alzheimer belirtilerini tespit eder
- Dört farklı seviyede sınıflandırma yapar:
  - Normal
  - Hafif
  - Orta
  - Şiddetli
- Analiz sonuçlarını grafiksel olarak gösterir

### Genel Özellikler
- Modern ve kullanıcı dostu arayüz
- Sürükle-bırak dosya yükleme
- Görüntü ön işleme ve iyileştirme
- Sonuçları PDF ve CSV formatında dışa aktarma
- Görüntü düzenleme araçları (parlaklık, kontrast, döndürme, vb.)

## Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- Node.js 14 veya üzeri
- CUDA destekli GPU (önerilen)

### Backend Kurulumu
1. Sanal ortam oluşturun ve aktif edin:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows için
source .venv/bin/activate  # Linux/Mac için
```

2. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

3. Model dosyalarını `models` klasörüne yerleştirin:
- `brain_tumor_best.pth`
- `alzheimer_best.pth`

4. Backend sunucusunu başlatın:
```bash
cd backend
uvicorn main:app --reload
```

### Frontend Kurulumu
1. Gerekli Node.js paketlerini yükleyin:
```bash
cd frontend
npm install
```

2. Geliştirme sunucusunu başlatın:
```bash
npm start
```

## Kullanım

1. Web tarayıcınızda `http://localhost:3000` adresine gidin
2. "Beyin Tümörü" veya "Alzheimer" sekmesini seçin
3. MR görüntüsünü sürükle-bırak ile yükleyin veya dosya seçiciyi kullanın
4. "Analiz Et" butonuna tıklayın
5. Sonuçları görüntüleyin ve gerekirse dışa aktarın

## Teknik Detaylar

### Backend
- FastAPI web framework'ü
- PyTorch ile derin öğrenme modelleri
- ResNet50 tabanlı özel model mimarisi
- GradCAM görselleştirme tekniği
- OpenCV görüntü işleme

### Frontend
- React.js web framework'ü
- Material-UI bileşen kütüphanesi
- Recharts grafik kütüphanesi
- Dropzone dosya yükleme
- PDF ve CSV dışa aktarma

## Proje Yapısı
```
.
├── backend/
│   ├── main.py                 # FastAPI ana uygulama
│   ├── brain_tumor_detector.py # Tümör tespit modülü
│   └── alzheimer_detector.py   # Alzheimer analiz modülü
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Ana React bileşeni
│   │   └── components/        # UI bileşenleri
│   └── public/                # Statik dosyalar
├── models/                    # Eğitilmiş model dosyaları
└── datasets/                  # Veri setleri
```

## Lisans
Bu proje MIT lisansı altında lisanslanmıştır.

## İletişim
Proje ile ilgili sorularınız için:
- GitHub: [@dagsevenonur](https://github.com/dagsevenonur)
- Okul: Hüsniye Özdilek Mesleki ve Teknik Anadolu Lisesi 