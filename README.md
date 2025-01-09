# MedVision: Makine Öğrenmesi ile Tümör ve Alzheimer Tespiti için Yapay Zeka Destekli Analiz Sistemi

Bu proje, ResNet50 derin öğrenme modeli ve makine öğrenmesi teknolojilerini kullanarak beyin MR görüntülerinde tümör tespiti ve Alzheimer hastalığı analizi yapabilen kapsamlı bir web tabanlı uygulamadır.

## Geliştiriciler

### Öğrenciler
Hüsniye Özdilek Mesleki ve Teknik Anadolu Lisesi  
Bilişim Teknolojileri - 12/A Sınıfı

### Proje Danışmanı
[@dagsevenonur](https://github.com/dagsevenonur)

## Özellikler

### Beyin Tümörü Tespiti
- ResNet50 derin öğrenme modeli ile otomatik tümör tespiti
- GradCAM teknolojisi ile tümör bölgesi görselleştirme
- Akıllı bounding box sistemi ile tümör konumu işaretleme
- Tespit sonuçlarını interaktif grafiklerle gösterme
- Detaylı olasılık dağılımları ve güven skorları
- Görüntü iyileştirme ve ön işleme teknikleri
- Yüksek doğruluk oranı ve düşük yanlış pozitif oranı

### Alzheimer Analizi
- ResNet50 tabanlı dört seviyeli Alzheimer sınıflandırması:
  - Normal: Belirgin bir patoloji yok
  - Hafif: Erken dönem belirtiler
  - Orta: Belirgin kognitif bozukluk
  - Şiddetli: İleri düzey nörodejenerasyon
- Her seviye için detaylı olasılık analizi
- Görsel ve sayısal sonuç raporlama
- Transfer learning ile optimize edilmiş model

### Görüntü İşleme Özellikleri
- Gelişmiş görüntü ön işleme:
  - CLAHE kontrast iyileştirme
  - Gürültü azaltma
  - Otomatik boyut normalizasyonu
- Kapsamlı görüntü düzenleme araçları:
  - Parlaklık ayarı (0-200%)
  - Kontrast kontrolü (0-200%)
  - 90° adımlarla döndürme
  - Yatay/dikey çevirme
  - Yakınlaştırma/uzaklaştırma (0.5x-2x)
- Gerçek zamanlı görüntü önizleme

### Kullanıcı Arayüzü
- Modern ve sezgisel tasarım
- Responsive ve mobil uyumlu arayüz
- Sürükle-bırak dosya yükleme
- İnteraktif sonuç görselleştirme
- Kapsamlı hata yönetimi ve kullanıcı bildirimleri
- Kolay gezinme için sekme tabanlı arayüz

### Raporlama ve Dışa Aktarma
- Detaylı PDF rapor oluşturma:
  - Hasta bilgileri
  - Analiz sonuçları
  - Görsel bulgular
  - Olasılık grafikleri
- CSV formatında veri dışa aktarma:
  - Sayısal sonuçlar
  - İstatistiksel veriler
  - Sınıflandırma detayları

## Teknik Altyapı

### Backend Mimarisi
- FastAPI web framework'ü:
  - Yüksek performanslı async işleme
  - Otomatik API dokümantasyonu
  - Güvenli dosya işleme
- PyTorch ve ResNet50:
  - Transfer learning ile özelleştirilmiş model
  - CUDA GPU desteği
  - Yüksek performanslı tahminleme
- OpenCV görüntü işleme:
  - Gelişmiş görüntü ön işleme
  - Gerçek zamanlı görüntü manipülasyonu
  - Özel renk uzayı dönüşümleri

### Frontend Teknolojileri
- React.js:
  - Komponent tabanlı mimari
  - Hooks ile durum yönetimi
  - Performans optimizasyonu
- Material-UI:
  - Modern ve responsive tasarım
  - Tema özelleştirme
  - Erişilebilirlik desteği
- Recharts:
  - İnteraktif veri görselleştirme
  - Özelleştirilebilir grafikler
  - Responsive grafik boyutlandırma

### Model Mimarisi
- Beyin Tümörü Modeli:
  - ResNet50 omurga
  - Transfer learning optimizasyonu
  - GradCAM görselleştirme
  - Binary sınıflandırma (Tümör var/yok)
  - Özel kayıp fonksiyonu
- Alzheimer Modeli:
  - ResNet50 omurga
  - Transfer learning
  - Çok sınıflı sınıflandırma
  - Dropout ve batch normalization
  - Özel aktivasyon fonksiyonları

## Kurulum

### Sistem Gereksinimleri
- İşletim Sistemi: Windows 10/11, Linux veya macOS
- Python: 3.8 veya üzeri
- Node.js: 14 veya üzeri
- RAM: Minimum 8GB (16GB önerilen)
- GPU: CUDA destekli NVIDIA GPU (önerilen)
- Depolama: Minimum 5GB boş alan

### Backend Kurulumu
1. Sanal ortam oluşturun:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows için
source .venv/bin/activate  # Linux/Mac için
```

2. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

3. Model dosyalarını yerleştirin:
```
models/
├── brain_tumor_best.pth    # Tümör tespit modeli
└── alzheimer_best.pth      # Alzheimer analiz modeli
```

4. Backend'i başlatın:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Kurulumu
1. Node.js bağımlılıklarını yükleyin:
```bash
cd frontend
npm install
```

2. Geliştirme sunucusunu başlatın:
```bash
npm start
```

3. Tarayıcıda açın:
```
http://localhost:3000
```

## Kullanım Kılavuzu

### Beyin Tümörü Analizi
1. "Beyin Tümörü" sekmesini seçin
2. MR görüntüsünü yükleyin
3. Gerekirse görüntü ayarlarını yapın
4. "Analiz Et" butonuna tıklayın
5. Sonuçları inceleyin:
   - Tümör varlığı/yokluğu
   - Güven skoru
   - Isı haritası görselleştirmesi
   - Bounding box işaretlemesi
   - Olasılık grafikleri

### Alzheimer Analizi
1. "Alzheimer" sekmesini seçin
2. MR görüntüsünü yükleyin
3. Gerekirse görüntü ayarlarını yapın
4. "Analiz Et" butonuna tıklayın
5. Sonuçları inceleyin:
   - Alzheimer seviyesi
   - Güven skoru
   - Seviye dağılım grafikleri
   - Detaylı analiz raporu

### Görüntü İşleme
- Parlaklık/Kontrast: Kaydırıcıları kullanın
- Döndürme: Döndürme butonlarını kullanın
- Çevirme: Yatay/dikey çevirme butonlarını kullanın
- Zoom: +/- butonları ile yakınlaştırın/uzaklaştırın

### Sonuçları Dışa Aktarma
1. Analiz tamamlandıktan sonra:
   - PDF raporu için "PDF Olarak İndir"
   - CSV verisi için "CSV Olarak İndir"
2. Dosyayı kaydedin ve açın

## Lisans
Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakın.

## İletişim
Proje ile ilgili sorularınız için:
- GitHub: [@dagsevenonur](https://github.com/dagsevenonur)
- Okul: Hüsniye Özdilek Mesleki ve Teknik Anadolu Lisesi

## Katkıda Bulunma
1. Bu depoyu fork edin
2. Yeni bir branch oluşturun
3. Değişikliklerinizi commit edin
4. Branch'inizi push edin
5. Pull request oluşturun

## Kullanılan Kaynaklar

### Veri Setleri
- [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) - Kaggle
- [Alzheimer's Dataset (4 class of Images)](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images) - Kaggle

### Modeller ve Algoritmalar
- [ResNet50](https://arxiv.org/abs/1512.03385) - Deep Residual Learning for Image Recognition
- [GradCAM](https://arxiv.org/abs/1610.02391) - Gradient-weighted Class Activation Mapping

### Yazılım Kütüphaneleri
- [PyTorch](https://pytorch.org/) - Derin öğrenme framework'ü
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [React](https://reactjs.org/) - Frontend kütüphanesi
- [Material-UI](https://mui.com/) - UI bileşen kütüphanesi
- [OpenCV](https://opencv.org/) - Görüntü işleme kütüphanesi

### Makaleler ve Dökümanlar
- [Brain Tumor Detection Using Convolutional Neural Networks](https://www.sciencedirect.com/science/article/pii/S1877050920307924)
- [Deep Learning for Alzheimer's Disease Detection](https://www.frontiersin.org/articles/10.3389/fnagi.2020.00184/full)
- [Medical Image Analysis using Deep Learning](https://www.nature.com/articles/s41598-019-42557-4) 