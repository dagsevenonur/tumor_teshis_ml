# Tıbbi Görüntü Analizi Yapay Zeka Sistemi

Bu proje, MR ve BT görüntülerini analiz ederek aşağıdaki teşhisleri yapabilen bir yapay zeka sistemidir:
- Beyin tümörü tespiti
- Alzheimer riski analizi
- Kanser teşhisi

## Teknolojiler

### Backend
- Python 3.9+
- FastAPI
- PyTorch
- OpenCV
- pydicom

### Frontend
- Electron.js
- React
- Material-UI

## Kurulum

1. Python bağımlılıklarını yükleyin:
```bash
pip install -r requirements.txt
```

2. Frontend bağımlılıklarını yükleyin:
```bash
cd frontend
npm install
```

## Geliştirme

### Backend'i başlatmak için:
```bash
uvicorn main:app --reload
```

### Frontend'i başlatmak için:
```bash
cd frontend
npm start
``` 