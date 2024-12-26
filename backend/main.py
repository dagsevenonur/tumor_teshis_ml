from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import pydicom
from PIL import Image
import io
import os
from brain_tumor_detector import BrainTumorDetector

app = FastAPI(title="Tıbbi Görüntü Analiz API")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model örneğini oluştur
brain_tumor_detector = BrainTumorDetector()

@app.get("/")
async def root():
    return {"message": "Tıbbi Görüntü Analiz API'sine Hoş Geldiniz"}

@app.post("/analyze/brain-tumor")
async def analyze_brain_tumor(file: UploadFile = File(...)):
    try:
        # Geçici dosya oluştur
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        # Modele gönder ve tahmin al
        result = brain_tumor_detector.detect(temp_path)
        
        # Geçici dosyayı sil
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Frontend'e uygun formatta yanıt döndür
        if result["success"]:
            return {
                "tumor_detected": result["has_tumor"],
                "confidence": result["confidence"],
                "all_probabilities": result["all_probabilities"]
            }
        else:
            return {
                "error": "Görüntü analizi sırasında bir hata oluştu",
                "details": result.get("error", "Bilinmeyen hata")
            }
            
    except Exception as e:
        # Hata durumunda geçici dosyayı temizle
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "error": "Görüntü analizi sırasında bir hata oluştu",
            "details": str(e)
        }

@app.post("/analyze/alzheimer")
async def analyze_alzheimer(file: UploadFile = File(...)):
    # TODO: Alzheimer analizi implement edilecek
    return {"risk_level": "low", "confidence": 0.0}

@app.post("/analyze/cancer")
async def analyze_cancer(file: UploadFile = File(...)):
    # TODO: Kanser analizi implement edilecek
    return {"cancer_detected": False, "confidence": 0.0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 