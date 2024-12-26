from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import pydicom
from PIL import Image
import io
import torch
from pathlib import Path

app = FastAPI(title="Tıbbi Görüntü Analiz API")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Tıbbi Görüntü Analiz API'sine Hoş Geldiniz"}

@app.post("/analyze/brain-tumor")
async def analyze_brain_tumor(file: UploadFile = File(...)):
    # Görüntüyü oku ve işle
    contents = await file.read()
    
    if file.filename.endswith(('.dcm', '.DCM')):
        # DICOM dosyası işleme
        dataset = pydicom.dcmread(io.BytesIO(contents))
        image = dataset.pixel_array
    else:
        # Normal görüntü dosyası işleme
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # TODO: Model tahminini implement et
    result = {
        "tumor_detected": False,
        "confidence": 0.0,
        "location": None
    }
    
    return result

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
    uvicorn.run(app, host="0.0.0.0", port=8000) 