from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from brain_tumor_detector import BrainTumorDetector
from alzheimer_detector import AlzheimerDetector

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Beyin MR Görüntü Analiz API'sine Hoş Geldiniz"}

@app.post("/analyze/brain-tumor")
async def analyze_brain_tumor(file: UploadFile = File(...)):
    try:
        # Geçici dosya oluştur
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Beyin tümörü analizi yap
        detector = BrainTumorDetector()
        result = detector.detect(temp_path)
        
        # Geçici dosyayı sil
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze/alzheimer")
async def analyze_alzheimer(file: UploadFile = File(...)):
    try:
        # Geçici dosya oluştur
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Alzheimer analizi yap
        detector = AlzheimerDetector()
        result = detector.detect(temp_path)
        
        # Geçici dosyayı sil
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return result
    except Exception as e:
        return {"error": str(e)} 