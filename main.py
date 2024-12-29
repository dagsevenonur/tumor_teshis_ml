from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from brain_tumor_detector import BrainTumorDetector
from alzheimer_detector import AlzheimerDetector
import os
import tempfile

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model örneklerini oluştur
brain_tumor_model = BrainTumorDetector()
alzheimer_model = AlzheimerDetector()

@app.post("/analyze/brain-tumor")
async def analyze_brain_tumor(file: UploadFile = File(...)):
    # Geçici dosya oluştur
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp_path = temp.name
    
    try:
        # Analiz yap
        result = brain_tumor_model.predict(temp_path)
        
        # Geçici dosyayı sil
        os.unlink(temp_path)
        
        return result
    except Exception as e:
        # Hata durumunda geçici dosyayı sil
        os.unlink(temp_path)
        return {"error": str(e)}

@app.post("/analyze/alzheimer")
async def analyze_alzheimer(file: UploadFile = File(...)):
    # Geçici dosya oluştur
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp_path = temp.name
    
    try:
        # Analiz yap
        result = alzheimer_model.predict(temp_path)
        
        # Geçici dosyayı sil
        os.unlink(temp_path)
        
        return result
    except Exception as e:
        # Hata durumunda geçici dosyayı sil
        os.unlink(temp_path)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000) 