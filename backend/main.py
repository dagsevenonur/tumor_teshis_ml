from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
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

# Dedektör nesnelerini oluştur
tumor_detector = BrainTumorDetector()
alzheimer_detector = AlzheimerDetector()

@app.post("/analyze/tumor")
async def analyze_tumor(file: UploadFile = File(...)):
    temp_file = None
    try:
        # Geçici dosya oluştur
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        content = await file.read()
        temp_file.write(content)
        temp_file.close()  # Dosyayı kapat
        
        # Analiz yap
        result = tumor_detector.detect(temp_file.name)
        
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Geçici dosyayı temizle
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"Geçici dosya silinirken hata oluştu: {str(e)}")

@app.post("/analyze/alzheimer")
async def analyze_alzheimer(file: UploadFile = File(...)):
    temp_file = None
    try:
        # Geçici dosya oluştur
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        content = await file.read()
        temp_file.write(content)
        temp_file.close()  # Dosyayı kapat
        
        # Analiz yap
        result = alzheimer_detector.detect(temp_file.name)
        
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Geçici dosyayı temizle
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"Geçici dosya silinirken hata oluştu: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 