import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import os

class BrainTumorDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ResNet50 modelini yükle
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # 2 sınıf: tümör var/yok
        self.model = self.model.to(self.device)
        
        # Model ağırlıklarını yükle
        try:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'brain_tumor_best.pth')
            print(f"Model yolu: {model_path}")
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Model ağırlıkları başarıyla yüklendi")
            else:
                print(f"Model dosyası bulunamadı: {model_path}")
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {str(e)}")
            print("Model varsayılan ağırlıklarla başlatıldı")
        
        self.model.eval()
        
        # Görüntü ön işleme
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, image):
        # Görüntüyü PIL Image formatına çevir
        if isinstance(image, np.ndarray):
            # OpenCV BGR'dan RGB'ye dönüşüm
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Tek kanallı görüntüyü RGB'ye dönüştür
            elif len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Kontrast iyileştirme
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Gürültü azaltma
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            image = Image.fromarray(image)
        
        return image
        
    def predict(self, image):
        try:
            # Görüntü ön işleme
            image = self.preprocess_image(image)
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Tahmin yap
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
                # Ham çıktıları yazdır
                print(f"Model çıktıları: {outputs[0]}")
                print(f"Olasılıklar: {probabilities}")
                
                # Tümör sınıfının olasılığı
                tumor_prob = probabilities[1].item()
                
                # Eşik değeri uygula
                threshold = 0.3
                has_tumor = tumor_prob > threshold
                
                detection = {
                    "confidence": tumor_prob,
                    "tumor_detected": has_tumor,
                    "location": {
                        "x_center": 112,  # Görüntü merkezi (224/2)
                        "y_center": 112,
                        "width": 50,      # Varsayılan boyut
                        "height": 50
                    }
                }
                
                print(f"Tahmin sonucu: Tümör={has_tumor}, Olasılık={tumor_prob:.4f}")
                
                return [detection] if has_tumor else []
                
        except Exception as e:
            print(f"Tahmin sırasında hata: {str(e)}")
            return [] 