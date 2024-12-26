import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os

class BrainTumorDetector:
    def __init__(self):
        # Model mimarisini oluştur
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        # Eğitilmiş model ağırlıklarını yükle
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'brain_tumor_best.pth')
        print(f"Model yolu: {model_path}")
        try:
            self.model.load_state_dict(torch.load(model_path))
            print("Model başarıyla yüklendi")
        except Exception as e:
            print(f"Model yüklenirken hata: {str(e)}")
        self.model.eval()
        
        # Görüntü dönüşümleri
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def enhance_image(self, image_array):
        # Görüntüyü BGR'dan RGB'ye dönüştür
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        elif len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
        # CLAHE uygula
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Gürültü azaltma
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return denoised
        
    def detect(self, image_path):
        try:
            print(f"Görüntü analiz ediliyor: {image_path}")
            
            # Görüntüyü yükle
            image = Image.open(image_path).convert('RGB')
            print(f"Görüntü boyutu: {image.size}")
            
            # Görüntü iyileştirme
            image_array = np.array(image)
            enhanced = self.enhance_image(image_array)
            enhanced_image = Image.fromarray(enhanced)
            
            # Görüntüyü modelin beklediği formata dönüştür
            image_tensor = self.transform(enhanced_image).unsqueeze(0)
            
            # Tahmin yap
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
                # Ham çıktıları yazdır
                print(f"Model çıktıları: {outputs[0]}")
                print(f"Olasılıklar: {probabilities}")
                
                # En yüksek olasılıklı sınıfı bul
                max_prob, predicted_class = torch.max(probabilities, dim=0)
                
                # Tümör var mı kontrol et
                threshold = 0.3  # Eşik değerini düşürdük
                has_tumor = predicted_class.item() == 1 and max_prob.item() > threshold
                
                # Tüm sınıfların olasılıklarını al
                class_probs = {
                    "no_tumor": probabilities[0].item(),
                    "tumor": probabilities[1].item()
                }
                
                print(f"Tahmin sonucu: Sınıf={predicted_class.item()}, Olasılık={max_prob.item():.4f}")
            
            return {
                "has_tumor": has_tumor,
                "confidence": max_prob.item(),
                "all_probabilities": class_probs,
                "predicted_class": predicted_class.item(),
                "threshold": threshold,
                "success": True
            }
            
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 