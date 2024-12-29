import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

class AlzheimerDetector:
    def __init__(self, model_path='../models/alzheimer_best.pth'):
        # Sınıf isimleri - Test verisiyle aynı sırada olmalı
        self.class_names = ['non_demented', 'very_mild_demented', 'mild_demented', 'moderate_demented']
        self.display_names = {
            'non_demented': 'Normal',
            'very_mild_demented': 'Çok Hafif',
            'mild_demented': 'Hafif',
            'moderate_demented': 'Orta'
        }
        
        # Model mimarisini oluştur
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Cihaz: {self.device}")
        
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        
        # Eğitim sırasında kullanılan mimari ile aynı yapıyı oluştur
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(self.class_names))
        )
        self.model = self.model.to(self.device)
        
        try:
            print(f"Model yükleniyor: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
            
            # Model ağırlıklarını güvenli şekilde yükle
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model başarıyla yüklendi")
            print(f"Eğitim doğruluğu: {checkpoint.get('train_acc', 0):.2%}")
            print(f"Doğrulama doğruluğu: {checkpoint.get('val_acc', 0):.2%}")
            
            self.model.eval()
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            raise Exception(f"Model yüklenemedi: {str(e)}")
        
        # Görüntü ön işleme
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        try:
            # Görüntüyü yükle ve ön işle
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")
            
            print(f"Görüntü analiz ediliyor: {image_path}")
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Tahmin yap
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_prob, predicted_idx = torch.max(probabilities, 1)
            
            # Sonuçları hazırla
            predicted_class = predicted_idx.item()
            confidence = predicted_prob.item()
            
            # Ham çıktıları yazdır
            print(f"Model çıktıları: {outputs[0]}")
            print(f"Olasılıklar: {probabilities[0]}")
            print(f"Tahmin: {self.display_names[self.class_names[predicted_class]]} ({confidence:.2%})")
            
            result = {
                'predicted_class': predicted_class,
                'prediction': self.display_names[self.class_names[predicted_class]],
                'confidence': confidence,
                'all_probabilities': {
                    self.display_names[class_name]: prob.item()
                    for class_name, prob in zip(self.class_names, probabilities[0])
                }
            }
            
            return result
        except Exception as e:
            print(f"Tahmin hatası: {str(e)}")
            raise Exception(f"Görüntü analizi başarısız: {str(e)}") 