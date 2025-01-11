import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

class AlzheimerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # EfficientNet-B4 backbone
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True)
        n_features = self.backbone.classifier.in_features
        
        # Backbone'un son katmanını kaldır
        self.backbone.classifier = nn.Identity()
        
        # Attention mekanizması
        self.attention = nn.Sequential(
            nn.Linear(n_features, n_features // 16),
            nn.ReLU(),
            nn.Linear(n_features // 16, n_features),
            nn.Sigmoid()
        )
        
        # Sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 4)  # 4 sınıf
        )
        
    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        
        # Attention
        att_weights = self.attention(features)
        features = features * att_weights
        
        # Classification
        out = self.classifier(features)
        return out

class AlzheimerDetector:
    def __init__(self):
        self.model = AlzheimerNet()
        
        # Görüntü dönüşüm işlemleri
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        # Model ağırlıklarını yükle
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'alzheimer_best.pth')
        print(f"Model yolu: {model_path}")
        try:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model başarıyla yüklendi")
            print(f"Eğitim doğruluğu: {checkpoint['train_acc']:.2f}%")
            print(f"Doğrulama doğruluğu: {checkpoint['val_acc']:.2f}%")
        except Exception as e:
            print(f"Model yüklenirken hata: {str(e)}")
        
        self.model.eval()

    def enhance_image(self, image_array):
        # RGB'ye dönüştür
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        elif len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        # CLAHE uygula (daha agresif)
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Bilateral filtering (edge-preserving noise reduction)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Contrast stretching
        p2, p98 = np.percentile(denoised, (2, 98))
        stretched = np.clip((denoised - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        
        # Unsharp masking for edge enhancement
        gaussian = cv2.GaussianBlur(stretched, (0, 0), 2.0)
        enhanced_final = cv2.addWeighted(stretched, 1.5, gaussian, -0.5, 0)
        
        return enhanced_final

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
            
            # CUDA kullanılabilirliğini kontrol et
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image_tensor = image_tensor.to(device)
            self.model = self.model.to(device)
            
            # Tahmin yap
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
                # Ham çıktıları yazdır
                print(f"Model çıktıları: {outputs[0]}")
                print(f"Olasılıklar: {probabilities}")
                
                # En yüksek olasılıklı sınıfı bul
                max_prob, predicted_class = torch.max(probabilities, dim=0)
                
                # Sınıf etiketlerini tanımla
                class_labels = ['Normal', 
                              'Çok Hafif', 
                              'Hafif', 
                              'Orta']
                prediction = class_labels[predicted_class.item()]
                
                # Tüm sınıfların olasılıklarını al
                class_probs = {
                    'Normal': probabilities[0].item(),
                    'Çok Hafif': probabilities[1].item(),
                    'Hafif': probabilities[2].item(),
                    'Orta': probabilities[3].item()
                }
                
                print(f"Tahmin sonucu: {prediction}, Olasılık: {max_prob.item():.4f}")
            
            return {
                "success": True,
                "prediction": prediction,
                "confidence": max_prob.item(),
                "all_probabilities": class_probs
            }
            
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 