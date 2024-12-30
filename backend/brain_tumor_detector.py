import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
import base64

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_heatmap(self, input_tensor, target_class):
        # Gradyan hesaplaması için requires_grad'ı aktif et
        input_tensor.requires_grad = True
        
        # Modelin çıktısını al
        output = self.model(input_tensor)
        
        # Hedef sınıf için gradyanları hesapla
        if output.requires_grad:
            self.model.zero_grad()
            target = torch.zeros_like(output)
            target[0][target_class] = 1
            output.backward(gradient=target)
            
            # Gradyan ve aktivasyon haritalarını al
            gradients = self.gradients.detach().cpu()
            activations = self.activations.detach().cpu()
            
            # Global Average Pooling uygula
            weights = torch.mean(gradients, dim=(2, 3))
            
            # Ağırlıklı aktivasyon haritası oluştur
            cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
            for i, w in enumerate(weights[0]):
                cam += w * activations[0, i, :, :]
            
            # ReLU uygula ve normalize et
            cam = torch.maximum(cam, torch.tensor(0))
            
            # Min-max normalizasyonu
            if cam.max() > 0:
                cam = cam - cam.min()
                cam = cam / cam.max()
                
                # Eşik değeri uygula (sadece yüksek aktivasyonları göster)
                threshold = 0.5  # Eşik değerini artırdık
                cam[cam < threshold] = 0
                
                # Yumuşatma için Gaussian blur uygula
                cam = cam.numpy()
                cam = cv2.GaussianBlur(cam, (5, 5), 0)
                
                return cam
            else:
                print("Aktivasyon haritası oluşturulamadı")
                return None
        else:
            print("Hata: Çıktı tensörü gradyan hesaplaması için ayarlanmamış")
            return None

class BrainTumorDetector:
    def __init__(self):
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        # GradCAM için son konvolüsyon katmanını hedefle
        self.target_layer = self.model.layer4[-1]
        self.grad_cam = GradCAM(self.model, self.target_layer)
        
        # Görüntü dönüşüm işlemleri tanımla
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        # Model ağırlıklarını yükle
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'brain_tumor_best.pth')
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
        
    def generate_heatmap(self, image_tensor, predicted_class):
        # Isı haritası oluştur
        heatmap = self.grad_cam.generate_heatmap(image_tensor, predicted_class)
        
        # Isı haritasını yeniden boyutlandır ve renklendir
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap
    
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
            self.model.zero_grad()  # Önceki gradyanları temizle
            with torch.set_grad_enabled(True):  # Gradyan hesaplamasını aktif et
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
                # Ham çıktıları yazdır
                print(f"Model çıktıları: {outputs[0]}")
                print(f"Olasılıklar: {probabilities}")
                
                # En yüksek olasılıklı sınıfı bul
                max_prob, predicted_class = torch.max(probabilities, dim=0)
                
                # Tümör var mı kontrol et
                threshold = 0.3
                has_tumor = predicted_class.item() == 1 and max_prob.item() > threshold
                
                # Isı haritası oluştur
                heatmap = None
                overlayed = None
                if has_tumor:
                    heatmap = self.grad_cam.generate_heatmap(image_tensor, predicted_class.item())
                    if heatmap is not None:
                        try:
                            # Orijinal görüntüyü yeniden boyutlandır
                            original_resized = cv2.resize(enhanced, (224, 224))
                            
                            # Isı haritasını hazırla
                            heatmap = cv2.resize(heatmap, (224, 224))  # Boyutu eşitle
                            
                            # Morfolojik işlemler uygula (gürültüyü azalt ve bağlı bölgeleri birleştir)
                            kernel = np.ones((5,5), np.uint8)
                            heatmap = cv2.morphologyEx(heatmap, cv2.MORPH_CLOSE, kernel)
                            heatmap = cv2.morphologyEx(heatmap, cv2.MORPH_OPEN, kernel)
                            
                            # Eşik değeri uygula
                            _, heatmap = cv2.threshold(heatmap, 0.5, 1.0, cv2.THRESH_BINARY)
                            
                            heatmap_uint8 = np.uint8(255 * heatmap)
                            
                            # Isı haritasını 3 kanallı hale getir ve renklendir
                            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                            
                            # Görüntüleri BGR formatına dönüştür
                            if len(original_resized.shape) == 3 and original_resized.shape[2] == 3:
                                original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
                            else:
                                original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
                            
                            # Debug bilgisi
                            print(f"Original shape: {original_bgr.shape}")
                            print(f"Heatmap shape: {heatmap_colored.shape}")
                            
                            # Görüntüleri birleştir
                            overlayed = cv2.addWeighted(original_bgr, 0.7, heatmap_colored, 0.3, 0)
                            
                            # Sonuçları RGB'ye geri dönüştür
                            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                            overlayed = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
                            
                            # Görüntüleri base64'e dönüştür
                            _, heatmap_buffer = cv2.imencode('.png', heatmap_colored)
                            _, overlay_buffer = cv2.imencode('.png', overlayed)
                            
                            heatmap_base64 = base64.b64encode(heatmap_buffer).decode('utf-8')
                            overlay_base64 = base64.b64encode(overlay_buffer).decode('utf-8')
                            
                        except Exception as e:
                            print(f"Isı haritası oluşturma hatası: {str(e)}")
                            return {
                                "success": False,
                                "error": f"Isı haritası oluşturma hatası: {str(e)}"
                            }
                
                # Tüm sınıfların olasılıklarını al
                class_probs = {
                    "no_tumor": probabilities[0].item(),
                    "tumor": probabilities[1].item()
                }
                
                print(f"Tahmin sonucu: Sınıf={predicted_class.item()}, Olasılık={max_prob.item():.4f}")
            
            result = {
                "has_tumor": has_tumor,
                "confidence": max_prob.item(),
                "all_probabilities": class_probs,
                "predicted_class": predicted_class.item(),
                "threshold": threshold,
                "success": True
            }
            
            if has_tumor and heatmap is not None:
                result["heatmap"] = heatmap_base64
                result["overlay"] = overlay_base64
            
            return result
            
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 