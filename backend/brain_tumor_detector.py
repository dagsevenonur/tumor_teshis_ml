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
                threshold = 0.7  # Eşik değerini daha da artırdık
                cam[cam < threshold] = 0
                
                # Yumuşatma için Gaussian blur uygula (daha büyük kernel ve sigma)
                cam = cam.numpy()
                cam = cv2.GaussianBlur(cam, (7, 7), 1.5)
                
                return cam
            else:
                print("Aktivasyon haritası oluşturulamadı")
                return None
        else:
            print("Hata: Çıktı tensörü gradyan hesaplaması için ayarlanmamış")
            return None

    def create_custom_colormap(self, heatmap):
        # Normalize edilmiş heatmap (0-1 arası)
        heatmap_norm = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        
        # 3 kanallı boş görüntü oluştur
        colored = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.float32)
        
        # Tümör bölgesi için mavi tonları (yüksek aktivasyon)
        tumor_mask = heatmap_norm > 0.7  # Eşik değerini artırdık
        non_tumor_mask = heatmap_norm <= 0.7
        
        # Tümör bölgesi (mavi tonları)
        colored[tumor_mask, 0] = 0  # R
        colored[tumor_mask, 1] = 0  # G
        colored[tumor_mask, 2] = 1  # B (sabit mavi)
        
        # Tümör olmayan bölge (kırmızı tonları)
        colored[non_tumor_mask, 0] = 0.8  # R (sabit kırmızı)
        colored[non_tumor_mask, 1] = 0  # G
        colored[non_tumor_mask, 2] = 0  # B
        
        # Yumuşak geçişler için Gaussian blur uygula
        colored = cv2.GaussianBlur(colored, (3, 3), 0.5)
        
        # Opaklık maskesi oluştur
        opacity = np.zeros((heatmap.shape[0], heatmap.shape[1]), dtype=np.float32)
        opacity[tumor_mask] = 0.8  # Tümör bölgesi daha belirgin
        opacity[non_tumor_mask] = 0.3  # Diğer bölgeler daha saydam
        
        # Opaklık maskesini uygula
        colored = colored * opacity[:, :, np.newaxis]
        
        return np.uint8(colored * 255)

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
        
    def create_custom_colormap(self, heatmap):
        # Normalize edilmiş heatmap (0-1 arası)
        heatmap_norm = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        
        # 3 kanallı boş görüntü oluştur
        colored = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.float32)
        
        # Tümör bölgesi için mavi tonları (yüksek aktivasyon)
        tumor_mask = heatmap_norm > 0.7  # Eşik değerini artırdık
        non_tumor_mask = heatmap_norm <= 0.7
        
        # Tümör bölgesi (mavi tonları)
        colored[tumor_mask, 0] = 0  # R
        colored[tumor_mask, 1] = 0  # G
        colored[tumor_mask, 2] = 1  # B (sabit mavi)
        
        # Tümör olmayan bölge (kırmızı tonları)
        colored[non_tumor_mask, 0] = 0.8  # R (sabit kırmızı)
        colored[non_tumor_mask, 1] = 0  # G
        colored[non_tumor_mask, 2] = 0  # B
        
        # Yumuşak geçişler için Gaussian blur uygula
        colored = cv2.GaussianBlur(colored, (3, 3), 0.5)
        
        # Opaklık maskesi oluştur
        opacity = np.zeros((heatmap.shape[0], heatmap.shape[1]), dtype=np.float32)
        opacity[tumor_mask] = 0.8  # Tümör bölgesi daha belirgin
        opacity[non_tumor_mask] = 0.3  # Diğer bölgeler daha saydam
        
        # Opaklık maskesini uygula
        colored = colored * opacity[:, :, np.newaxis]
        
        return np.uint8(colored * 255)
        
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
                
                # Isı haritası ve bounding box oluştur
                heatmap = None
                overlayed = None
                bbox_image = None
                if has_tumor:
                    # Isı haritası oluştur
                    heatmap = self.grad_cam.generate_heatmap(image_tensor, predicted_class.item())
                    if heatmap is not None:
                        try:
                            # Orijinal görüntüyü yeniden boyutlandır
                            original_resized = cv2.resize(enhanced, (224, 224))
                            
                            # Isı haritasını hazırla
                            heatmap = cv2.resize(heatmap, (224, 224))
                            
                            # Morfolojik işlemler uygula
                            kernel = np.ones((3,3), np.uint8)
                            heatmap = cv2.morphologyEx(heatmap, cv2.MORPH_OPEN, kernel)
                            heatmap = cv2.morphologyEx(heatmap, cv2.MORPH_CLOSE, kernel)
                            
                            # Küçük bölgeleri temizle
                            contours, _ = cv2.findContours(np.uint8(heatmap * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            min_area = 100  # Minimum alanı artırdık
                            for cnt in contours:
                                if cv2.contourArea(cnt) < min_area:
                                    cv2.drawContours(heatmap, [cnt], -1, 0, -1)
                            
                            # Özel renk haritası uygula
                            heatmap_colored = self.create_custom_colormap(heatmap)
                            
                            # Görüntüleri BGR formatına dönüştür
                            if len(original_resized.shape) == 3 and original_resized.shape[2] == 3:
                                original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
                            else:
                                original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
                            
                            # Görüntüleri birleştir
                            overlayed = cv2.addWeighted(original_bgr, 0.7, cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR), 0.3, 0)
                            
                            # Bounding box için görüntü hazırla
                            bbox_image = original_bgr.copy()
                            
                            # Isı haritasını normalize et ve binary mask oluştur
                            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            _, binary = cv2.threshold(heatmap_norm, int(255 * 0.7), 255, cv2.THRESH_BINARY)
                            
                            # Gürültüyü temizle
                            kernel = np.ones((5,5), np.uint8)
                            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                            
                            # Konturları bul
                            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                # En büyük konturu bul
                                largest_contour = max(contours, key=cv2.contourArea)
                                
                                # Kontur alanını hesapla
                                area = cv2.contourArea(largest_contour)
                                
                                # Minimum alan kontrolü
                                if area > 100:  # Minimum alan eşiği
                                    # Konturun sınırlayıcı dikdörtgenini bul
                                    x, y, w, h = cv2.boundingRect(largest_contour)
                                    
                                    # Box'ı sola doğru küçült
                                    x = x + int(w * 0.1)  # X koordinatını sağa kaydır
                                    w = int(w * 0.8)      # Genişliği azalt
                                    
                                    # Bounding box çiz (ince çizgi)
                                    cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    
                                    # Etiket metni
                                    label = f"Tumor"
                                    
                                    # Etiket arka planı için boyut hesapla
                                    (label_width, label_height), _ = cv2.getTextSize(
                                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                                    )
                                    
                                    # Etiket konumunu ayarla
                                    label_x = x
                                    label_y = y - 10 if y - 10 > label_height else y + h + label_height + 10
                                    
                                    # Etiket arka planı
                                    cv2.rectangle(
                                        bbox_image,
                                        (label_x, label_y - label_height - 5),
                                        (label_x + label_width + 5, label_y + 5),
                                        (0, 255, 0),
                                        -1
                                    )
                                    
                                    # Etiket metni (daha ince ve okunaklı)
                                    cv2.putText(
                                        bbox_image,
                                        label,
                                        (label_x + 3, label_y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 0, 0),
                                        1
                                    )
                            
                            # Sonuçları RGB'ye geri dönüştür
                            overlayed = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
                            bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)
                            
                            # Görüntüleri base64'e dönüştür
                            _, heatmap_buffer = cv2.imencode('.png', heatmap_colored)
                            _, overlay_buffer = cv2.imencode('.png', overlayed)
                            _, bbox_buffer = cv2.imencode('.png', bbox_image)
                            
                            heatmap_base64 = base64.b64encode(heatmap_buffer).decode('utf-8')
                            overlay_base64 = base64.b64encode(overlay_buffer).decode('utf-8')
                            bbox_base64 = base64.b64encode(bbox_buffer).decode('utf-8')
                            
                        except Exception as e:
                            print(f"Görselleştirme hatası: {str(e)}")
                            return {
                                "success": False,
                                "error": f"Görselleştirme hatası: {str(e)}"
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
                result["bbox_image"] = bbox_base64
            
            return result
            
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 