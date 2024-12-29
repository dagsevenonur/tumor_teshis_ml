import os
import requests
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __init__(self, image_paths, class_names):
        self.image_paths = image_paths
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, image_path

def test_tumor_model():
    print("\nBeyin Tümörü Modeli Test Ediliyor...")
    yes_path = Path("test/tumor/yes")
    no_path = Path("test/tumor/no")
    
    total_images = 0
    correct_predictions = 0
    
    results = []
    
    # CUDA kullanılabilirliğini kontrol et
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Tümörlü görüntüleri test et
    for img_path in tqdm(list(yes_path.glob("*")), desc="Tümörlü görüntüler test ediliyor"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            total_images += 1
            with open(img_path, 'rb') as img:
                files = {'file': (img_path.name, img, 'image/jpeg')}
                try:
                    response = requests.post('http://localhost:8000/analyze/tumor', files=files)
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('has_tumor', False):
                            correct_predictions += 1
                            prediction_status = "Doğru"
                        else:
                            prediction_status = "Yanlış"
                        
                        results.append(f"Dosya: {img_path.name}, Gerçek: Tümörlü, Tahmin: {'Tümörlü' if result.get('has_tumor', False) else 'Normal'}, Durum: {prediction_status}")
                    else:
                        results.append(f"Hata: {img_path.name} analiz edilemedi. Status: {response.status_code}")
                except Exception as e:
                    results.append(f"Hata: {img_path.name} analiz edilemedi. Hata: {str(e)}")
    
    # Tümörsüz görüntüleri test et
    for img_path in tqdm(list(no_path.glob("*")), desc="Normal görüntüler test ediliyor"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            total_images += 1
            with open(img_path, 'rb') as img:
                files = {'file': (img_path.name, img, 'image/jpeg')}
                try:
                    response = requests.post('http://localhost:8000/analyze/tumor', files=files)
                    if response.status_code == 200:
                        result = response.json()
                        if not result.get('has_tumor', True):
                            correct_predictions += 1
                            prediction_status = "Doğru"
                        else:
                            prediction_status = "Yanlış"
                        
                        results.append(f"Dosya: {img_path.name}, Gerçek: Normal, Tahmin: {'Tümörlü' if result.get('has_tumor', False) else 'Normal'}, Durum: {prediction_status}")
                    else:
                        results.append(f"Hata: {img_path.name} analiz edilemedi. Status: {response.status_code}")
                except Exception as e:
                    results.append(f"Hata: {img_path.name} analiz edilemedi. Hata: {str(e)}")
    
    accuracy = (correct_predictions / total_images * 100) if total_images > 0 else 0
    
    return {
        'model_name': 'Beyin Tümörü Modeli',
        'total_images': total_images,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'results': results
    }

def test_alzheimer_model():
    print("\nAlzheimer Modeli Test Ediliyor...")
    # Test klasöründeki sınıf isimleri
    classes = ['non_demented', 'very_mild_demented', 'mild_demented', 'moderate_demented']
    class_mapping = {
        'non_demented': 'Normal',
        'very_mild_demented': 'Çok Hafif',
        'mild_demented': 'Hafif',
        'moderate_demented': 'Orta'
    }
    
    # CUDA kullanılabilirliğini kontrol et
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Toplam görüntü sayısını hesapla
    total_images = 0
    all_images = []
    for class_name in classes:
        class_path = Path(f"test/alzheimer/{class_name}")
        if class_path.exists():
            images = list(class_path.glob("*"))
            total_images += len(images)
            all_images.extend([(img, class_name) for img in images])
    
    print(f"Toplam {total_images} görüntü test edilecek...")
    correct_predictions = 0
    results = []
    
    # Batch processing için veri yükleyici oluştur
    batch_size = 32  # GPU belleğine göre ayarlayın
    
    # Progress bar oluştur
    progress_bar = tqdm(total=total_images, desc="Alzheimer Testi")
    
    # Görüntüleri batch'ler halinde işle
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i + batch_size]
        
        for img_path, class_name in batch:
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                with open(img_path, 'rb') as img:
                    files = {'file': (img_path.name, img, 'image/jpeg')}
                    try:
                        response = requests.post('http://localhost:8000/analyze/alzheimer', files=files, timeout=30)
                        if response.status_code == 200:
                            result = response.json()
                            predicted_class = result.get('prediction', '')
                            expected_class = class_mapping.get(class_name)
                            
                            if expected_class == predicted_class:
                                correct_predictions += 1
                                prediction_status = "Doğru"
                            else:
                                prediction_status = "Yanlış"
                            
                            results.append(f"Dosya: {img_path.name}, Gerçek: {expected_class}, Tahmin: {predicted_class}, Durum: {prediction_status}, Güven: {result.get('confidence', 0):.2%}")
                        else:
                            results.append(f"Hata: {img_path.name} analiz edilemedi. Status: {response.status_code}")
                    except Exception as e:
                        results.append(f"Hata: {img_path.name} analiz edilemedi. Hata: {str(e)}")
                progress_bar.update(1)
    
    progress_bar.close()
    accuracy = (correct_predictions / total_images * 100) if total_images > 0 else 0
    
    return {
        'model_name': 'Alzheimer Modeli',
        'total_images': total_images,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'results': results
    }

def save_results(tumor_results, alzheimer_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test/test_results_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== MODEL TEST SONUÇLARI ===\n")
        f.write(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Tümör Modeli Sonuçları
        f.write(f"=== {tumor_results['model_name']} ===\n")
        f.write(f"Toplam Test Görüntüsü: {tumor_results['total_images']}\n")
        f.write(f"Doğru Tahmin Sayısı: {tumor_results['correct_predictions']}\n")
        f.write(f"Doğruluk Oranı: %{tumor_results['accuracy']:.2f}\n\n")
        f.write("Detaylı Sonuçlar:\n")
        for result in tumor_results['results']:
            f.write(f"{result}\n")
        f.write("\n")
        
        # Alzheimer Modeli Sonuçları
        f.write(f"=== {alzheimer_results['model_name']} ===\n")
        f.write(f"Toplam Test Görüntüsü: {alzheimer_results['total_images']}\n")
        f.write(f"Doğru Tahmin Sayısı: {alzheimer_results['correct_predictions']}\n")
        f.write(f"Doğruluk Oranı: %{alzheimer_results['accuracy']:.2f}\n\n")
        f.write("Detaylı Sonuçlar:\n")
        for result in alzheimer_results['results']:
            f.write(f"{result}\n")
    
    print(f"\nTest sonuçları '{filename}' dosyasına kaydedildi.")

def main():
    print("Model Test Süreci Başlatılıyor...")
    
    # Backend'in çalışır durumda olduğunu kontrol et
    try:
        requests.get('http://localhost:8000')
    except:
        print("Hata: Backend sunucusuna bağlanılamadı. Lütfen sunucunun çalıştığından emin olun.")
        return
    
    # CUDA kullanılabilirliğini kontrol et
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Modelleri test et
    tumor_results = test_tumor_model()
    time.sleep(1)  # Ardışık istekler arasında küçük bir bekleme
    alzheimer_results = test_alzheimer_model()
    
    # Sonuçları kaydet
    save_results(tumor_results, alzheimer_results)
    
    # Özet sonuçları ekrana yazdır
    print("\n=== TEST SONUÇLARI ===")
    print(f"\n{tumor_results['model_name']}:")
    print(f"Toplam Test: {tumor_results['total_images']}")
    print(f"Doğru Tahmin: {tumor_results['correct_predictions']}")
    print(f"Doğruluk Oranı: %{tumor_results['accuracy']:.2f}")
    
    print(f"\n{alzheimer_results['model_name']}:")
    print(f"Toplam Test: {alzheimer_results['total_images']}")
    print(f"Doğru Tahmin: {alzheimer_results['correct_predictions']}")
    print(f"Doğruluk Oranı: %{alzheimer_results['accuracy']:.2f}")

if __name__ == "__main__":
    main() 