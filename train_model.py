import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm

class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def enhance_image(self, image):
        # Görüntüyü numpy dizisine çevir
        image_array = np.array(image)
        
        # Görüntü BGR ise RGB'ye çevir
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
        
        return Image.fromarray(denoised)
    
    def __getitem__(self, idx):
        try:
            # Görüntüyü yükle
            image_path = self.image_paths[idx]
            
            # CSV dosyası kontrolü
            if image_path.endswith('.csv'):
                # CSV'den görüntü matrisini oku
                image_array = pd.read_csv(image_path).values
                image_array = (image_array * 255).astype(np.uint8)  # Normalize et
                image = Image.fromarray(image_array).convert('RGB')
            else:
                # Normal görüntü dosyasını oku
                image = Image.open(image_path).convert('RGB')
            
            # Görüntü iyileştirme
            image = self.enhance_image(image)
            
            # Transform uygula
            if self.transform:
                image = self.transform(image)
                
            return image, self.labels[idx]
        except Exception as e:
            print(f"Hata: {image_path} dosyası işlenirken hata oluştu - {str(e)}")
            # Hata durumunda siyah görüntü döndür
            if self.transform:
                return torch.zeros((3, 224, 224)), self.labels[idx]
            return Image.new('RGB', (224, 224), 'black'), self.labels[idx]

def prepare_data(data_dir):
    # Veri yollarını ve etiketleri topla
    image_paths = []
    labels = []
    
    # Desteklenen formatlar
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.csv'}
    
    # Tümörlü görüntüler
    tumor_dir = os.path.join(data_dir, 'yes')
    for img_name in os.listdir(tumor_dir):
        if Path(img_name).suffix.lower() in valid_extensions:
            image_paths.append(os.path.join(tumor_dir, img_name))
            labels.append(1)  # 1: tümör var
        
    # Tümörsüz görüntüler
    no_tumor_dir = os.path.join(data_dir, 'no')
    for img_name in os.listdir(no_tumor_dir):
        if Path(img_name).suffix.lower() in valid_extensions:
            image_paths.append(os.path.join(no_tumor_dir, img_name))
            labels.append(0)  # 0: tümör yok
    
    print(f"Toplam görüntü sayısı: {len(image_paths)}")
    print(f"Tümörlü görüntü sayısı: {labels.count(1)}")
    print(f"Tümörsüz görüntü sayısı: {labels.count(0)}")
    
    return train_test_split(image_paths, labels, test_size=0.2, random_state=42)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEğitim device: {device}")
    if device.type == 'cuda':
        print(f"GPU modeli: {torch.cuda.get_device_name(0)}")
        print(f"Kullanılabilir GPU sayısı: {torch.cuda.device_count()}")
    
    model = model.to(device)
    best_val_acc = 0.0
    patience = 5  # Erken durdurma için sabır
    no_improve = 0  # İyileşme olmayan epoch sayısı
    
    print("\nEğitim başlıyor...")
    print(f"Toplam epoch: {num_epochs}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Eğitim seti büyüklüğü: {len(train_loader.dataset)}")
    print(f"Doğrulama seti büyüklüğü: {len(val_loader.dataset)}\n")
    
    total_steps = len(train_loader) * num_epochs
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Eğitim
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # tqdm ile ilerleme çubuğu
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Eğitim]', 
                         leave=True, position=0)
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # İlerleme çubuğunu güncelle
            global_step += 1
            train_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'loss': f'{train_loss/len(train_loader):.4f}',
                'acc': f'{train_acc:.2f}%',
                'genel ilerleme': f'{global_step/total_steps*100:.1f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # Doğrulama
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # tqdm ile doğrulama ilerleme çubuğu
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Doğrulama]',
                       leave=True, position=0)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # İlerleme çubuğunu güncelle
                val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'loss': f'{val_loss/len(val_loader):.4f}',
                    'acc': f'{val_acc:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        epoch_time = time.time() - epoch_start_time
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - {epoch_time:.1f} saniye')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, 'models/brain_tumor_best.pth')
            print(f'En iyi model kaydedildi! Doğrulama doğruluğu: {val_acc:.2f}%')
            no_improve = 0
        else:
            no_improve += 1
        
        # Erken durdurma kontrolü
        if no_improve >= patience:
            print(f'\nSon {patience} epoch boyunca iyileşme olmadı. Eğitim durduruluyor...')
            break
        
        print('-' * 80)

def main():
    # Veri dönüşümleri (daha güçlü veri artırma)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test için daha basit dönüşümler
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Veriyi hazırla
    train_paths, val_paths, train_labels, val_labels = prepare_data('datasets/brain_tumor')
    
    # Veri setlerini oluştur
    train_dataset = BrainTumorDataset(train_paths, train_labels, transform)
    val_dataset = BrainTumorDataset(val_paths, val_labels, val_transform)
    
    # Veri yükleyicileri
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4, pin_memory=True)
    
    # Model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Kayıp fonksiyonu ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # L2 regularization ekledik
    
    # Modeli eğit
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)

if __name__ == '__main__':
    main() 