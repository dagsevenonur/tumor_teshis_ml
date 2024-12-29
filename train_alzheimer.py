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
from pathlib import Path
import time
from tqdm import tqdm

# CUDA ayarları
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class AlzheimerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def enhance_image(self, image):
        # Görüntü iyileştirme
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            # RGB görüntüyü LAB'a dönüştür
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE uygula
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Kanalları birleştir
            enhanced_lab = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Gürültü azaltma
            denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            return Image.fromarray(denoised)
        return image
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image = self.enhance_image(image)
            
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Hata: {image_path} dosyası işlenirken hata oluştu - {str(e)}")
            if self.transform:
                return torch.zeros((3, 224, 224)), self.labels[idx]
            return Image.new('RGB', (224, 224), 'black'), self.labels[idx]

def prepare_data(data_dir):
    classes = ['non_demented', 'very_mild_demented', 'mild_demented', 'moderate_demented']
    image_paths = []
    labels = []
    
    for idx, class_name in enumerate(classes):
        class_dir = Path(data_dir) / class_name
        if class_dir.exists():
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_paths.append(str(img_path))
                    labels.append(idx)
    
    # Verileri karıştır ve böl
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return train_paths, val_paths, train_labels, val_labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, model_save_path='models/alzheimer_best.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEğitim device: {device}")
    
    model = model.to(device)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_acc = 0.0
    patience = 5
    no_improve = 0
    
    # Models klasörünü oluştur
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print("\nEğitim başlıyor...")
    print(f"Toplam epoch: {num_epochs}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Eğitim seti büyüklüğü: {len(train_loader.dataset)}")
    print(f"Doğrulama seti büyüklüğü: {len(val_loader.dataset)}\n")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Eğitim
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Eğitim]')
        
        for images, labels in train_pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'loss': f'{train_loss/len(train_loader):.4f}',
                'acc': f'{train_acc:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # Doğrulama
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Doğrulama]')
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
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
        
        # Her epoch sonunda modeli kaydet
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, f'models/alzheimer_epoch_{epoch+1}.pth')
        print(f'Model kaydedildi: models/alzheimer_epoch_{epoch+1}.pth')
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, model_save_path)
            print(f'En iyi model kaydedildi! Doğrulama doğruluğu: {val_acc:.2f}%')
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f'\nSon {patience} epoch boyunca iyileşme olmadı. Eğitim durduruluyor...')
            break
        
        print('-' * 80)
    
    # Son modeli kaydet
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
    }, 'models/alzheimer_final.pth')
    print('\nSon model kaydedildi: models/alzheimer_final.pth')

def main():
    # Veri dönüşümleri
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Veri yolları
    data_dir = 'datasets/alzheimer_processed/train'
    train_paths, val_paths, train_labels, val_labels = prepare_data(data_dir)
    
    # Veri yükleyiciler
    train_dataset = AlzheimerDataset(train_paths, train_labels, transform)
    val_dataset = AlzheimerDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=8, pin_memory=True)
    
    # Model oluştur
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    
    # Modeli özelleştir
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 4)  # 4 sınıf
    )
    
    # Kayıp fonksiyonu ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Eğitim
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)

if __name__ == '__main__':
    main() 