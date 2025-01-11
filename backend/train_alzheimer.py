import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from alzheimer_detector import AlzheimerNet
import time
import matplotlib.pyplot as plt

class AlzheimerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
    def enhance_image(self, image):
        # RGB'ye dönüştür
        image_array = np.array(image)
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        # CLAHE uygula
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Bilateral filtering
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Contrast stretching
        p2, p98 = np.percentile(denoised, (2, 98))
        stretched = np.clip((denoised - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        
        # Unsharp masking
        gaussian = cv2.GaussianBlur(stretched, (0, 0), 2.0)
        enhanced_final = cv2.addWeighted(stretched, 1.5, gaussian, -0.5, 0)
        
        return Image.fromarray(enhanced_final)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        # Görüntü iyileştirme
        image = self.enhance_image(image)
        
        if self.augment:
            # Veri artırma
            if np.random.random() > 0.5:
                image = transforms.RandomHorizontalFlip()(image)
            if np.random.random() > 0.5:
                image = transforms.RandomVerticalFlip()(image)
            if np.random.random() > 0.5:
                angle = np.random.uniform(-30, 30)
                image = transforms.functional.rotate(image, angle)
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                contrast = np.random.uniform(0.8, 1.2)
                saturation = np.random.uniform(0.8, 1.2)
                image = transforms.ColorJitter(brightness=brightness, 
                                            contrast=contrast,
                                            saturation=saturation)(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    start_time = time.time()
    # CUDA kontrolü
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"\nCUDA kullanılıyor: {torch.cuda.get_device_name(0)}")
        print(f"Kullanılabilir GPU Sayısı: {torch.cuda.device_count()}")
        print(f"Mevcut GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\nUYARI: CUDA bulunamadı, CPU kullanılıyor!")

    # Model kaydetme yolları
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../models')
    progress_dir = os.path.join(models_dir, 'progress')
    os.makedirs(progress_dir, exist_ok=True)

    model = model.to(device)
    
    # İlk GPU bellek kullanımını göster
    if torch.cuda.is_available():
        print(f"Başlangıç GPU Bellek Kullanımı: {torch.cuda.memory_allocated()/1024**2:.1f}MB / "
              f"{torch.cuda.memory_reserved()/1024**2:.1f}MB")
    
    best_val_acc = 0.0
    best_model_state = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("\nEğitim başlatılıyor...")
    print(f"Toplam epoch sayısı: {num_epochs}")
    print(f"Eğitim seti büyüklüğü: {len(train_loader.dataset)}")
    print(f"Doğrulama seti büyüklüğü: {len(val_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        # Eğitim aşaması
        model.train()
        running_loss = 0.0
        running_corrects = 0
        batch_times = []
        
        progress_bar = tqdm(train_loader, desc='Eğitim')
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            batch_start = time.time()
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # İstatistikler
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Batch süresi
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Progress bar güncelleme
            progress_bar.set_postfix({
                'batch': f'{batch_idx+1}/{len(train_loader)}',
                'loss': f'{loss.item():.4f}',
                'acc': f'{torch.sum(preds == labels.data).item()/len(labels):.4f}',
                'time/batch': f'{np.mean(batch_times):.3f}s'
            })
            
            # GPU bellek kullanımı kontrolünü kaldırdık
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (running_corrects.double() / len(train_loader.dataset)).cpu()
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f'\nEğitim - Kayıp: {epoch_loss:.4f} Doğruluk: {epoch_acc:.4f}')
        
        # Doğrulama aşaması
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        progress_bar = tqdm(val_loader, desc='Doğrulama')
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Progress bar güncelleme
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{torch.sum(preds == labels.data).item()/len(labels):.4f}'
                })
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = (running_corrects.double() / len(val_loader.dataset)).cpu()
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Doğrulama - Kayıp: {val_loss:.4f} Doğruluk: {val_acc:.4f}')
        
        # En iyi model durumunu güncelle
        if val_acc > best_val_acc:
            print(f'\nYeni en iyi model! Doğruluk: {val_acc:.4f} (önceki: {best_val_acc:.4f})')
            best_val_acc = val_acc
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': epoch_acc * 100,
                'val_acc': val_acc * 100,
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }
        
        # Eğitim grafiklerini çiz ve kaydet
        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot([loss.cpu().item() if torch.is_tensor(loss) else loss for loss in train_losses], label='Eğitim')
            plt.plot([loss.cpu().item() if torch.is_tensor(loss) else loss for loss in val_losses], label='Doğrulama')
            plt.title('Kayıp Değerleri')
            plt.xlabel('Epoch')
            plt.ylabel('Kayıp')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot([acc.cpu().item() if torch.is_tensor(acc) else acc for acc in train_accs], label='Eğitim')
            plt.plot([acc.cpu().item() if torch.is_tensor(acc) else acc for acc in val_accs], label='Doğrulama')
            plt.title('Doğruluk Değerleri')
            plt.xlabel('Epoch')
            plt.ylabel('Doğruluk')
            plt.legend()
            
            plt.tight_layout()
            progress_plot_path = os.path.join(progress_dir, f'training_progress_epoch_{epoch+1}.png')
            plt.savefig(progress_plot_path)
            plt.close()
    
    print("\nEğitim tamamlandı!")
    print(f"En iyi doğrulama doğruluğu: {best_val_acc:.4f}")
    print(f"Toplam eğitim süresi: {time.time() - start_time:.2f} saniye")
    
    return best_model_state

def main():
    # Veri yolu ve etiketleri hazırla
    data_dir = "../datasets/alzheimer"  # Ana veri seti klasörü
    image_paths = []
    labels = []
    
    # Veri setini yükle
    class_to_idx = {
        'NonDemented': 0,      # Normal
        'VeryMildDemented': 1,  # Hafif
        'MildDemented': 2,     # Orta
        'ModerateDemented': 3   # Şiddetli
    }
    
    # Her bir sınıf klasöründen görüntüleri yükle
    for class_name, label in class_to_idx.items():
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_dir, img_name)
                    image_paths.append(image_path)
                    labels.append(label)
    
    print(f"Toplam {len(image_paths)} görüntü bulundu")
    for class_name, idx in class_to_idx.items():
        count = labels.count(idx)
        print(f"{class_name}: {count} görüntü")
    
    # Veriyi eğitim ve doğrulama setlerine ayır
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Veri dönüşümlerini tanımla
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Veri yükleyicileri oluştur
    train_dataset = AlzheimerDataset(X_train, y_train, transform=train_transform, augment=True)
    val_dataset = AlzheimerDataset(X_val, y_val, transform=train_transform, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Model, kayıp fonksiyonu ve optimizasyon
    model = AlzheimerNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Modeli eğit
    best_model_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler
    )
    
    # En iyi modeli kaydet
    final_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'alzheimer_best.pth')
    torch.save(best_model_state, final_model_path)
    print(f"En iyi model kaydedildi: {final_model_path}")
    print(f"Doğrulama doğruluğu: {best_model_state['val_acc']:.2f}%")

if __name__ == "__main__":
    main() 