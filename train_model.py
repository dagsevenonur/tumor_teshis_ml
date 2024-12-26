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
        # Görüntüyü yükle
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Görüntü iyileştirme
        image = self.enhance_image(image)
        
        # Transform uygula
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

def prepare_data(data_dir):
    # Veri yollarını ve etiketleri topla
    image_paths = []
    labels = []
    
    # Tümörlü görüntüler
    tumor_dir = os.path.join(data_dir, 'yes')
    for img_name in os.listdir(tumor_dir):
        image_paths.append(os.path.join(tumor_dir, img_name))
        labels.append(1)  # 1: tümör var
        
    # Tümörsüz görüntüler
    no_tumor_dir = os.path.join(data_dir, 'no')
    for img_name in os.listdir(no_tumor_dir):
        image_paths.append(os.path.join(no_tumor_dir, img_name))
        labels.append(0)  # 0: tümör yok
    
    return train_test_split(image_paths, labels, test_size=0.2, random_state=42)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Eğitim
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
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
        
        train_acc = 100. * train_correct / train_total
        
        # Doğrulama
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/brain_tumor_best.pth')

def main():
    # Veri dönüşümleri
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Veriyi hazırla
    train_paths, val_paths, train_labels, val_labels = prepare_data('datasets/brain_tumor')
    
    # Veri setlerini oluştur
    train_dataset = BrainTumorDataset(train_paths, train_labels, transform)
    val_dataset = BrainTumorDataset(val_paths, val_labels, transform)
    
    # Veri yükleyicileri
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 sınıf: tümör var/yok
    
    # Kayıp fonksiyonu ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Modeli eğit
    train_model(model, train_loader, val_loader, criterion, optimizer)

if __name__ == '__main__':
    main() 