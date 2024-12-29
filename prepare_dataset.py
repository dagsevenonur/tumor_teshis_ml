import pandas as pd
import os
from PIL import Image
import io

# Sınıf isimleri
CLASS_NAMES = ['non_demented', 'very_mild_demented', 'mild_demented', 'moderate_demented']

# Veri seti yolları
train_path = "datasets/Alzheimer MRI Disease Classification Dataset/Data/train-00000-of-00001-c08a401c53fe5312.parquet"
test_path = "datasets/Alzheimer MRI Disease Classification Dataset/Data/test-00000-of-00001-44110b9df98c5585.parquet"

# Hedef klasörler
base_dir = "datasets/alzheimer_processed"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Klasörleri oluştur
for class_name in CLASS_NAMES:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

def save_images(df, base_path, split_name):
    print(f"\n{split_name} görüntüleri kaydediliyor...")
    for idx, row in df.iterrows():
        # Sınıf adını al
        class_name = CLASS_NAMES[row['label']]
        
        # Görüntüyü bytes'dan PIL Image'a dönüştür
        img_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        
        # Dosya adı oluştur
        file_name = f"{split_name}_{idx:04d}.jpg"
        save_path = os.path.join(base_path, class_name, file_name)
        
        # Görüntüyü kaydet
        img.save(save_path, 'JPEG')
        
        if idx % 100 == 0:
            print(f"İşlenen görüntü: {idx}")

# Train verilerini işle
print("Train verisi okunuyor...")
train_df = pd.read_parquet(train_path)
save_images(train_df, train_dir, "train")

# Test verilerini işle
print("\nTest verisi okunuyor...")
test_df = pd.read_parquet(test_path)
save_images(test_df, test_dir, "test")

print("\nİşlem tamamlandı!") 