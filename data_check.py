import pandas as pd
import os

# Dosya yolları
train_path = "datasets/Alzheimer MRI Disease Classification Dataset/Data/train-00000-of-00001-c08a401c53fe5312.parquet"
test_path = "datasets/Alzheimer MRI Disease Classification Dataset/Data/test-00000-of-00001-44110b9df98c5585.parquet"

# Dosyaları oku
print("Train verisi okunuyor...")
train_df = pd.read_parquet(train_path)
print("\nTrain veri yapısı:")
print(train_df.info())
print("\nİlk birkaç satır:")
print(train_df.head())

print("\nTest verisi okunuyor...")
test_df = pd.read_parquet(test_path)
print("\nTest veri yapısı:")
print(test_df.info())
print("\nİlk birkaç satır:")
print(test_df.head()) 