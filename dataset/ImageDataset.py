from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torch
import hashlib
import face_recognition

torch.manual_seed(42)

class ImageDataset(Dataset):
    """Custom dataset for loading images from CSV files with pixel values and emotion labels."""
    def __init__(self,csv_file, fix=None, transform=None, rows=None):
        self.data_frame = pd.read_csv(csv_file, nrows=rows)
        
        print(f"Loaded {len(self.data_frame)} images from {csv_file}")
        
        if self.data_frame.columns[0] == '':
            self.data_frame = self.data_frame.drop(self.data_frame.columns[0], axis=1)
        
        self.transform = transform
        self.fix = fix
        
        if self.fix:
            self.data_frame = self.data_frame[self.data_frame['pixels'].apply(self.is_valid_image)]
            self.data_frame = self.data_frame[self.data_frame['pixels'].apply(self.contains_face)]
            self.data_frame = self.remove_duplicate_images(self.data_frame)
            print(f"Filtered to {len(self.data_frame)} images")


        self.data_frame.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= len(self.data_frame):
            raise IndexError("Index out of bounds")
        
        img_pixel_str = self.data_frame.iloc[idx]['pixels']
        pixels = np.array(img_pixel_str.split(), dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(pixels, 'L')

        if self.transform:
            image = self.transform(image)

        label = int(self.data_frame.iloc[idx]['emotion'])
        return image, label
    
    def is_valid_image(self, pixel_string):
        pixels = np.array(pixel_string.split(), dtype=np.uint8)
        return np.any(pixels > 0)

    def contains_face(self, pixel_string):
        pixels = np.array(pixel_string.split(), dtype=np.uint8).reshape(48, 48)
        face_locations_hog = face_recognition.face_locations(pixels, model="hog")
        face_locations_cnn = face_recognition.face_locations(pixels, model="cnn")
        return len(face_locations_hog) > 0 or len(face_locations_cnn) > 0

    def remove_duplicate_images(self, data_frame):
        seen_hashes = set()
        unique_rows = []
        
        for idx, row in data_frame.iterrows():
            img_pixel_str = row['pixels']
            pixels = np.array(img_pixel_str.split(), dtype=np.uint8)
            img_hash = hashlib.md5(pixels).hexdigest()
            
            if img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                unique_rows.append(row)
        
        return pd.DataFrame(unique_rows)

