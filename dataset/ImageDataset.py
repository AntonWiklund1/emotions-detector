from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torch
import hashlib

torch.manual_seed(42)

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, rows=None):
        # Handle CSV with or without index column
        self.data_frame = pd.read_csv(csv_file, nrows=rows)
        
        # If the CSV has an extra index column, drop it
        if self.data_frame.columns[0] == '':
            self.data_frame = self.data_frame.drop(self.data_frame.columns[0], axis=1)
        
        self.transform = transform

        # Filter out black or corrupted images
        self.data_frame = self.data_frame[self.data_frame['pixels'].apply(self.is_valid_image)]
        
        # Remove duplicate images
        self.data_frame = self.remove_duplicate_images(self.data_frame)
        
        self.data_frame.reset_index(drop=True, inplace=True)  # Reset index to avoid any out-of-bounds issues

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
        return np.any(pixels > 0)  # Check if any pixel is not black

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
