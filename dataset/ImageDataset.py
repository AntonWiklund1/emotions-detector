from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torch
torch.manual_seed(42)


class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, rows=None):
        """
        Initialize the dataset.

        Args:
            csv_file (str): Path to the CSV file containing image data and labels.
            transform (callable, optional): Optional transform to be applied on each image.
            rows (int, optional): Number of rows to read from the CSV file. Useful for debugging or reduced dataset training.
        """
        self.data_frame = pd.read_csv(csv_file, nrows=rows)
        self.transform = transform

        # Filter out black or corrupted images
        self.data_frame = self.data_frame[self.data_frame['pixels'].apply(self.is_valid_image)]
        
    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Retrieve the pixel string and convert it into numpy array of integers
        img_pixel_str = self.data_frame.iloc[idx]['pixels']
        pixels = np.array(img_pixel_str.split(), dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(pixels, 'L')  # Create a PIL image in grayscale ('L' mode)

        # Convert grayscale to RGB
        # image = image.convert('RGB')
        
        # Apply the transform if it is specified
        if self.transform:
            image = self.transform(image)

        # Retrieve the label and convert it to integer
        label = int(self.data_frame.iloc[idx]['emotion'])
        return image, label
    
    def is_valid_image(self, pixel_string):
        pixels = np.array(pixel_string.split(), dtype=np.uint8)
        return np.any(pixels > 0)  # Check if any pixel is not black

