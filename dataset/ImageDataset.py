from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import torch

torch.manual_seed(42)

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, processor=None, rows=None):
        """
        Initialize the dataset.

        Args:
            csv_file (str): Path to the CSV file containing image data and labels.
            transform (callable, optional): Optional transform to be applied on each image.
            processor (callable, optional): Processor from Hugging Face for image preprocessing.
            rows (int, optional): Number of rows to read from the CSV file.
        """
        self.data_frame = pd.read_csv(csv_file, nrows=rows)
        self.transform = transform
        self.processor = processor

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_pixel_str = self.data_frame.iloc[idx]['pixels']
        label = int(self.data_frame.iloc[idx]['emotion'])
        pixels = np.fromstring(img_pixel_str, dtype=int, sep=' ').reshape(48, 48)
        image = Image.fromarray(pixels, 'L').convert('RGB')  # Convert grayscale to RGB if necessary

        if self.processor:
            image = self.processor(images=image, return_tensors='pt').pixel_values.squeeze(0)
        elif self.transform:
            image = self.transform(image)

        return image, label
