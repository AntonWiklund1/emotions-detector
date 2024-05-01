from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import torch
torch.manual_seed(42)

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Initialize the dataset.

        Args:
            csv_file (str): Path to the CSV file containing image data and labels.
            transform (callable, optional): Transform to be applied on each image.
        """
        self.data_frame = pd.read_csv(csv_file)[:1000]  # Load the first 1000 entries
        self.transform = transform

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: Containing an image and its label.
        """
        # Retrieve row by index and then access columns directly
        img_pixel_str = self.data_frame.iloc[idx]['pixels']  # Correct indexing to fetch single row's pixels
        label = int(self.data_frame.iloc[idx]['emotion'])  # Correct indexing to fetch single row's label

        # Parse the pixel string
        pixels = list(map(int, img_pixel_str.split()))
        image = np.array(pixels, dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(image).convert('RGB')  # Convert grayscale to RGB

        if self.transform:
            image = self.transform(image)

        return image, label
