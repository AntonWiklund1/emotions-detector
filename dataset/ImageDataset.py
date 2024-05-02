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
        """
        Retrieve an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: Containing an image and its label.
        """
        img_pixel_str = self.data_frame.iloc[idx]['pixels']
        label = int(self.data_frame.iloc[idx]['emotion'])

        # Parse the pixel string and prepare the image
        pixels = list(map(int, img_pixel_str.split()))
        image = np.array(pixels, dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(image).convert('RGB')  # Convert grayscale to RGB

        # Apply the Hugging Face processor if provided
        if self.processor:
            # Convert PIL image to bytes and process with Hugging Face processor
            image = self.processor(images=image, return_tensors='pt').pixel_values.squeeze(0)
        elif self.transform:
            image = self.transform(image)

        return image, label
