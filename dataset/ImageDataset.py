from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)[:1000]
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_pixel_str = self.data_frame.iloc[idx, 1]  # Assuming 'pixels' column is the second
        pixels = list(map(int, img_pixel_str.split()))
        image = np.array(pixels).reshape(48, 48).astype(np.uint8)
        image = Image.fromarray(image).convert('RGB')  # Convert grayscale to RGB
        label = int(self.data_frame.iloc[idx, 0])  # Assuming 'emotion' column is the first

        if self.transform:
            image = self.transform(image)

        return image, label

