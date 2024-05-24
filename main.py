import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.DeiT import DeiT
from model.loss import distillation_loss
from dataset import ImageDataset
from model.ResNeXt import resnext50
import constants
from evaluation import test, train_and_validate
from evaluation.train import load_checkpoint
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from visualize.visualize import visualize_dataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = constants.image_size
patch_size = constants.patch_size
num_classes = constants.num_classes
embed_dim = constants.embed_dim
batch_size = constants.batch_size
lr = constants.lr
wight_decay = constants.weight_decay
num_epochs = constants.num_epochs


# Set seed for reproducibility
torch.manual_seed(42)

def main():
    # Initialize SummaryWriter once
    writer = SummaryWriter('runs/deit')
    
    df = pd.read_csv("./data/filtered_dataset.csv")

    # Convert pixel strings to numpy arrays
    pixel_arrays = np.array([np.array(row.split(), dtype=np.uint8).reshape(48, 48) for row in df['pixels']])

    # Flatten the arrays to compute global mean and std
    all_pixels = pixel_arrays.ravel()
    mean = all_pixels.mean() / 255.0  # Scale to [0, 1]
    std = all_pixels.std() / 255.0

    print(f"Mean: {mean}, Std: {std}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    model = DeiT(image_size, patch_size, num_classes, embed_dim).to(device)

    teacher_model = resnext50().to(device)
    teacher_model_weights = torch.load('checkpoint.pth')
    teacher_model.load_state_dict(teacher_model_weights['model_state_dict'])
    teacher_model.eval()  # Set teacher model to evaluation mode

    print("lr: ", lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wight_decay)
    #cosine annealing learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1)

    dataset_csv_file = "./data/filtered_dataset.csv"
    full_dataset = ImageDataset(csv_file=dataset_csv_file, transform=transform)

    visualize_dataset(full_dataset, num_images=64)

    # Split dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    #oversample train dataset

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    criterion = distillation_loss

    # Train and validate the model
    train_and_validate(model, train_data_loader, val_data_loader, teacher_model, criterion, optimizer, scheduler, writer)

    # Test the model
    test_csv_file = "./data/test_with_emotions.csv"

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    test_dataset = ImageDataset(csv_file=test_csv_file, transform=transform_test)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_acc = test(model, test_data_loader)

    print(f"Test accuracy: {test_acc}")

    # Close the writer after training and testing
    writer.close()

if __name__ == "__main__":
    main()
