import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import datetime
from model.ResNeXt import resnext50
from visualize.visualize import plot_confusion_matrix, visualize_predictions, visualize_dataset
from dataset.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms

import colorama
from colorama import Fore, Style
colorama.init()
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
import numpy as np
import pandas as pd
import warnings

from utils import create_folds, log, load_pretrained_weights, get_checkpoint, CustomLRFinder

import constants
from .validate import validate
from .predict import test

from torch_lr_finder import LRFinder

from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.backends.cudnn.benchmark = True


warnings.filterwarnings("ignore")

# Format the datetime
now = datetime.datetime.now()
formatted_time = now.strftime("%m%d-%H%M")
current_time = formatted_time

# Hyperparameters
batch_size = constants.batch_size
lr = constants.lr
num_epochs = constants.num_epochs
weight_decay = constants.weight_decay
label_smoothing = constants.label_smoothing

# Empty cache
torch.cuda.empty_cache()
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
#torch.backends.cudnn.benchmark = True # Enable cuDNN benchmark mode for faster training 

best_global_val_loss = float('inf')
best_global_val_accuracy = float('-inf')

def train_and_validate(model, train_loader, val_loader, fold_number, criterion, optimizer, scheduler, scaler, start_epoch=0, start_step=0):
    global best_global_val_loss
    global best_global_val_accuracy

    tensorboard_title = f"ed_resnet_fold_{fold_number}_{current_time}"
    writer = SummaryWriter(f"runs/{tensorboard_title}")
    log(f"{tensorboard_title} - Hyperparameters: batch_size={batch_size}, lr={lr}, num_epochs={num_epochs}, optimizer=AdamW, scheduler=ReduceLROnPlateau, weight_decay={weight_decay}")

    best_val_loss = float('inf')
    best_val_accuracy = float('-inf')

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            step = start_step + epoch * len(train_loader) + batch_idx
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            writer.add_scalar('Training loss', loss.item(), step)

        # Log histograms of weights and gradients
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)
        #     if param.grad is not None:
        #         writer.add_histogram(f'{name}.grad', param.grad, epoch)

        avg_training_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average training loss: {avg_training_loss:.4f}")

        val_loss, val_accuracy = validate(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.2f}%")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            if best_val_loss < best_global_val_loss:
                best_global_val_loss = best_val_loss
                best_global_val_accuracy = best_val_accuracy
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'epoch': epoch,
                    'step': step
                }, "checkpoint_test.pth")
                print(f"{Fore.GREEN}New best validation loss: {val_loss:.4f} Saving model...{Style.RESET_ALL}")
        writer.add_scalar('Validation loss', val_loss, epoch)
        writer.add_scalar('Validation accuracy', val_accuracy, epoch)
        scheduler.step(val_loss)  # Scheduler step at the end of each epoch

    writer.close()
    log(f"{tensorboard_title} - Best global validation loss: {best_global_val_loss:.4f}, validation accuracy: {best_global_val_accuracy:.2f}%")
    torch.save(model.state_dict(), "last.pth")


def find_learning_rate(model, train_loader, criterion):
    print("Finding optimal learning rate...")
    optimizer = AdamW(model.parameters(), lr=1e-7, weight_decay=weight_decay)

    lr_finder = CustomLRFinder(model, optimizer, criterion, device='cuda')
    
    lr_finder.range_test(train_loader, end_lr=10, num_iter=1000, step_mode='exp')
    
    lr_finder.plot()
    lr_finder.reset()


def calculate_class_weights(data_frame):
    class_counts = data_frame['emotion'].value_counts().sort_index().values
    total_count = len(data_frame)
    class_weights = total_count / (len(class_counts) * class_counts)
    return class_weights

def main():
    df = pd.read_csv("./data/train.csv")

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

    dataset_csv_file = "./data/train.csv"
    full_dataset = ImageDataset(csv_file=dataset_csv_file, transform=transform)

    #visualize_dataset(full_dataset, num_images=64)

    folds = create_folds(full_dataset, n_splits=5)
    print(f"Training {len(folds)} folds on {device}")

    # Calculate class weights
    class_weights = calculate_class_weights(df)
    sample_weights = [class_weights[label] for label in df['emotion']]
    pretrained_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1, progress=True)

    # Initialize variables to resume training if checkpoint exists
    start_epoch = 0
    start_step = 0

    for i, (train_idx, val_idx) in enumerate(folds):
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}, step {start_step}")

        # Continue training or start fresh if no checkpoint is loaded
        print(f"Training fold {i+1}/{len(folds)}")

        # Use SubsetRandomSampler for training and validation indices
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx, generator=torch.Generator().manual_seed(42))
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx, generator=torch.Generator().manual_seed(42))

        # Create a new sampler for the training subset
        train_subset_weights = [sample_weights[idx] for idx in train_idx]
        train_sampler = WeightedRandomSampler(weights=train_subset_weights, num_samples=len(train_subset_weights), replacement=True)

        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=8, pin_memory=True)

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        model = resnext50().to(device)
        # checkpoint = torch.load('checkpoint.pth')
        # model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)
        scaler = torch.cuda.amp.GradScaler()

        #get_checkpoint("checkpoint.pth", model, optimizer, scheduler, scaler)

        #load_pretrained_weights(model, pretrained_model)

        #find_learning_rate(model, train_loader, criterion)

        train_and_validate(model, train_loader, val_loader, i+1, criterion, optimizer, scheduler, scaler, start_epoch=start_epoch, start_step=start_step)

        #emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        #plot_confusion_matrix(model, val_loader, emotion_classes, device)
        #visualize_predictions(model, full_dataset, device, num_samples=5)
    
    print(f"Best global validation loss: {best_global_val_loss:.4f}, validation accuracy: {best_global_val_accuracy:.2f}%")

    accuracy, avg_time_per_image = test(model)

    print(f"Test accuracy: {accuracy:.2f}%, Average time per image: {avg_time_per_image:.4f} seconds")

if __name__ == '__main__':
    main()
