import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler

import datetime
from .resnet import resnet50, resnet100
from .SEBlockResNet import se_resnet50
from .ResNeXt import resnext50
from dataset.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR  # Import StepLR

from sklearn.model_selection import KFold
import colorama
from colorama import Fore, Style
colorama.init()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Format the datetime
now = datetime.datetime.now()
formatted_time = now.strftime("%m%d-%H%M")
current_time = formatted_time

# Hyperparameters
batch_size = 128
lr =  5e-4  # 5e-4 for resnet50
weight_decay = 0.02
num_epochs = 100
step_size = 30  # Number of epochs to decay learning rate
gamma = 0.1  # Factor by which the learning rate will be reduced

# Uncomment these if you want to switch back to the previous scheduler
#warmup_epochs = 5

# Empty cache
torch.cuda.empty_cache()
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_global_val_loss = float('inf')
best_global_val_accuracy = float('-inf')

def train_and_validate(model, train_loader, val_loader, fold_number, criterion, optimizer, scheduler, scaler):
    global best_global_val_loss
    global best_global_val_accuracy

    tensorboard_title = f"ed_resnet_fold_{fold_number}_{current_time}"
    writer = SummaryWriter(f"runs/{tensorboard_title}")
    log(f"{tensorboard_title} - Hyperparameters: batch_size={batch_size}, lr={lr}, num_epochs={num_epochs}, optimizer=AdamW, scheduler=StepLR with step_size={step_size}, gamma={gamma}")
    
    best_val_loss = float('inf')
    best_val_accuracy = float('-inf')

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            writer.add_scalar('Training loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Log histograms of weights and gradients
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'{name}.grad', param.grad, epoch)

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
                torch.save(model.state_dict(), "model.pth")
                print(f"{Fore.GREEN}New best validation loss: {val_loss:.4f} Saving model...{Style.RESET_ALL}")
        writer.add_scalar('Validation loss', val_loss, epoch)
        writer.add_scalar('Validation accuracy', val_accuracy, epoch)
        scheduler.step()  # Scheduler step at the end of each epoch
    
    writer.close()
    log(f"{tensorboard_title} - Best global validation loss: {best_global_val_loss:.4f}, validation accuracy: {best_global_val_accuracy:.2f}%")
    torch.save(model.state_dict(), "last.pth")

def validate(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = val_loss / len(val_loader)
    avg_accuracy = 100 * correct / total
    return avg_loss, avg_accuracy

def create_folds(full_dataset, n_splits):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(kfold.split(full_dataset))

def visualize_dataset(dataset, num_images=16):
    rows = int(num_images ** 0.5)
    cols = int(num_images ** 0.5)
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        image, _ = dataset[idx]
        ax.imshow(image.squeeze(), cmap='gray')  # Assuming the image is grayscale
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def log(text):
    """Log the text to log.txt"""
    with open("log.txt", 'a') as f:
        f.write(text + '\n')

def plot_confusion_matrix(model, val_loader, classes, device):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def visualize_predictions(model, dataset, device, num_samples=5):
    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(device)
            with torch.cuda.amp.autocast():
                output = model(image)
                _, pred = torch.max(output, 1)
            image = image.cpu().squeeze().numpy()
            label = label
            pred = pred.item()
            
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f"Pred: {emotion_classes[pred]}\nTrue: {emotion_classes[label]}")
            axes[i].axis('off')
    
    plt.show()

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
        transforms.RandAugment(num_ops=6, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    dataset_csv_file = "./data/train.csv"
    full_dataset = ImageDataset(csv_file=dataset_csv_file, transform=transform)

    visualize_dataset(full_dataset, num_images=64)

    folds = create_folds(full_dataset, n_splits=5)
    print(f"Training {len(folds)} folds on {device}")

    # Calculate class weights
    class_weights = calculate_class_weights(df)
    sample_weights = [class_weights[label] for label in df['emotion']]
    
    for i, (train_idx, val_idx) in enumerate(folds):
        model = resnext50().to(device)
        print(f"Training fold {i+1}/{len(folds)}")
        
        # Use SubsetRandomSampler for training and validation indices
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx, generator=torch.Generator().manual_seed(42))
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx, generator=torch.Generator().manual_seed(42))

        # Create a new sampler for the training subset
        train_subset_weights = [sample_weights[idx] for idx in train_idx]
        train_sampler = WeightedRandomSampler(weights=train_subset_weights, num_samples=len(train_subset_weights), replacement=True)

        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=8, pin_memory=True)

        class_weights_tensor = torch.tensor(class_weights).float().to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        scaler = torch.cuda.amp.GradScaler()

        train_and_validate(model, train_loader, val_loader, i+1, criterion, optimizer, scheduler, scaler)

        emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        plot_confusion_matrix(model, val_loader, emotion_classes, device)
        visualize_predictions(model, full_dataset, device, num_samples=5)

        break  # Remove this line to train all folds

if __name__ == '__main__':
    main()

