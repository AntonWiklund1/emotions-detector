import torch

from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import datetime
from .res_model import resnet
from dataset.ImageDataset import ImageDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import KFold
import colorama
from colorama import Fore, Style
colorama.init()

import matplotlib.pyplot as plt

# Format the datetime
now = datetime.datetime.now()
formatted_time = now.strftime("%m%d-%H%M")
current_time = formatted_time


#hyperparameters
batch_size = 256
lr = 0.01
step_size = 30
gamma = 0.1 # Multiplicative factor of learning rate decay


# Training parameters
num_epochs = 100
folds = 5
torch.manual_seed(42)

best_global_val_loss = float('inf')
best_global_val_accuracy = float('-inf')

def train_and_validate(model, train_loader, val_loader, fold_number, criterion, optimizer, scheduler):

    global best_global_val_loss
    global best_global_val_accuracy

    tensorboard_title = f"ed_resnet_fold_{fold_number}_{current_time}"
    writer = SummaryWriter(f"runs/{tensorboard_title}")
    log(f"{tensorboard_title} - Hyperparameters: batch_size={batch_size}, lr={lr}, num_epochs={num_epochs}, step_size={step_size}, gamma={gamma}")
    
    best_val_loss = float('inf')
    best_val_accuracy = float('-inf')

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Training loss', loss.item(), epoch * len(train_loader) + batch_idx)

        val_loss, val_accuracy = validate(model, val_loader, criterion)
        print(f"Epoch: {epoch}, Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            if best_val_loss < best_global_val_loss:
                best_global_val_loss = best_val_loss
                best_global_val_accuracy = best_val_accuracy
                torch.save(model.state_dict(), "model.pth")
                print(f"{Fore.GREEN}New best validation loss: {val_loss} Saving model...{Style.RESET_ALL}")
        writer.add_scalar('Validation loss', val_loss, epoch)
        writer.add_scalar('Validation accuracy', val_accuracy, epoch)
        scheduler.step()
    
    writer.close()
    log(f"{tensorboard_title} - Best global validation loss: {best_global_val_loss}, validation accuracy: {best_global_val_accuracy}")

def validate(model,val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
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

def visualize_dataset(dataset, index):
    image, _ = dataset[index]
    plt.imshow(image.squeeze(), cmap='gray')  # Assuming the image is grayscale
    plt.show()

def log(text):
    """Log the text to log.txt"""
    with open("log.txt", 'a') as f:
        f.write(text + '\n')

def main():
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Flips the image horizontally with a probability of 0.5
        transforms.RandomRotation(degrees=10),  # Rotates the image by up to 10 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Random affine transformation
        transforms.Resize((48, 48)),  # Resize back to 48x48 if transformations cause size changes
        transforms.ToTensor(),  # Convert the PIL Image to a tensor
        transforms.Normalize(mean=[0.485], std=[0.229])  # Normalization
    ])


    dataset_csv_file = "./data/train.csv"
    full_dataset = ImageDataset(csv_file=dataset_csv_file, transform=transform)
    for i in range(2):
        visualize_dataset(full_dataset, i)

    folds = create_folds(full_dataset, n_splits=5)
    print(f"Training {len(folds)} folds on {device}")
    for i, (train_idx, val_idx) in enumerate(folds):
        model = resnet().to(device)
        print(f"Training fold {i+1}/{len(folds)}")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_subsampler)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        train_and_validate(model, train_loader, val_loader, i+1, criterion, optimizer, scheduler)
        break # Remove this line to train all folds

if __name__ == '__main__':
    main()