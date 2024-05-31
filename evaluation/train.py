import torch
torch.manual_seed(42)

import torch.nn as nn
import constants
from evaluation.validate import validate
from torch.utils.tensorboard import SummaryWriter
import datetime
import colorama
from colorama import Fore, Style
colorama.init()

from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# Format the datetime
now = datetime.datetime.now()
formatted_time = now.strftime("%m%d-%H%M")
current_time = formatted_time

lambda_coeff = constants.lambda_coeff
T = constants.T
num_epochs = constants.num_epochs

checkpoint = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_validate(model, data_loader, val_loader, teacher_model, criterion, optimizer, scheduler, writer):
    best_global_val_loss = float('inf')
    best_global_val_accuracy = float('-inf')
    best_val_loss = float('inf')
    start_epoch = 0
    start_step = 0

    model, optimizer, scheduler, best_val_loss, start_epoch, start_step = load_checkpoint(model, optimizer, scheduler, device)

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        model.train()
        teacher_model.eval()  # Teacher in no-grad mode

        for batch_idx, (images, labels) in enumerate(data_loader):
            step = start_step + epoch * len(data_loader) + batch_idx
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                class_logits, dist_logits = model(images)
                with torch.no_grad():
                    teacher_logits = teacher_model(images)

                loss = criterion((class_logits, dist_logits), teacher_logits, labels)

            scaler.scale(loss).backward()
            total_loss += loss.item()
            writer.add_scalar('Training loss', loss.item(), step)

        avg_training_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average training loss: {avg_training_loss:.4f}")

        val_loss, val_accuracy = validate(model, teacher_model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_global_val_loss = val_loss
            best_global_val_accuracy = val_accuracy
            save_checkpoint(model, optimizer, scheduler, best_val_loss, epoch, step)
            print(f"{Fore.GREEN}New best validation loss: {val_loss:.4f} Saving model...{Style.RESET_ALL}")

        writer.add_scalar('Validation loss', val_loss, epoch)
        writer.add_scalar('Validation accuracy', val_accuracy, epoch)
        scheduler.step(val_loss)

    writer.close()

def save_checkpoint(model, optimizer, scheduler, val_loss, epoch, step):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_loss': val_loss,
        'epoch': epoch,
        'step': step
    }
    torch.save(checkpoint, "DeiT_checkpoint.pth")
    print(f"Saved checkpoint at epoch {epoch} with validation loss {val_loss}")

def load_checkpoint(model, optimizer, scheduler, device):
    try:
        checkpoint = torch.load("DeiT_checkpoint.pth", map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        val_loss = checkpoint['val_loss']
        print(f"Loaded model with validation loss {val_loss}")
        return model, optimizer, scheduler, val_loss, checkpoint['epoch'], checkpoint['step']
    except FileNotFoundError:
        print("No checkpoint found. Training from scratch.")
    except RuntimeError as e:
        print(f"Error loading model: {str(e)}")
    return model, optimizer, scheduler, float('inf'), 0, 0
