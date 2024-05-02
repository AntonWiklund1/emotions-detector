import torch
import torch.nn as nn
import constants
from evaluation.validate import validate
from torch.utils.tensorboard import SummaryWriter

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


writer = SummaryWriter(f'runs/ed_{current_time}')

lambda_coeff = constants.lambda_coeff
T = constants.T
num_epochs = constants.num_epochs
torch.manual_seed(42)

def train_and_validate(model, data_loader, val_loader, teacher_model, criterion, optimizer, device):
    best_val_loss = float('inf')  # Initialize the best validation loss to infinity
    for epoch in range(num_epochs):
        model.train()
        teacher_model.eval()  # Teacher in no-grad mode

        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass through the student model
            class_logits, dist_logits = model(images)

            # Ensure to pass 'images' as 'inputs' to the DistillationLoss
            loss = criterion(images, (class_logits, dist_logits), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add to tensorboard
            writer.add_scalar('Training loss', loss.item(), epoch * len(data_loader) + batch_idx)

        # Validation after each epoch
        val_loss, val_accuracy = validate(model, teacher_model, val_loader, criterion, device)
        writer.add_scalar('Validation loss', val_loss, epoch)
        print(f"Epoch {epoch}: Training Loss: {loss.item()}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved better model with validation loss:", val_loss)

