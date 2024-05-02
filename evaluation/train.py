import torch
import torch.nn as nn
import constants
from evaluation.validate import validate

lambda_coeff = constants.lambda_coeff
T = constants.T
num_epochs = constants.num_epochs
torch.manual_seed(42)

def train_and_validate(model, data_loader, val_loader, teacher_model, criterion, optimizer, device):
    best_val_loss = float('inf')  # Initialize the best validation loss to infinity
    print(f"Training the model on {device} ")
    for epoch in range(num_epochs):
        model.train()
        teacher_model.eval()

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            class_logits, dist_logits = model(images)
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            
            # Calculate losses
            loss_class = criterion(class_logits, labels)
            loss_distill = criterion(dist_logits, torch.softmax(teacher_logits / T, dim=-1))
            loss = loss_class + lambda_coeff * loss_distill

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation step after each epoch
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Training Loss: {loss.item()}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved better model with validation loss:", val_loss)