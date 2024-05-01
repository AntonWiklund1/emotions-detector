import torch
import torch.nn as nn
import constants

lambda_coeff = constants.lambda_coeff
T = constants.T
num_epochs = constants.num_epochs

def train(model, data_loader, teacher_model, criterion, optimizer, device):
    model.train()
    teacher_model.eval()
    for epoch in range(num_epochs):
        
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

        print(f"Loss: {loss.item()}, Epoch: {epoch}")
