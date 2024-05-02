import torch

def validate(model, teacher_model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # No gradients needed for validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass through the student model
            class_logits, dist_logits = model(images)
            # Forward pass through the teacher model for distillation evaluation
            teacher_outputs = teacher_model(images)
            # Compute the loss using all required inputs
            loss = criterion(images, (class_logits, dist_logits), labels)
            total_loss += loss.item()
            # Calculate accuracy if needed
            _, predicted = torch.max(class_logits.data, 1)
            total_accuracy += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = total_accuracy / len(val_loader.dataset) * 100
    return avg_loss, avg_accuracy

