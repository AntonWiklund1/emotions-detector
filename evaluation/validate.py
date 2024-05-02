import torch

def validate(model, data_loader, criterion, device):
    model.eval()  # Sets the model to evaluation mode.
    total_loss = 0
    correct = 0  # To count the number of correct predictions.
    total = 0  # Total number of labels processed.
    
    with torch.no_grad():  # No gradients need to be computed during validation.
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            class_logits = outputs[0]

            loss = criterion(class_logits, labels)
            total_loss += loss.item()  # Summing up the loss for averaging later.
            
            # Calculate accuracy:
            correct += (class_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    val_loss = total_loss / len(data_loader)  # Average loss
    val_accuracy = 100 * correct / total  # Overall accuracy percentage

    return val_loss, val_accuracy
