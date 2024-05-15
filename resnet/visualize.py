from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_confusion_matrix(model, val_loader, classes, device):
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

def visualize_predictions(model, dataset, device, num_samples=5, emotion_classes=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            _, pred = torch.max(output, 1)
            image = image.cpu().squeeze().numpy()
            label = label.item()
            pred = pred.item()
            
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f"Pred: {emotion_classes[pred]}\nTrue: {emotion_classes[label]}")
            axes[i].axis('off')
    
    plt.show()