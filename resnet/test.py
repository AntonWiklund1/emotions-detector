import torch
from .resnet import resnet100, resnet50
from .SEBlockResNet import se_resnet50
from .ResNeXt import resnext50
from .grad_cam import GradCAM
from dataset.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def analyze_with_gradcam(test_loader, model, grad_cam, device, num_samples=5):
    model.eval()
    samples_processed = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # If you trained with mixed precision, use autocast here as well
            with torch.cuda.amp.autocast():
                outputs = model(images)
                
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(images.size(0)):
                if samples_processed >= num_samples:
                    return
                with torch.set_grad_enabled(True):
                    cam = grad_cam.generate_cam(images[i].unsqueeze(0), target_class=labels[i].item())
                original_image = images[i].cpu().numpy().transpose(1, 2, 0)
                grad_cam.visualize_cam(cam, original_image)
                samples_processed += 1

def test():
    
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5077], std=[0.2550])  # Normalize using training mean and std
    ])

    test_df = ImageDataset('./data/test_with_emotions.csv', transform=transform)
    test_loader = DataLoader(test_df, batch_size=128, shuffle=True)


    model = resnext50().to(device)
    model.load_state_dict(torch.load('last.pth'))

    model.eval()

    #print the information of the model
    #print(model)
    
    correct = 0
    total = 0

    target_layer = model.layer4[-1].conv3  # Change this to the layer you want to target
    grad_cam = GradCAM(model, target_layer)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            analyze_with_gradcam(test_loader, model, grad_cam, device, num_samples=5)

            # Limiting to visualize Grad-CAM for only the first batch
            break
            
    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")


def main():
    test()

if __name__ == "__main__":
    main()
