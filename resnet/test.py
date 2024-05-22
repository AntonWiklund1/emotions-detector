import torch
from .resnet import own_resnet100, own_resnet50
from .SEBlockResNet import se_resnet50
from .ResNeXt import resnext50
from .grad_cam import GradCAM
from .visualize import visualize_dataset
from dataset.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)

def analyze_with_gradcam(images, labels, model, grad_cam, device, num_samples=5):
    model.eval()
    samples_processed = 0

    with torch.no_grad():
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
        transforms.Normalize(mean=[0.5077], std=[0.2550])
    ])

    test_df = ImageDataset('./data/test_with_emotions.csv', transform=transform)

    visualize_dataset(test_df, num_images=64)
    test_loader = DataLoader(test_df, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

    model = resnext50().to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    correct = 0
    total = 0

    target_layer = model.layer4[-1].conv3  # Change this to the layer you want to target
    grad_cam = GradCAM(model, target_layer)

    total_time = 0
    num_images = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()

            with torch.cuda.amp.autocast():
                outputs = model(images)

            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time
            num_images += images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Analyze Grad-CAM for the first batch only
            # analyze_with_gradcam(images, labels, model, grad_cam, device, num_samples=5)
            #break  # Only analyze the first batch for Grad-CAM

    accuracy = 100 * correct / total
    avg_time_per_image = total_time / num_images

    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Average time per image: {avg_time_per_image:.4f} seconds")

def main():
    test()

if __name__ == "__main__":
    main()
