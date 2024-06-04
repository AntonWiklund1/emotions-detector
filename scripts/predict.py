import torch
from model.ResNeXt import resnext50
from visualize import GradCAM
from visualize.visualize import visualize_dataset, visualize_predictions
from dataset.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary  # Importing summary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
from visualize import GradCAM
import sys

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

def test(model):

    
    df = pd.read_csv("./data/test_with_emotions.csv")

    # Convert pixel strings to numpy arrays
    pixel_arrays = np.array([np.array(row.split(), dtype=np.uint8).reshape(48, 48) for row in df['pixels']])

    # Flatten the arrays to compute global mean and std
    all_pixels = pixel_arrays.ravel()
    mean = all_pixels.mean() / 255.0  # Scale to [0, 1]
    std = all_pixels.std() / 255.0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    test_df = ImageDataset('./data/test_with_emotions.csv', transform=transform)

    #visualize_dataset(test_df, num_images=64)
    test_loader = DataLoader(test_df, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

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
            # break  # Only analyze the first batch for Grad-CAM

    accuracy = 100 * correct / total
    avg_time_per_image = total_time / num_images

    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Average time per image: {avg_time_per_image:.4f} seconds")

    return accuracy, avg_time_per_image

def main():

    # model = resnext50().to(device)
    # checkpoint = torch.load('checkpoint.pth', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])

    model = torch.load('my_own_model.pkl', map_location=device)
    model.eval()

    # Redirect stdout to a file
    with open('model_summary.txt', 'w') as f:
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = f  # Redirect standard output to the file
        summary(model, input_size=(1, 224, 224))  # Print the summary
        sys.stdout = original_stdout  # Reset standard output to its original value
    
    print("Testing the model...")
    test(model)

if __name__ == "__main__":
    main()