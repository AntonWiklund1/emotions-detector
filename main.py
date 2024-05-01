import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from model.model import DeiT
import constants
import train.train as train
from dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader

from train.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = constants.image_size
patch_size = constants.patch_size
num_classes = constants.num_classes
embed_dim = constants.embed_dim
T = constants.T
batch_size = constants.batch_size


def main():
    model = DeiT(image_size, patch_size, num_classes, embed_dim).to(device)
    teacher_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to the input size expected by the model
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    csv_file = "./data/train.csv"
    dataset = ImageDataset(csv_file=csv_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train(model, data_loader, teacher_model, criterion, optimizer, device)


if __name__ == "__main__":
    main()
