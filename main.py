import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from model.model import DeiT
import constants
import evaluation.train as train
from dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import AutoImageProcessor, AutoModelForImageClassification
from evaluation import test, train_and_validate, validate
from model.losses import DistillationLoss

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = constants.image_size
patch_size = constants.patch_size
num_classes = constants.num_classes
embed_dim = constants.embed_dim
T = constants.T
batch_size = constants.batch_size
lr = constants.lr
lambda_coeff = constants.lambda_coeff

#SET SEED
torch.manual_seed(42)

def main():
    model = DeiT(image_size, patch_size, num_classes, embed_dim).to(device)
    # teacher_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    # teacher_model.fc = nn.Linear(in_features=teacher_model.fc.in_features, out_features=num_classes).to(device)

    processor = AutoImageProcessor.from_pretrained("ycbq999/facial_emotions_image_detection")
    teacher_model = AutoModelForImageClassification.from_pretrained("ycbq999/facial_emotions_image_detection").to(device)


    # Load the models checkpoint if it matches the model architecture
    try:
        model.load_state_dict(torch.load("best_model.pth"))
        print("Loaded model checkpoint")
    except RuntimeError:
        print("Model checkpoint does not match the model architecture. Training the model from scratch")
    

    base_criterion = nn.CrossEntropyLoss()
    criterion = DistillationLoss(base_criterion, teacher_model, 'soft', lambda_coeff, T)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_csv_file = "./data/train.csv"  # Single dataset file
    full_dataset = ImageDataset(csv_file=dataset_csv_file, transform=transform, rows=5000, processor=processor)

    # Split dataset into train and validation
    train_size = int(0.8 * len(full_dataset))  # 80% of the dataset for training
    val_size = len(full_dataset) - train_size  # Remaining 20% for validation
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train and validate the model
    train_and_validate(model, train_data_loader, val_data_loader, teacher_model, criterion, optimizer, device)

    # Test the model
    test_csv_file = "./data/test_with_emotions.csv"
    test_dataset = ImageDataset(csv_file=test_csv_file, transform=transform)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test(model, test_data_loader)

if __name__ == "__main__":
    main()

