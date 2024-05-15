import torch
from .res_model import resnet
from dataset.ImageDataset import ImageDataset
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test():
    model = resnet().to(device)
    test_df = ImageDataset('./data/test_with_emotions.csv')
    test_loader = DataLoader(test_df, batch_size=1, shuffle=True)
    model.state_dict(torch.load('model.pth'))
    model.eval()

    correct = 0
    total = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Test accuracy: {accuracy}")


def main():
    test()

if __name__ == "__main__":
    main()