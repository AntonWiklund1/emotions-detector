import torch
from model.ResNeXt import ResNeXtBottleneck, ResNeXt
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from dataset.ImageDataset import ImageDataset
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Define the quantizable model
class QuantizableResNeXt(ResNeXt):
    def __init__(self, block, layers, image_channels=1, num_classes=7, cardinality=32, bottleneck_width=4):
        super(QuantizableResNeXt, self).__init__(block, layers, image_channels, num_classes, cardinality, bottleneck_width)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

def quantizable_resnext50(img_channels=1, num_classes=7):
    return QuantizableResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], img_channels, num_classes)

# Fuse Conv+BN+ReLU layers
def fuse_model(model):
    for m in model.modules():
        if isinstance(m, ResNeXtBottleneck):
            # Ensure the layers exist before fusing and ReLU is present
            if hasattr(m, 'conv1') and hasattr(m, 'bn1') and isinstance(m.relu, torch.nn.ReLU):
                fuse_modules(m, ['conv1', 'bn1', 'relu'], inplace=True)
            if hasattr(m, 'conv2') and hasattr(m, 'bn2') and isinstance(m.relu, torch.nn.ReLU):
                fuse_modules(m, ['conv2', 'bn2', 'relu'], inplace=True)
            if hasattr(m, 'conv3') and hasattr(m, 'bn3'):
                fuse_modules(m, ['conv3', 'bn3'], inplace=True)
            # Fuse the downsample layers if they exist
            if hasattr(m, 'identity_downsample') and m.identity_downsample is not None:
                fuse_modules(m.identity_downsample, ['0', '1'], inplace=True)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model_fp32 = quantizable_resnext50().to(device)
    checkpoint = torch.load('checkpoint.pth', map_location=device)
    model_fp32.load_state_dict(checkpoint['model_state_dict'])
    model_fp32.eval()

    # Print model to identify layers
    print(model_fp32)

    # Fuse model layers
    fuse_model(model_fp32)

    # Set the quantization configuration to use per_tensor_affine
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Load and process the dataset
    df = pd.read_csv("./data/train.csv")

    # Convert pixel strings to numpy arrays
    pixel_arrays = np.array([np.array(row.split(), dtype=np.uint8).reshape(48, 48) for row in df['pixels']])

    # Flatten the arrays to compute global mean and std
    all_pixels = pixel_arrays.ravel()
    mean = all_pixels.mean() / 255.0  # Scale to [0, 1]
    std = all_pixels.std() / 255.0

    print(f"Mean: {mean}, Std: {std}")

    # Prepare the model for static quantization
    torch.quantization.prepare(model_fp32, inplace=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    # Calibrate the model with representative data
    full_dataset = ImageDataset(csv_file="./data/train.csv", transform=transform, rows=1000)
    calibration_loader = DataLoader(full_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

    print("Calibrating model...")
    # Run calibration
    for input, _ in calibration_loader:
        input = input.to(device)
        model_fp32(input)
    print("Calibration complete.")

    # Move the model to CPU before quantization
    model_fp32.to('cpu')

    # Convert to quantized version
    torch.quantization.convert(model_fp32, inplace=True)

    # The model is now quantized and ready for inference
    model_int8 = model_fp32

    # Save the quantized model using torch.jit.save
    scripted_model = torch.jit.script(model_int8)
    torch.jit.save(scripted_model, 'quantized_ResNeXt50_scripted.pth')
