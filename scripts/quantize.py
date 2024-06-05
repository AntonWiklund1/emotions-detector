import torch
import torch.nn as nn
import math
from model.ResNeXt import CBAM, ResNeXt
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from dataset.ImageDataset import ImageDataset
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Set the quantization backend to 'qnnpack'
torch.backends.quantized.engine = 'qnnpack'

# Custom quantized batch normalization function
def quantized_batch_norm(input, running_mean, running_var, weight, bias, eps, output_scale, output_zero_point):
    return torch.ops.quantized.batch_norm2d(
        input, weight, bias, running_mean, running_var, eps, output_scale.item(), output_zero_point.item()
    )

# Define the quantizable model
class QuantizableResNeXt(ResNeXt):
    def __init__(self, block, layers, image_channels=1, num_classes=7, cardinality=32, bottleneck_width=4):
        super(QuantizableResNeXt, self).__init__(block, layers, image_channels, num_classes, cardinality, bottleneck_width)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        print("Input to quant:", x.shape, x.dtype)
        x = self.quant(x)
        print("Input after quant:", x.shape, x.dtype)
        x = self._forward_impl(x)
        print("Output from _forward_impl:", x.shape, x.dtype)
        x = self.dequant(x)
        print("Output after dequant:", x.shape, x.dtype)
        return x

class ResNeXtBottleneckQuantized(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, cardinality=32, bottleneck_width=4, identity_downsample=None, stride=1):
        super(ResNeXtBottleneckQuantized, self).__init__()
        D = int(math.floor(out_channels * (bottleneck_width / 64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(in_channels, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)

        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)

        self.conv3 = nn.Conv2d(D * C, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.identity_downsample = identity_downsample

        self.cbam = CBAM(out_channels * self.expansion)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if out.dtype == torch.quint8:
            out = quantized_batch_norm(out, self.bn1.running_mean, self.bn1.running_var, self.bn1.weight, self.bn1.bias, self.bn1.eps, out.q_scale(), out.q_zero_point())
        else:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if out.dtype == torch.quint8:
            out = quantized_batch_norm(out, self.bn2.running_mean, self.bn2.running_var, self.bn2.weight, self.bn2.bias, self.bn2.eps, out.q_scale(), out.q_zero_point())
        else:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if out.dtype == torch.quint8:
            out = quantized_batch_norm(out, self.bn3.running_mean, self.bn3.running_var, self.bn3.weight, self.bn3.bias, self.bn3.eps, out.q_scale(), out.q_zero_point())
        else:
            out = self.bn3(out)

        out = self.cbam(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.relu(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

def quantizable_resnext50(img_channels=1, num_classes=7):
    return QuantizableResNeXt(ResNeXtBottleneckQuantized, [3, 4, 6, 3], img_channels, num_classes)

# Fuse Conv+BN+ReLU layers
def fuse_model(model):
    for m in model.modules():
        if isinstance(m, ResNeXtBottleneckQuantized):
            # Ensure the layers exist before fusing and ReLU is present
            if hasattr(m, 'conv1') and hasattr(m, 'bn1') and isinstance(m.relu, torch.nn.ReLU):
                fuse_modules(m, ['conv1', 'bn1', 'relu'], inplace=True)
            if hasattr(m, 'conv2') and hasattr(m, 'bn2') and isinstance(m.relu, torch.nn.ReLU):
                fuse_modules(m, ['conv2', 'bn2', 'relu'], inplace=True)
            if hasattr(m, 'conv3') and hasattr(m, 'bn3') and isinstance(m.relu, torch.nn.ReLU):
                fuse_modules(m, ['conv3', 'bn3', 'relu'], inplace=True)
            elif hasattr(m, 'conv3') and hasattr(m, 'bn3'):
                fuse_modules(m, ['conv3', 'bn3'], inplace=True)
            # Fuse the downsample layers if they exist
            if hasattr(m, 'identity_downsample') and m.identity_downsample is not None:
                if isinstance(m.identity_downsample[1], torch.nn.BatchNorm2d):
                    fuse_modules(m.identity_downsample, ['0', '1'], inplace=True)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model_fp32 = quantizable_resnext50().to(device)
    checkpoint = torch.load('checkpoint.pth', map_location=device)
    model_fp32.load_state_dict(checkpoint['model_state_dict'])
    model_fp32.eval()

    # Fuse model layers
    fuse_model(model_fp32)

    # Set the quantization configuration to use qnnpack with per_channel_weight_observer
    model_fp32.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.default_observer,
        weight=torch.quantization.default_per_channel_weight_observer
    )
    print("QConfig:", model_fp32.qconfig)

    # Prepare the model for static quantization
    torch.quantization.prepare(model_fp32, inplace=True)
    print("Model prepared for quantization")

    # Load and process the dataset
    df = pd.read_csv("./data/train.csv")

    # Convert pixel strings to numpy arrays
    pixel_arrays = np.array([np.array(row.split(), dtype=np.uint8).reshape(48, 48) for row in df['pixels']])

    # Flatten the arrays to compute global mean and std
    all_pixels = pixel_arrays.ravel()
    mean = all_pixels.mean() / 255.0  # Scale to [0, 1]
    std = all_pixels.std() / 255.0

    print(f"Mean: {mean}, Std: {std}")

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
        print(f"Input shape during calibration: {input.shape}")
        model_fp32(input)
    print("Calibration complete.")

    # Movethe model to CPU before quantization
    model_fp32.to('cpu')

    # Convert to quantized version
    torch.quantization.convert(model_fp32, inplace=True)

    # Save the print model to a file after quantization
    with open('model_fp32_after_quantization.txt', 'w') as f:
        f.write(str(model_fp32))

    # The model is now quantized and ready for inference
    model_int8 = model_fp32

    # Save the quantized model using torch.jit.save
    scripted_model = torch.jit.script(model_int8)
    torch.jit.save(scripted_model, 'quantized_ResNeXt50_scripted.pth')

    # Verify the quantized model during inference
    scripted_model.eval()
    for input, _ in calibration_loader:
        input = input.to('cpu')
        print("Running inference on quantized model...")
        print("Input to quant:", input.shape, input.dtype)
        output = scripted_model(input)
        print(f"Inference output shape: {output.shape}, dtype: {output.dtype}")

