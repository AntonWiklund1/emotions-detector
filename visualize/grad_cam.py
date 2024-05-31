import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to save gradients and activations
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class=None):
        with torch.set_grad_enabled(True):
            model_output = self.model(input_image)
            if target_class is None:
                target_class = model_output.argmax(dim=1).item()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Target for backprop
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(input_image.device)
            one_hot_output[0][target_class] = 1
            
            # Backward pass with respect to the specific class
            model_output.backward(gradient=one_hot_output, retain_graph=True)
            
            # Get the gradients from the target layer
            guided_gradients = self.gradients.cpu().data.numpy()
            
            # Get the activations of the target layer
            target_activations = self.activations.cpu().data.numpy()[0]
            
            # Weighted sum of the target activations
            weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
            cam = np.ones(target_activations.shape[1:], dtype=np.float32)
            
            for i, w in enumerate(weights):
                cam += w * target_activations[i, :, :]
            
            cam = np.maximum(cam, 0)
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            return cam

    def visualize_cam(self, cam, original_image, alpha=0.4):
        # Ensure cam is a numpy array and float32
        cam = np.array(cam, dtype=np.float32)
        
        # Resize cam to match the size of the original image
        cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Ensure the heatmap is properly created
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        # Normalize the original image
        original_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image))
        
        # Ensure the original image is in the correct format
        if len(original_image.shape) == 2:  # Grayscale to RGB
            original_image = np.stack([original_image] * 3, axis=-1)
        elif original_image.shape[0] == 1:  # Single channel to RGB
            original_image = np.repeat(original_image, 3, axis=0).transpose(1, 2, 0)
        elif original_image.shape[0] == 3:  # If the image is in (C, H, W) format
            original_image = original_image.transpose(1, 2, 0)

        # Combine heatmap with the original image
        cam = heatmap * alpha + original_image * (1 - alpha)
        cam = cam / np.max(cam)  # Scale between 0 and 1

        plt.imshow(np.uint8(255 * cam))
        plt.axis('off')
        plt.show()
