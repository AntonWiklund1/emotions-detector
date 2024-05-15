import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class GradCAM:
    """ 
    Class for generating Grad-CAM visualizations.

    Parameters:
    - model (nn.Module): PyTorch model to visualize.
    - target_layer (nn.Module): Target layer to visualize.
    
    return:
    - cam: Class Activation Map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Register hook to save gradients of the target layer
        target_layer.register_backward_hook(self.save_gradients)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, target_class=None):
        model_output = self.model(input_image)
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        
        # Backward pass with respect to the specific class
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Get the gradients from the target layer
        guided_gradients = self.gradients[0].cpu().data.numpy()
        
        # Get the activations of the target layer
        target_activations = self.target_layer.output[0].cpu().data.numpy()
        
        # Weighted sum of the target activations
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        cam = np.ones(target_activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * target_activations[i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

    def visualize_cam(self, cam, original_image, alpha=0.6):
        # Resize cam to 48x48
        cam = cv2.resize(cam, (48, 48))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(original_image)
        cam = cam / np.max(cam)
        plt.imshow(np.uint8(255 * cam))
        plt.show()

# Usage example
model.eval()
grad_cam = GradCAM(model, model.layer4)  # Use the last layer for visualization
input_image = torch.randn(1, 1, 48, 48)  # Example tensor, replace with actual data
cam = grad_cam.generate_cam(input_image)
original_image = input_image[0].detach().cpu().numpy()[0]  # Convert your input image to displayable format
grad_cam.visualize_cam(cam, original_image)
