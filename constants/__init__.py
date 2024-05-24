image_size = 224
patch_size = 16
num_classes = 7
embed_dim = 768
T = 3  # Temperature for distillation
lambda_coeff = 0.1  # Coefficient for distillation loss
num_epochs = 100  # Number of epochs for training
batch_size = 128  # Batch size for training
lr = 0.0005 * (batch_size / 512)  # Learning rate (scaled by batch size)
weight_decay = 0.05
label_smoothing = 0.1