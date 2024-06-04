
#for both the ResNeXt and Deit models
image_size = 224
num_classes = 7
batch_size = 128
lr = 1e-3 # 0.1 for DeiT
num_epochs = 150
step_size = 30
gamma = 0.1
weight_decay = 0.03 #0.05 for DeiT
label_smoothing = 0.1

# DeiT specific
patch_size = 16
embed_dim = 768
T = 3  # Temperature for distillation
lambda_coeff = 0.1  # Coefficient for distillation loss
num_heads = 12
num_layers = 12
