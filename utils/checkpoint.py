import torch
import os

from model.ResNeXt import resnext50
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import constants

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = constants.lr
weight_decay = constants.weight_decay


def load_checkpoint(filepath, model, optimizer, scheduler, scaler):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    start_step = checkpoint['step']
    return model, optimizer, scheduler, scaler, start_epoch, start_step

def get_checkpoint(filepath, model, optimizer, scheduler, scaler):
    # Load the checkpoint if it exists
    if os.path.exists(filepath):
        print(f"Checkpoint found at {filepath}. Resuming training.")
        model = model
        optimizer = optimizer
        scheduler = scheduler
        scaler = scaler
        model, optimizer, scheduler, scaler, start_epoch, start_step = load_checkpoint(filepath, model, optimizer, scheduler, scaler)
        return model, optimizer, scheduler, scaler, start_epoch, start_step
    else:
        print("No checkpoint found. Starting training from scratch.")
        model = model
        optimizer = optimizer
        scheduler = scheduler
        scaler = scaler
        return model, optimizer, scheduler, scaler, 0, 0