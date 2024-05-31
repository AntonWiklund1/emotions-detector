import torch
from torch_lr_finder import LRFinder
from torch.cuda.amp import autocast, GradScaler

class CustomLRFinder(LRFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()

    def _train_batch(self, train_iter, accumulation_steps, non_blocking_transfer=True):
        self.model.train()
        total_loss = None  # for late initialization

        self.optimizer.zero_grad()
        for i in range(accumulation_steps):
            inputs, labels = next(train_iter)
            inputs, labels = self._move_to_device(
                inputs, labels, non_blocking=non_blocking_transfer
            )

            # Forward pass
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            # Loss should be averaged in each step
            loss /= accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return total_loss.item()
