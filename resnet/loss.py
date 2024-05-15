import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float).to(device)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents overflow
        F_loss = (1 - pt) ** self.gamma * BCE_loss # focal loss
        if self.alpha is not None: # apply class weights
            alpha_t = self.alpha[targets].to(inputs.device)
            F_loss = alpha_t * F_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
