import torch
from torch.nn import functional as F
import constants

def distillation_loss(student_logits, teacher_logits, true_labels, temperature=constants.T, alpha=constants.lambda_coeff, label_smoothing=constants.label_smoothing):
    # Unpack the logits
    student_class_logits, student_dist_logits = student_logits

    # Compute the soft targets from teacher logits
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_dist_logits / temperature, dim=1)
    
    # Distillation loss
    distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    
    # Label smoothing for cross-entropy loss
    num_classes = student_class_logits.size(1)
    smoothing = label_smoothing
    confidence = 1.0 - smoothing
    true_dist = torch.zeros_like(student_class_logits).scatter_(1, true_labels.unsqueeze(1), confidence)
    true_dist += smoothing / num_classes
    
    # Cross-entropy loss with label smoothing
    ce_loss = torch.mean(torch.sum(-true_dist * F.log_softmax(student_class_logits, dim=-1), dim=-1))
    
    # Combine the two losses
    return alpha * distillation_loss + (1.0 - alpha) * ce_loss
