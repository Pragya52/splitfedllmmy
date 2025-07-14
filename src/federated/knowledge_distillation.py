import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class KnowledgeDistillationLoss:
    """Knowledge distillation loss implementation"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha
    
    def __call__(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                 labels: torch.Tensor, ignore_index: int = -100):
        """
        Compute combined knowledge distillation loss
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model (soft targets)
            labels: Ground truth labels
            ignore_index: Index to ignore in loss calculation
        
        Returns:
            Combined loss (task + distillation)
        """
        # Task loss (standard cross-entropy)
        task_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index
        )
        
        # Knowledge distillation loss
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = task_loss + self.alpha * kd_loss
        
        return total_loss, task_loss, kd_loss
