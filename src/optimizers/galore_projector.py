import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class GaLoreProjector:
    """GaLore gradient projection for memory-efficient training"""
    
    def __init__(self, rank: int = 1024, update_proj_gap: int = 500, scale: float = 0.25):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.step = 0
        self.proj_matrices = {}
        
    def project_gradient(self, grad: torch.Tensor, param_name: str) -> torch.Tensor:
        """Project gradient to low-rank subspace"""
        if param_name not in self.proj_matrices:
            self._init_projection_matrix(grad, param_name)
        
        if grad.dim() != 2:
            return grad  # Don't project 1D tensors
            
        P = self.proj_matrices[param_name]['P']
        Q = self.proj_matrices[param_name]['Q']
        
        # Project gradient: R = P^T @ G @ Q
        proj_grad = P.T @ grad @ Q
        return proj_grad
    
    def unproject_gradient(self, proj_grad: torch.Tensor, param_name: str) -> torch.Tensor:
        """Unproject gradient back to original space"""
        if param_name not in self.proj_matrices or proj_grad.dim() != 2:
            return proj_grad
            
        P = self.proj_matrices[param_name]['P']
        Q = self.proj_matrices[param_name]['Q']
        
        # Unproject: G = P @ R @ Q^T
        grad = P @ proj_grad @ Q.T
        return grad
    
    def _init_projection_matrix(self, grad: torch.Tensor, param_name: str):
        """Initialize projection matrices P and Q"""
        if grad.dim() != 2:
            return
            
        m, n = grad.shape
        rank = min(self.rank, min(m, n))
        
        # Use only one projection matrix for memory efficiency
        if m <= n:
            # Use P only
            U, _, _ = torch.svd(grad)
            P = U[:, :rank].clone().detach()
            Q = torch.eye(n, device=grad.device)
        else:
            # Use Q only  
            P = torch.eye(m, device=grad.device)
            _, _, V = torch.svd(grad)
            Q = V[:, :rank].clone().detach()
            
        self.proj_matrices[param_name] = {'P': P, 'Q': Q}
    
    def update_projection_matrices(self, gradients: Dict[str, torch.Tensor]):
        """Update projection matrices periodically"""
        self.step += 1
        if self.step % self.update_proj_gap == 0:
            logger.info(f"Updating projection matrices at step {self.step}")
            for param_name, grad in gradients.items():
                if grad.dim() == 2:
                    self._init_projection_matrix(grad, param_name)
