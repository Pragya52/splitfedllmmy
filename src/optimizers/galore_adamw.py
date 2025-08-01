import torch
import math
from torch.optim import Optimizer
from .galore_projector import GaLoreProjector

class GaLoreAdamW(Optimizer):
    """AdamW optimizer with GaLore gradient projection"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, rank=1024, update_proj_gap=500, scale=0.25):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.projector = GaLoreProjector(rank, update_proj_gap, scale)
        self.param_names = {}
        
        # Map parameters to names
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'param_name'):
                    self.param_names[id(p)] = p.param_name
                else:
                    self.param_names[id(p)] = f"param_{id(p)}"
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step with gradient projection"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Collect gradients for projection update
        gradients = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_name = self.param_names[id(p)]
                    gradients[param_name] = p.grad.data.clone()
        
        # Update projection matrices if needed
        self.projector.update_projection_matrices(gradients)
        
        # Perform optimization with projected gradients
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                param_name = self.param_names[id(p)]
                
                # Project gradient
                proj_grad = self.projector.project_gradient(p.grad.data, param_name)
                proj_grad = proj_grad * self.projector.scale
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(proj_grad)
                    state['exp_avg_sq'] = torch.zeros_like(proj_grad)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(proj_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(proj_grad, proj_grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                # Update parameters
                update = exp_avg / denom
                
                # Unproject the update back to original parameter space
                update = self.projector.unproject_gradient(update, param_name)
                
                p.data.add_(update, alpha=-step_size)
        
        return loss
