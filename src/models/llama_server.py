import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import logging
from dataclasses import asdict

from .llama_layers import LlamaLayerRange
from ..optimizers.galore_adamw import GaLoreAdamW

logger = logging.getLogger(__name__)

class FederatedServer:
    """Federated Learning Server with middle LLaMA layers and GaLore optimization"""
    
    def __init__(self, config, tokenizer, device='cuda'):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        
        # Convert to LlamaConfig for compatibility
        llama_config = self._convert_to_llama_config(config.model)
        
        # Server has: layers 3-30 + copy of layers 31-32 + copy of lm_head (for teacher)
        self.layers_3_30 = LlamaLayerRange(llama_config, 3, 30)
        self.layers_31_32_copy = LlamaLayerRange(llama_config, 31, 32)
        self.lm_head_copy = nn.Linear(config.model.hidden_size, config.model.vocab_size, bias=False)
        
        # Move to device
        self.layers_3_30.to(device)
        self.layers_31_32_copy.to(device)
        self.lm_head_copy.to(device)
        
        # GaLore optimizer for layers 3-30 only
        self.optimizer = GaLoreAdamW(
            self.layers_3_30.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            rank=config.galore.rank,
            update_proj_gap=config.galore.update_proj_gap,
            scale=config.galore.scale
        )
        
        # Regular optimizer for teacher model (layers 31-32 copy + lm_head copy)
        self.teacher_optimizer = torch.optim.AdamW([
            {'params': self.layers_31_32_copy.parameters()},
            {'params': self.lm_head_copy.parameters()}
        ], lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        
        # Client parameters storage
        self.client_parameters = {}
        self.aggregated_parameters = {}
        
        # Metrics
        self.server_losses = []
        
    def _convert_to_llama_config(self, model_config):
        """Convert ModelConfig to LlamaConfig-like object"""
        class LlamaConfigLike:
            def __init__(self, model_config):
                for key, value in asdict(model_config).items():
                    setattr(self, key, value)
        
        return LlamaConfigLike(model_config)
    
    def process_hidden_states(self, client_id: int, hidden_states: torch.Tensor, 
                            attention_mask: torch.Tensor, position_ids: torch.Tensor = None):
        """Process hidden states from client and return processed states + soft targets"""
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)
        
        self.layers_3_30.train()
        self.layers_31_32_copy.train()
        self.lm_head_copy.train()
        
        # Store original hidden states for ensemble
        if not hasattr(self, 'batch_hidden_states'):
            self.batch_hidden_states = []
            self.batch_attention_masks = []
            self.batch_position_ids = []
        
        self.batch_hidden_states.append(hidden_states)
        self.batch_attention_masks.append(attention_mask)
        self.batch_position_ids.append(position_ids)
        
        # Individual processing through layers 3-30
        with torch.no_grad():
            processed_states = self.layers_3_30(hidden_states, attention_mask, position_ids)
        
        # Generate soft targets using ensemble approach
        soft_targets = self._generate_ensemble_soft_targets()
        
        # Extract soft targets for this client
        batch_size = hidden_states.shape[0]
        client_soft_targets = soft_targets[client_id * batch_size:(client_id + 1) * batch_size]
        
        return processed_states.detach(), client_soft_targets.detach()
    
    def _generate_ensemble_soft_targets(self):
        """Generate soft targets using ensemble of all client hidden states"""
        if len(self.batch_hidden_states) < self.config.federated.num_clients:
            # Return dummy soft targets if not all clients have sent data yet
            dummy_shape = (self.batch_hidden_states[0].shape[0], 
                          self.batch_hidden_states[0].shape[1], 
                          self.config.model.vocab_size)
            return torch.randn(dummy_shape, device=self.device)
        
        # Concatenate all client hidden states
        combined_hidden = torch.cat(self.batch_hidden_states, dim=0)
        combined_attention = torch.cat(self.batch_attention_masks, dim=0)
        combined_positions = torch.cat(self.batch_position_ids, dim=0) if self.batch_position_ids[0] is not None else None
        
        # Process through server layers
        ensemble_middle = self.layers_3_30(combined_hidden, combined_attention, combined_positions)
        
        # Complete teacher forward pass
        ensemble_final = self.layers_31_32_copy(ensemble_middle, combined_attention, combined_positions)
        teacher_logits = self.lm_head_copy(ensemble_final)
        
        # Generate soft targets
        temperature = self.config.kd.temperature
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        
        # Clear batch storage
        self.batch_hidden_states = []
        self.batch_attention_masks = []
        self.batch_position_ids = []
        
        return soft_targets
    
    def aggregate_client_parameters(self, client_parameters_list: List[Dict]):
        """Aggregate client parameters using FedAvg"""
        if not client_parameters_list:
            return
        
        # Initialize aggregated parameters
        aggregated = {}
        
        for component in ['embedding', 'layers_1_2', 'layers_31_32', 'lm_head']:
            aggregated[component] = {}
            
            # Get parameter names from first client
            if component in client_parameters_list[0]:
                for param_name in client_parameters_list[0][component].keys():
                    # Average parameters across clients
                    param_sum = None
                    count = 0
                    
                    for client_params in client_parameters_list:
                        if component in client_params and param_name in client_params[component]:
                            if param_sum is None:
                                param_sum = client_params[component][param_name].clone()
                            else:
                                param_sum += client_params[component][param_name]
                            count += 1
                    
                    if count > 0:
                        aggregated[component][param_name] = param_sum / count
        
        self.aggregated_parameters = aggregated
        logger.info(f"Aggregated parameters from {len(client_parameters_list)} clients")
        
        return aggregated
    
    def get_global_parameters(self):
        """Get global parameters to send to clients"""
        return self.aggregated_parameters
