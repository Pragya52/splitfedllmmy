import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import logging
from dataclasses import asdict

from .llama_layers import LlamaLayerRange
from ..optimizers.galore_adamw import GaLoreAdamW
from .quantization import QuantizedCommunication

logger = logging.getLogger(__name__)

class FederatedServer:
    """Federated Learning Server with quantized communication and middle LLaMA layers"""
    
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
        
        # ✅ NEW: Initialize quantized communication
        quantization_k = getattr(config.quantization, 'k', 10.0) if hasattr(config, 'quantization') else 10.0
        self.quantizer = QuantizedCommunication(k=quantization_k, auto_calibrate=True)
        
        # Move to device
        self.layers_3_30.to(device)
        self.layers_31_32_copy.to(device)
        self.lm_head_copy.to(device)
        self.quantizer.to(device)  # ✅ NEW: Move quantizer to device
        
        # ✅ MODIFIED: GaLore optimizer for layers 3-30 only
        self.optimizer = GaLoreAdamW(
            self.layers_3_30.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            rank=config.galore.rank,
            update_proj_gap=config.galore.update_proj_gap,
            scale=config.galore.scale
        )
        
        # Regular optimizer for teacher model + quantizer
        teacher_params = [
            {'params': self.layers_31_32_copy.parameters()},
            {'params': self.lm_head_copy.parameters()}
        ]
        
        # ✅ NEW: Add quantizer parameters if quantization is enabled
        quantization_enabled = getattr(config, 'quantization', None) and getattr(config.quantization, 'enabled', False)
        if quantization_enabled:
            teacher_params.append({'params': self.quantizer.parameters()})
        
        self.teacher_optimizer = torch.optim.AdamW(
            teacher_params, 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay
        )
        
        # Client parameters storage
        self.client_parameters = {}
        self.aggregated_parameters = {}
        
        # Metrics
        self.server_losses = []
        self.quantization_mse = []  # ✅ NEW: Track quantization errors
        
    def _convert_to_llama_config(self, model_config):
        """Convert ModelConfig to LlamaConfig-like object"""
        class LlamaConfigLike:
            def __init__(self, model_config):
                if isinstance(model_config, dict):
                    for key, value in model_config.items():
                        setattr(self, key, value)
                else:
                    for key, value in model_config.__dict__.items():
                        setattr(self, key, value)
        return LlamaConfigLike(model_config)
    
    def process_quantized_hidden_states(self, client_id: int, quantized_hidden_states: torch.Tensor, 
                                      attention_mask: torch.Tensor, position_ids: torch.Tensor = None):
        """
        Process quantized hidden states from client and return quantized processed states + soft targets
        
        Args:
            client_id: ID of the client sending data
            quantized_hidden_states: Already quantized hidden states from client
            attention_mask: Attention mask for the sequence
            position_ids: Position IDs for the sequence
            
        Returns:
            Tuple of (quantized_processed_states, soft_targets)
        """
        quantized_hidden_states = quantized_hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)
        
        self.layers_3_30.train()
        self.layers_31_32_copy.train()
        self.lm_head_copy.train()
        self.quantizer.train()  # ✅ NEW: Set quantizer to training mode
        
        # ✅ NEW: Store original quantized input for MSE calculation
        original_quantized_input = quantized_hidden_states.detach().clone()
        
        # Step 1: Process quantized hidden states through server layers (3-30)
        processed_states = self.layers_3_30(quantized_hidden_states, attention_mask, position_ids)
        
        # Step 2: Store original processed states for soft target generation
        original_processed_states = processed_states.detach().clone()
        
        # Step 3: Quantize processed states before sending back to client
        quantized_processed_states = self.quantizer.quantize_server_to_client(processed_states)
        
        # ✅ NEW: Calculate quantization error for monitoring
        quantization_mse = F.mse_loss(original_processed_states, quantized_processed_states.detach())
        self.quantization_mse.append(quantization_mse.item())
        
        # Step 4: Generate soft targets using the original (non-quantized) processed states
        with torch.no_grad():
            ensemble_final = self.layers_31_32_copy(original_processed_states, attention_mask, position_ids)
            teacher_logits = self.lm_head_copy(ensemble_final)
            
            # Generate soft targets
            temperature = self.config.kd.temperature
            soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        
        logger.debug(
            f"Server processed Client {client_id}: "
            f"Input shape: {quantized_hidden_states.shape}, "
            f"Output shape: {quantized_processed_states.shape}, "
            f"Quantization MSE: {quantization_mse.item():.6f}"
        )
        
        return quantized_processed_states, soft_targets
    
    def process_hidden_states(self, client_id: int, hidden_states: torch.Tensor, 
                            attention_mask: torch.Tensor, position_ids: torch.Tensor = None):
        """
        Process hidden states from client (standard non-quantized version)
        """
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)
        
        self.layers_3_30.train()
        self.layers_31_32_copy.train()
        self.lm_head_copy.train()
        
        # Check if quantization is enabled
        quantization_enabled = getattr(self.config, 'quantization', None) and getattr(self.config.quantization, 'enabled', False)
        
        if quantization_enabled:
            # If quantization is enabled, quantize first then process
            quantized_hidden_states = self.quantizer.quantize_client_to_server(hidden_states)
            return self.process_quantized_hidden_states(client_id, quantized_hidden_states, attention_mask, position_ids)
        else:
            # Standard processing without quantization
            processed_states = self.layers_3_30(hidden_states, attention_mask, position_ids)
            
            # Generate soft targets
            with torch.no_grad():
                ensemble_final = self.layers_31_32_copy(processed_states, attention_mask, position_ids)
                teacher_logits = self.lm_head_copy(ensemble_final)
                
                temperature = self.config.kd.temperature
                soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
            
            return processed_states.detach(), soft_targets.detach()
    
    def aggregate_client_parameters(self, client_parameters_list: List[Dict]):
        """Aggregate client parameters using FedAvg (including quantizer parameters)"""
        if not client_parameters_list:
            return
        
        # Initialize aggregated parameters
        aggregated = {}
        
        # ✅ MODIFIED: Include quantizer in aggregation if present
        components = ['embedding', 'layers_1_2', 'layers_31_32', 'lm_head']
        
        # Check if any client has quantizer parameters
        has_quantizer = any('quantizer' in client_params for client_params in client_parameters_list)
        if has_quantizer:
            components.append('quantizer')
        
        for component in components:
            aggregated[component] = {}
            
            # Get parameter names from first client that has this component
            first_client_with_component = None
            for client_params in client_parameters_list:
                if component in client_params:
                    first_client_with_component = client_params
                    break
            
            if first_client_with_component is None:
                continue
                
            for param_name in first_client_with_component[component].keys():
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
        
        # ✅ NEW: Update server's quantizer with aggregated parameters
        if 'quantizer' in aggregated and aggregated['quantizer']:
            try:
                server_quantizer_state = {}
                for key, value in aggregated['quantizer'].items():
                    server_quantizer_state[key] = value.to(self.device)
                self.quantizer.load_state_dict(server_quantizer_state, strict=False)
                logger.info("Server quantizer updated with aggregated client parameters")
            except Exception as e:
                logger.warning(f"Could not update server quantizer: {e}")
        
        component_info = f"(including quantizers)" if has_quantizer else ""
        logger.info(f"Aggregated parameters from {len(client_parameters_list)} clients {component_info}")
        
        return aggregated
    
    def get_global_parameters(self):
        """Get global parameters to send to clients (including server's quantizer)"""
        global_params = self.aggregated_parameters.copy()
        
        # ✅ NEW: Include server's quantizer state in global parameters
        quantization_enabled = getattr(self.config, 'quantization', None) and getattr(self.config.quantization, 'enabled', False)
        if quantization_enabled:
            try:
                global_params['server_quantizer'] = {
                    k: v.cpu().clone() for k, v in self.quantizer.state_dict().items()
                }
            except Exception as e:
                logger.warning(f"Could not include server quantizer in global parameters: {e}")
        
        return global_params
    
    def train_teacher_model(self, batch_data):
        """
        Optional: Train the teacher model (layers 31-32 + lm_head) on server
        """
        self.teacher_optimizer.zero_grad()
        
        # Process batch through server pipeline
        input_ids = batch_data['input_ids'].to(self.device)
        attention_mask = batch_data['attention_mask'].to(self.device)
        labels = batch_data['labels'].to(self.device)
        position_ids = torch.arange(0, input_ids.shape[1], device=self.device).unsqueeze(0)
        
        # Simulate client-side processing (for teacher training)
        with torch.no_grad():
            simulated_hidden_states = torch.randn(
                input_ids.shape[0], input_ids.shape[1], self.config.model.hidden_size,
                device=self.device
            )
        
        # Process through server layers
        processed_states = self.layers_3_30(simulated_hidden_states, attention_mask, position_ids)
        
        # Teacher forward pass
        teacher_final = self.layers_31_32_copy(processed_states, attention_mask, position_ids)
        teacher_logits = self.lm_head_copy(teacher_final)
        
        # Teacher loss
        teacher_loss = F.cross_entropy(
            teacher_logits.view(-1, teacher_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        teacher_loss.backward()
        
        # Gradient clipping for teacher
        clip_params = list(self.layers_31_32_copy.parameters()) + list(self.lm_head_copy.parameters())
        
        # Include quantizer parameters if quantization is enabled
        quantization_enabled = getattr(self.config, 'quantization', None) and getattr(self.config.quantization, 'enabled', False)
        if quantization_enabled:
            clip_params.extend(list(self.quantizer.parameters()))
        
        torch.nn.utils.clip_grad_norm_(
            clip_params,
            self.config.training.gradient_clipping
        )
        
        self.teacher_optimizer.step()
        
        self.server_losses.append(teacher_loss.item())
        
        return teacher_loss.item()
    
    # ✅ NEW: Method to get quantization statistics
    def get_quantization_stats(self):
        """Get server-side quantization statistics for analysis"""
        quantization_enabled = getattr(self.config, 'quantization', None) and getattr(self.config.quantization, 'enabled', False)
        
        if not quantization_enabled:
            return {
                'quantization_enabled': False,
                'avg_quantization_mse': 0.0,
                'total_processed_batches': 0
            }
        
        avg_quantization_mse = sum(self.quantization_mse) / len(self.quantization_mse) if self.quantization_mse else 0
        
        return {
            'quantization_enabled': True,
            'avg_quantization_mse': avg_quantization_mse,
            'client_to_server_scale': self.quantizer.client_to_server_quantizer.scale.item(),
            'client_to_server_zero_point': self.quantizer.client_to_server_quantizer.zero_point.item(),
            'server_to_client_scale': self.quantizer.server_to_client_quantizer.scale.item(),
            'server_to_client_zero_point': self.quantizer.server_to_client_quantizer.zero_point.item(),
            'total_processed_batches': len(self.quantization_mse)
        }
    
    # ✅ NEW: Method to reset quantization statistics
    def reset_quantization_stats(self):
        """Reset quantization statistics for new round"""
        self.quantization_mse = []
    
    # ✅ NEW: Method to log quantization summary
    def log_quantization_summary(self):
        """Log summary of quantization performance"""
        stats = self.get_quantization_stats()
        
        if not stats['quantization_enabled']:
            logger.info("=== Standard (Non-Quantized) Server Summary ===")
            return
            
        logger.info("=== Server Quantization Summary ===")
        logger.info(f"Average Quantization MSE: {stats['avg_quantization_mse']:.6f}")
        logger.info(f"Client->Server Scale: {stats['client_to_server_scale']:.6f}")
        logger.info(f"Client->Server Zero Point: {stats['client_to_server_zero_point']:.6f}")
        logger.info(f"Server->Client Scale: {stats['server_to_client_scale']:.6f}")
        logger.info(f"Server->Client Zero Point: {stats['server_to_client_zero_point']:.6f}")
        logger.info(f"Total Processed Batches: {stats['total_processed_batches']}")
        logger.info("===================================")
