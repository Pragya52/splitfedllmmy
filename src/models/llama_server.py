# Modified version of your FederatedServer class with quantization integration

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import logging
from dataclasses import asdict

from .llama_layers import LlamaLayerRange
from ..optimizers.galore_adamw import GaLoreAdamW
from .quantization import QuantizedCommunication  # Import our quantization module

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
        quantization_k = getattr(config, 'quantization_k', 10.0)  # Allow config override
        self.quantizer = QuantizedCommunication(k=quantization_k, auto_calibrate=True)
        
        # Move to device
        self.layers_3_30.to(device)
        self.layers_31_32_copy.to(device)
        self.lm_head_copy.to(device)
        self.quantizer.to(device)  # ✅ NEW: Move quantizer to device
        
        # ✅ MODIFIED: GaLore optimizer for layers 3-30 + quantizer parameters
        # Separate optimizers because GaLore is only for specific layers
        self.optimizer = GaLoreAdamW(
            self.layers_3_30.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            rank=config.galore.rank,
            update_proj_gap=config.galore.update_proj_gap,
            scale=config.galore.scale
        )
        
        # Regular optimizer for teacher model + quantizer
        self.teacher_optimizer = torch.optim.AdamW([
            {'params': self.layers_31_32_copy.parameters()},
            {'params': self.lm_head_copy.parameters()},
            {'params': self.quantizer.parameters()}  # ✅ NEW: Include quantizer parameters
        ], lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        
        # Client parameters storage
        self.client_parameters = {}
        self.aggregated_parameters = {}
        
        # Metrics
        self.server_losses = []
        self.quantization_mse = []  # ✅ NEW: Track quantization errors
        
    def _convert_to_llama_config(self, model_config):
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
        # This allows gradients to flow back through the quantization process
        processed_states = self.layers_3_30(quantized_hidden_states, attention_mask, position_ids)
        
        # Step 2: Store original processed states for soft target generation
        original_processed_states = processed_states.detach().clone()
        
        # Step 3: Quantize processed states before sending back to client
        quantized_processed_states = self.quantizer.quantize_server_to_client(processed_states)
        
        # ✅ NEW: Calculate quantization error for monitoring
        quantization_mse = F.mse_loss(original_processed_states, quantized_processed_states.detach())
        self.quantization_mse.append(quantization_mse.item())
        
        # Step 4: Generate soft targets using the original (non-quantized) processed states
        # This ensures that the teacher signal is not degraded by quantization
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
        Legacy method for backward compatibility (redirects to quantized version)
        """
        logger.warning("Using legacy process_hidden_states method. Consider updating to process_quantized_hidden_states.")
        
        # First quantize the incoming hidden states
        quantized_hidden_states = self.quantizer.quantize_client_to_server(hidden_states)
        
        # Then process as normal
        return self.process_quantized_hidden_states(client_id, quantized_hidden_states, attention_mask, position_ids)
    
    def aggregate_client_parameters(self, client_parameters_list: List[Dict]):
        """Aggregate client parameters using FedAvg (including quantizer parameters)"""
        if not client_parameters_list:
            return
        
        # Initialize aggregated parameters
        aggregated = {}
        
        # ✅ MODIFIED: Include quantizer in aggregation
        for component in ['embedding', 'layers_1_2', 'layers_31_32', 'lm_head', 'quantizer']:
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
        
        logger.info(f"Aggregated parameters from {len(client_parameters_list)} clients (including quantizers)")
        
        return aggregated
    
    def get_global_parameters(self):
        """Get global parameters to send to clients (including server's quantizer)"""
        global_params = self.aggregated_parameters.copy()
        
        # ✅ NEW: Include server's quantizer state in global parameters
        # This helps synchronize quantization scales across all participants
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
        This can help improve the quality of soft targets
        """
        self.teacher_optimizer.zero_grad()
        
        # Process batch through server pipeline
        input_ids = batch_data['input_ids'].to(self.device)
        attention_mask = batch_data['attention_mask'].to(self.device)
        labels = batch_data['labels'].to(self.device)
        position_ids = torch.arange(0, input_ids.shape[1], device=self.device).unsqueeze(0)
        
        # Simulate client-side processing (for teacher training)
        with torch.no_grad():
            # This would normally come from clients, but for teacher training we simulate it
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
        torch.nn.utils.clip_grad_norm_(
            list(self.layers_31_32_copy.parameters()) + 
            list(self.lm_head_copy.parameters()) +
            list(self.quantizer.parameters()),
            self.config.training.gradient_clipping
        )
        
        self.teacher_optimizer.step()
        
        self.server_losses.append(teacher_loss.item())
        
        return teacher_loss.item()
    
    # ✅ NEW: Method to get quantization statistics
    def get_quantization_stats(self):
        """Get server-side quantization statistics for analysis"""
        avg_quantization_mse = sum(self.quantization_mse) / len(self.quantization_mse) if self.quantization_mse else 0
        
        return {
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
        logger.info("=== Server Quantization Summary ===")
        logger.info(f"Average Quantization MSE: {stats['avg_quantization_mse']:.6f}")
        logger.info(f"Client->Server Scale: {stats['client_to_server_scale']:.6f}")
        logger.info(f"Client->Server Zero Point: {stats['client_to_server_zero_point']:.6f}")
        logger.info(f"Server->Client Scale: {stats['server_to_client_scale']:.6f}")
        logger.info(f"Server->Client Zero Point: {stats['server_to_client_zero_point']:.6f}")
        logger.info(f"Total Processed Batches: {stats['total_processed_batches']}")
        logger.info("===================================")


# ✅ NEW: Enhanced Communication Manager with Quantization Support
class QuantizedCommunicationManager:
    """Enhanced communication manager that handles quantized federated training"""
    
    def __init__(self, config):
        self.config = config
        self.server = None
        self.clients = []
        
    def initialize_server(self, tokenizer, device='cuda'):
        """Initialize the server with quantization support"""
        self.server = FederatedServer(self.config, tokenizer, device)
        logger.info("Quantized federated server initialized")
        
    def initialize_clients(self, tokenizer, device='cuda'):
        """Initialize clients with quantization support"""
        from .llama_client import FederatedClient  # Import the updated client
        
        self.clients = []
        for client_id in range(self.config.federated.num_clients):
            client = FederatedClient(client_id, self.config, tokenizer, device)
            self.clients.append(client)
        
        logger.info(f"Initialized {len(self.clients)} quantized federated clients")
        
    def run_quantized_federated_training(self, device='cuda'):
        """Run federated training with quantization"""
        from transformers import AutoTokenizer
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Initialize server and clients
        self.initialize_server(tokenizer, device)
        self.initialize_clients(tokenizer, device)
        
        results = []
        
        for round_num in range(self.config.federated.num_rounds):
            logger.info(f"=== Quantized Federated Round {round_num + 1}/{self.config.federated.num_rounds} ===")
            
            # Reset server quantization stats for this round
            self.server.reset_quantization_stats()
            
            # Client training with quantized communication
            client_losses = []
            client_parameters = []
            
            for client in self.clients:
                # Train client (this now includes quantized communication)
                client_loss = client.train_round(self.server)
                client_losses.append(client_loss)
                
                # Get client parameters (including quantizer)
                client_params = client.get_parameters()
                client_parameters.append(client_params)
            
            # Server aggregation (including quantizer parameters)
            aggregated_params = self.server.aggregate_client_parameters(client_parameters)
            
            # Update all clients with aggregated parameters
            for client in self.clients:
                client.set_parameters(aggregated_params)
            
            # Log quantization performance
            self.server.log_quantization_summary()
            
            # Collect round results
            avg_client_loss = sum(client_losses) / len(client_losses)
            server_stats = self.server.get_quantization_stats()
            
            round_result = {
                'round': round_num + 1,
                'avg_client_loss': avg_client_loss,
                'client_losses': client_losses,
                'server_quantization_mse': server_stats['avg_quantization_mse'],
                'client_quantization_stats': [client.get_quantization_stats() for client in self.clients]
            }
            
            results.append(round_result)
            
            logger.info(f"Round {round_num + 1} completed: Avg Loss = {avg_client_loss:.4f}, "
                       f"Quantization MSE = {server_stats['avg_quantization_mse']:.6f}")
        
        logger.info("Quantized federated training completed successfully!")
        return results
Convert ModelConfig to LlamaConfig-like object"""
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
