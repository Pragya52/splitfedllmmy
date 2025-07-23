# Modified version of your FederatedClient class with quantization integration

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import asdict
import logging

from .llama_layers import LlamaLayerRange
from ..data.medical_dataset import MedicalQADataset
from .quantization import QuantizedCommunication  # Import our quantization module

logger = logging.getLogger(__name__)

class FederatedClient:
    """Federated Learning Client with quantized communication and partial LLaMA model"""
    
    def __init__(self, client_id: int, config, tokenizer, device='cuda'):
        self.client_id = client_id
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        
        # Convert to LlamaConfig for compatibility
        llama_config = self._convert_to_llama_config(config.model)
        
        # Client has: embedding + layers 1-2 + layers 31-32 + lm_head
        self.embedding = nn.Embedding(config.model.vocab_size, config.model.hidden_size)
        self.layers_1_2 = LlamaLayerRange(llama_config, 1, 2)
        self.layers_31_32 = LlamaLayerRange(llama_config, 31, 32)
        self.lm_head = nn.Linear(config.model.hidden_size, config.model.vocab_size, bias=False)
        
        # ✅ NEW: Initialize quantized communication
        quantization_k = getattr(config, 'quantization_k', 10.0)  # Allow config override
        self.quantizer = QuantizedCommunication(k=quantization_k, auto_calibrate=True)
        
        # Move to device
        self.embedding.to(device)
        self.layers_1_2.to(device)
        self.layers_31_32.to(device)
        self.lm_head.to(device)
        self.quantizer.to(device)  # ✅ NEW: Move quantizer to device
        
        # ✅ MODIFIED: Include quantizer parameters in optimizer
        self.optimizer = torch.optim.AdamW([
            {'params': self.embedding.parameters()},
            {'params': self.layers_1_2.parameters()},
            {'params': self.layers_31_32.parameters()},
            {'params': self.lm_head.parameters()},
            {'params': self.quantizer.parameters()}  # ✅ NEW: Optimize quantization scales
        ], lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        
        # Dataset (unchanged)
        self.dataset = MedicalQADataset(
            tokenizer, 
            max_length=config.training.max_seq_length,
            client_id=client_id,
            total_clients=config.federated.num_clients
        )
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=config.training.batch_size, 
            shuffle=True
        )
        
        # Metrics
        self.training_loss = []
        self.round_losses = []
        
        # ✅ NEW: Quantization metrics
        self.quantization_mse = []
        
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
    
    def forward_initial_layers(self, input_ids, attention_mask, position_ids=None):
        """Forward pass through initial layers (embedding + layers 1-2)"""
        hidden_states = self.embedding(input_ids)
        hidden_states = self.layers_1_2(hidden_states, attention_mask, position_ids)
        return hidden_states
    
    def forward_final_layers(self, hidden_states, attention_mask, position_ids=None):
        """Forward pass through final layers (layers 31-32 + lm_head)"""
        hidden_states = self.layers_31_32(hidden_states, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)
        return logits
    
    def train_round(self, server_manager):
        """Train for one federated round with quantized communication"""
        self.embedding.train()
        self.layers_1_2.train()
        self.layers_31_32.train()
        self.lm_head.train()
        self.quantizer.train()  # ✅ NEW: Set quantizer to training mode
        
        total_loss = 0.0
        num_batches = 0
        total_quantization_mse = 0.0  # ✅ NEW: Track quantization error
        
        for epoch in range(self.config.federated.local_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Create position_ids
                position_ids = torch.arange(0, input_ids.shape[1], device=self.device).unsqueeze(0)
                
                # Step 1: Forward through initial layers
                hidden_states = self.forward_initial_layers(input_ids, attention_mask, position_ids)
                
                # ✅ NEW: Step 2: Quantize hidden states before sending to server
                original_hidden_states = hidden_states.detach().clone()  # For MSE calculation
                quantized_hidden_states = self.quantizer.quantize_client_to_server(hidden_states)
                
                # ✅ NEW: Calculate quantization error for monitoring
                quantization_mse = F.mse_loss(original_hidden_states, quantized_hidden_states.detach())
                total_quantization_mse += quantization_mse.item()
                
                # Step 3: Send quantized data to server and get quantized processed states + soft targets
                try:
                    # ✅ MODIFIED: Use new method name for quantized communication
                    processed_states, soft_targets = server_manager.process_quantized_hidden_states(
                        self.client_id, quantized_hidden_states, attention_mask, position_ids
                    )
                except Exception as e:
                    logger.error(f"Client {self.client_id}: Server communication error: {e}")
                    continue
                
                # Step 4: Forward through final layers (processed_states are already quantized by server)
                student_logits = self.forward_final_layers(processed_states, attention_mask, position_ids)
                
                # Step 5: Compute losses
                # Task loss (standard cross-entropy)
                task_loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                # Knowledge distillation loss
                temperature = self.config.kd.temperature
                alpha = self.config.kd.alpha
                
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(soft_targets / temperature, dim=-1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                # Total loss
                total_loss_batch = task_loss + alpha * kd_loss
                
                # Step 6: Backward pass (gradients flow through quantization automatically!)
                total_loss_batch.backward()
                
                # ✅ MODIFIED: Include quantizer parameters in gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.embedding.parameters()) + 
                    list(self.layers_1_2.parameters()) + 
                    list(self.layers_31_32.parameters()) + 
                    list(self.lm_head.parameters()) +
                    list(self.quantizer.parameters()),  # ✅ NEW: Include quantizer gradients
                    self.config.training.gradient_clipping
                )
                
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
                
                # ✅ MODIFIED: Enhanced logging with quantization info
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Client {self.client_id}, Epoch {epoch}, Batch {batch_idx}: "
                        f"Task Loss: {task_loss.item():.4f}, KD Loss: {kd_loss.item():.4f}, "
                        f"Total Loss: {total_loss_batch.item():.4f}, "
                        f"Quantization MSE: {quantization_mse.item():.6f}"  # ✅ NEW
                    )
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_quantization_mse = total_quantization_mse / max(num_batches, 1)
        
        self.round_losses.append(avg_loss)
        self.quantization_mse.append(avg_quantization_mse)  # ✅ NEW: Store quantization metrics
        
        logger.info(
            f"Client {self.client_id} finished training round with average loss: {avg_loss:.4f}, "
            f"average quantization MSE: {avg_quantization_mse:.6f}"
        )
        
        return avg_loss
    
    def get_parameters(self):
        """Get client parameters for aggregation (including quantizer)"""
        params = {}
        params['embedding'] = {k: v.cpu().clone() for k, v in self.embedding.state_dict().items()}
        params['layers_1_2'] = {k: v.cpu().clone() for k, v in self.layers_1_2.state_dict().items()}
        params['layers_31_32'] = {k: v.cpu().clone() for k, v in self.layers_31_32.state_dict().items()}
        params['lm_head'] = {k: v.cpu().clone() for k, v in self.lm_head.state_dict().items()}
        params['quantizer'] = {k: v.cpu().clone() for k, v in self.quantizer.state_dict().items()}  # ✅ NEW
        return params
    
    def set_parameters(self, parameters):
        """Set client parameters from aggregation (including quantizer)"""
        if 'embedding' in parameters:
            self.embedding.load_state_dict(parameters['embedding'])
        if 'layers_1_2' in parameters:
            self.layers_1_2.load_state_dict(parameters['layers_1_2'])
        if 'layers_31_32' in parameters:
            self.layers_31_32.load_state_dict(parameters['layers_31_32'])
        if 'lm_head' in parameters:
            self.lm_head.load_state_dict(parameters['lm_head'])
        if 'quantizer' in parameters:  # ✅ NEW: Load quantizer parameters
            self.quantizer.load_state_dict(parameters['quantizer'])
    
    # ✅ NEW: Method to get quantization statistics
    def get_quantization_stats(self):
        """Get quantization statistics for analysis"""
        return {
            'avg_quantization_mse': sum(self.quantization_mse) / len(self.quantization_mse) if self.quantization_mse else 0,
            'client_to_server_scale': self.quantizer.client_to_server_quantizer.scale.item(),
            'client_to_server_zero_point': self.quantizer.client_to_server_quantizer.zero_point.item(),
            'server_to_client_scale': self.quantizer.server_to_client_quantizer.scale.item(),
            'server_to_client_zero_point': self.quantizer.server_to_client_quantizer.zero_point.item(),
        }
