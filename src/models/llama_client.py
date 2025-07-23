import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import asdict
import logging

from .llama_layers import LlamaLayerRange
from ..data.medical_dataset import MedicalQADataset

logger = logging.getLogger(__name__)

class FederatedClient:
    """Federated Learning Client with partial LLaMA model"""
    
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
        
        # Move to device
        self.embedding.to(device)
        self.layers_1_2.to(device)
        self.layers_31_32.to(device)
        self.lm_head.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW([
            {'params': self.embedding.parameters()},
            {'params': self.layers_1_2.parameters()},
            {'params': self.layers_31_32.parameters()},
            {'params': self.lm_head.parameters()}
        ], lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        
        # Dataset
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
        
    def _convert_to_llama_config(self, model_config):
     """Convert ModelConfig to LlamaConfig-like object"""
     class LlamaConfigLike:
        def __init__(self, model_config):
            if isinstance(model_config, dict):
                # Handle dictionary (from YAML)
                for key, value in model_config.items():
                    setattr(self, key, value)
            else:
                # Handle dataclass (from Python)
                for key, value in model_config.__dict__.items():
                    setattr(self, key, value)
    
     return LlamaConfigLike(model_config)
    
    def forward_initial_layers(self, input_ids, attention_mask, position_ids=None):
        """Forward pass through initial layers (embedding + layers 1-2)"""
        # Embedding
        hidden_states = self.embedding(input_ids)
        
        # Layers 1-2
        hidden_states = self.layers_1_2(hidden_states, attention_mask, position_ids)
        
        return hidden_states
    
    def forward_final_layers(self, hidden_states, attention_mask, position_ids=None):
        """Forward pass through final layers (layers 31-32 + lm_head)"""
        # Layers 31-32
        hidden_states = self.layers_31_32(hidden_states, attention_mask, position_ids)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def train_round(self, server_manager):
        """Train for one federated round"""
        self.embedding.train()
        self.layers_1_2.train()
        self.layers_31_32.train()
        self.lm_head.train()
        
        total_loss = 0.0
        num_batches = 0
        
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
                
                # Add noise for privacy
                noise_std = 0.01
                noise = torch.randn_like(hidden_states) * noise_std
                hidden_states_with_noise = hidden_states + noise
                
                # Step 2: Send to server and get processed hidden states + soft targets
                try:
                    processed_states, soft_targets = server_manager.process_hidden_states(
                        self.client_id, hidden_states_with_noise, attention_mask, position_ids
                    )
                except Exception as e:
                    logger.error(f"Client {self.client_id}: Server communication error: {e}")
                    continue
                
                # Step 3: Forward through final layers
                student_logits = self.forward_final_layers(processed_states, attention_mask, position_ids)
                
                # Step 4: Compute losses
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
                
                # Backward pass
                total_loss_batch.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.embedding.parameters()) + 
                    list(self.layers_1_2.parameters()) + 
                    list(self.layers_31_32.parameters()) + 
                    list(self.lm_head.parameters()),
                    self.config.training.gradient_clipping
                )
                
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Client {self.client_id}, Epoch {epoch}, Batch {batch_idx}: "
                        f"Task Loss: {task_loss.item():.4f}, KD Loss: {kd_loss.item():.4f}, "
                        f"Total Loss: {total_loss_batch.item():.4f}"
                    )
        
        avg_loss = total_loss / max(num_batches, 1)
        self.round_losses.append(avg_loss)
        logger.info(f"Client {self.client_id} finished training round with average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def get_parameters(self):
        """Get client parameters for aggregation"""
        params = {}
        params['embedding'] = {k: v.cpu().clone() for k, v in self.embedding.state_dict().items()}
        params['layers_1_2'] = {k: v.cpu().clone() for k, v in self.layers_1_2.state_dict().items()}
        params['layers_31_32'] = {k: v.cpu().clone() for k, v in self.layers_31_32.state_dict().items()}
        params['lm_head'] = {k: v.cpu().clone() for k, v in self.lm_head.state_dict().items()}
        return params
    
    def set_parameters(self, parameters):
        """Set client parameters from aggregation"""
        if 'embedding' in parameters:
            self.embedding.load_state_dict(parameters['embedding'])
        if 'layers_1_2' in parameters:
            self.layers_1_2.load_state_dict(parameters['layers_1_2'])
        if 'layers_31_32' in parameters:
            self.layers_31_32.load_state_dict(parameters['layers_31_32'])
        if 'lm_head' in parameters:
            self.lm_head.load_state_dict(parameters['lm_head'])
