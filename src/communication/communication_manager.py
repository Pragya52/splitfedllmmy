import logging
from typing import List
from ..models.llama_client import FederatedClient
from ..models.llama_server import FederatedServer

logger = logging.getLogger(__name__)

class CommunicationManager:
    """Handles communication between clients and server"""
    
    def __init__(self, config):
        self.config = config
        self.server = None
        self.clients = []
        
    def setup_server(self, tokenizer, device='cuda'):
        """Setup federated server"""
        self.server = FederatedServer(self.config, tokenizer, device)
        logger.info("Server setup completed")
        
    def setup_clients(self, tokenizer, device='cuda'):
        """Setup federated clients"""
        self.clients = []
        for client_id in range(self.config.federated.num_clients):
            client = FederatedClient(client_id, self.config, tokenizer, device)
            self.clients.append(client)
        logger.info(f"Setup {len(self.clients)} clients")
    
    def federated_round(self, round_num: int):
        """Execute one federated learning round"""
        logger.info(f"Starting federated round {round_num}")
        
        # Client training
        client_parameters = []
        for client in self.clients:
            logger.info(f"Training client {client.client_id}")
            
            # Train client
            client_loss = client.train_round(self.server)
            
            # Get client parameters
            params = client.get_parameters()
            client_parameters.append(params)
        
        # Server aggregation
        logger.info("Aggregating client parameters")
        aggregated_params = self.server.aggregate_client_parameters(client_parameters)
        
        # Update clients with aggregated parameters
        for client in self.clients:
            client.set_parameters(aggregated_params)
        
        # Collect metrics
        client_losses = [client.round_losses[-1] for client in self.clients if client.round_losses]
        avg_client_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
        
        logger.info(f"Round {round_num} completed. Average client loss: {avg_client_loss:.4f}")
        
        return {
            'round': round_num,
            'avg_client_loss': avg_client_loss,
            'client_losses': client_losses,
            'server_losses': self.server.server_losses[-1] if self.server.server_losses else 0.0
        }
    
    def run_federated_training(self, device='cuda'):
        """Run complete federated training"""
        # Create tokenizer
        tokenizer = self._create_tokenizer()
        
        # Setup server and clients
        self.setup_server(tokenizer, device)
        self.setup_clients(tokenizer, device)
        
        # Run federated rounds
        results = []
        for round_num in range(self.config.federated.num_rounds):
            round_result = self.federated_round(round_num)
            results.append(round_result)
        
        logger.info("Federated training completed!")
        return results
    
    def _create_tokenizer(self):
        """Create tokenizer"""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except:
            logger.warning("Could not load real tokenizer, using dummy tokenizer")
            return self._create_dummy_tokenizer()
    
    def _create_dummy_tokenizer(self):
        """Create dummy tokenizer for testing"""
        import torch
        
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 32000
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
                self.pad_token_id = 0
                self.eos_token_id = 1
            
            def __call__(self, text, **kwargs):
                max_length = kwargs.get('max_length', 512)
                tokens = text.split()[:max_length-2]
                
                input_ids = [2] + [hash(token) % (self.vocab_size-10) + 10 for token in tokens] + [1]
                
                if len(input_ids) < max_length:
                    input_ids += [0] * (max_length - len(input_ids))
                else:
                    input_ids = input_ids[:max_length]
                
                attention_mask = [1 if id != 0 else 0 for id in input_ids]
                
                return {
                    'input_ids': torch.tensor([input_ids]),
                    'attention_mask': torch.tensor([attention_mask])
                }
        
        return DummyTokenizer()
