import torch
import logging
from typing import List, Dict
from transformers import AutoTokenizer

from ..models.llama_server import FederatedServer
from ..models.llama_client import FederatedClient

logger = logging.getLogger(__name__)

class CommunicationManager:
    """Enhanced communication manager that handles both standard and quantized federated training"""
    
    def __init__(self, config):
        self.config = config
        self.server = None
        self.clients = []
        
    def initialize_server(self, tokenizer, device='cuda'):
        """Initialize the server"""
        self.server = FederatedServer(self.config, tokenizer, device)
        
        quantization_enabled = getattr(self.config, 'quantization', None) and getattr(self.config.quantization, 'enabled', False)
        server_type = "quantized" if quantization_enabled else "standard"
        logger.info(f"Initialized {server_type} federated server")
        
    def initialize_clients(self, tokenizer, device='cuda'):
        """Initialize clients"""
        self.clients = []
        for client_id in range(self.config.federated.num_clients):
            client = FederatedClient(client_id, self.config, tokenizer, device)
            self.clients.append(client)
        
        quantization_enabled = getattr(self.config, 'quantization', None) and getattr(self.config.quantization, 'enabled', False)
        client_type = "quantized" if quantization_enabled else "standard"
        logger.info(f"Initialized {len(self.clients)} {client_type} federated clients")
        
    def run_federated_training(self, device='cuda'):
        """Run federated training (supports both quantized and standard modes)"""
        # Initialize tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
            
        # Initialize server and clients
        self.initialize_server(tokenizer, device)
        self.initialize_clients(tokenizer, device)
        
        results = []
        
        quantization_enabled = getattr(self.config, 'quantization', None) and getattr(self.config.quantization, 'enabled', False)
        training_mode = "Quantized" if quantization_enabled else "Standard"
        
        for round_num in range(self.config.federated.num_rounds):
            logger.info(f"=== {training_mode} Federated Round {round_num + 1}/{self.config.federated.num_rounds} ===")
            
            # Reset server quantization stats for this round (if applicable)
            if hasattr(self.server, 'reset_quantization_stats'):
                self.server.reset_quantization_stats()
            
            # Client training
            client_losses = []
            client_parameters = []
            
            for client in self.clients:
                # Train client
                client_loss = client.train_round(self.server)
                client_losses.append(client_loss)
                
                # Get client parameters
                client_params = client.get_parameters()
                client_parameters.append(client_params)
            
            # Server aggregation
            aggregated_params = self.server.aggregate_client_parameters(client_parameters)
            
            # Update all clients with aggregated parameters
            for client in self.clients:
                client.set_parameters(aggregated_params)
            
            # Log performance summary
            if hasattr(self.server, 'log_quantization_summary'):
                self.server.log_quantization_summary()
            
            # Collect round results
            avg_client_loss = sum(client_losses) / len(client_losses)
            
            round_result = {
                'round': round_num + 1,
                'avg_client_loss': avg_client_loss,
                'client_losses': client_losses,
            }
            
            # Add quantization metrics if available
            if quantization_enabled:
                if hasattr(self.server, 'get_quantization_stats'):
                    server_stats = self.server.get_quantization_stats()
                    round_result['server_quantization_mse'] = server_stats.get('avg_quantization_mse', 0.0)
                
                # Collect client quantization stats
                client_quantization_stats = []
                for client in self.clients:
                    if hasattr(client, 'get_quantization_stats'):
                        client_stats = client.get_quantization_stats()
                        client_quantization_stats.append(client_stats)
                
                if client_quantization_stats:
                    round_result['client_quantization_stats'] = client_quantization_stats
            
            results.append(round_result)
            
            # Log round summary
            log_msg = f"Round {round_num + 1} completed: Avg Loss = {avg_client_loss:.4f}"
            if quantization_enabled and 'server_quantization_mse' in round_result:
                log_msg += f", Quantization MSE = {round_result['server_quantization_mse']:.6f}"
            logger.info(log_msg)
        
        training_type = "Quantized federated" if quantization_enabled else "Standard federated"
        logger.info(f"{training_type} training completed successfully!")
        return results


# ✅ NEW: Specialized Quantized Communication Manager (for backward compatibility)
class QuantizedCommunicationManager(CommunicationManager):
    """
    Specialized communication manager for quantized federated training
    Inherits from CommunicationManager but ensures quantization is properly configured
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Ensure quantization is configured
        if not hasattr(config, 'quantization') or config.quantization is None:
            from ..utils.config import QuantizationConfig
            config.quantization = QuantizationConfig()
            logger.warning("No quantization config found, using default settings")
    
    def run_quantized_federated_training(self, device='cuda'):
        """
        Run federated training with quantization explicitly enabled
        """
        # Force enable quantization
        self.config.quantization.enabled = True
        
        logger.info("🔢 Starting QUANTIZED federated training...")
        logger.info(f"   Quantization K: {self.config.quantization.k}")
        logger.info(f"   Learnable scales: {self.config.quantization.learnable_scale}")
        logger.info(f"   Privacy noise: {self.config.quantization.privacy_noise_std}")
        
        return self.run_federated_training(device)
    
    def run_standard_federated_training(self, device='cuda'):
        """
        Run federated training with quantization explicitly disabled
        """
        # Force disable quantization
        self.config.quantization.enabled = False
        
        logger.info("📊 Starting STANDARD (non-quantized) federated training...")
        
        return self.run_federated_training(device)
    
    def compare_quantized_vs_standard(self, device='cuda'):
        """
        Run comparison between quantized and standard federated learning
        """
        logger.info("🔬 RUNNING QUANTIZATION COMPARISON EXPERIMENT")
        logger.info("="*60)
        
        results_comparison = {}
        
        # Run with quantization
        logger.info("🔢 Running WITH quantization...")
        results_quantized = self.run_quantized_federated_training(device)
        results_comparison['quantized'] = results_quantized
        
        # Reinitialize for fair comparison
        self.server = None
        self.clients = []
        
        # Run without quantization  
        logger.info("📊 Running WITHOUT quantization...")
        results_standard = self.run_standard_federated_training(device)
        results_comparison['standard'] = results_standard
        
        # Compare results
        logger.info("\n" + "="*60)
        logger.info("QUANTIZATION COMPARISON RESULTS")
        logger.info("="*60)
        
        if results_quantized and results_standard:
            final_loss_quantized = results_quantized[-1]['avg_client_loss']
            final_loss_standard = results_standard[-1]['avg_client_loss']
            
            loss_difference = final_loss_quantized - final_loss_standard
            loss_percentage = (loss_difference / final_loss_standard) * 100 if final_loss_standard > 0 else 0
            
            logger.info(f"Final Loss (Standard):   {final_loss_standard:.4f}")
            logger.info(f"Final Loss (Quantized):  {final_loss_quantized:.4f}")
            logger.info(f"Loss Difference:         {loss_difference:+.4f} ({loss_percentage:+.1f}%)")
            
            if abs(loss_percentage) < 5:
                logger.info("✅ Quantization maintains similar performance (< 5% difference)")
            elif loss_percentage > 0:
                logger.info("⚠️  Quantization slightly reduces performance")
            else:
                logger.info("🎉 Quantization improves performance!")
