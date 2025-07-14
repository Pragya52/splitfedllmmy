#!/usr/bin/env python3
"""Basic usage example for SplitFedLLM"""

import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import Config
from src.communication.communication_manager import CommunicationManager
from src.utils.metrics import TrainingMetrics

def main():
    print("🚀 SplitFedLLM Basic Usage Example")
    print("=" * 40)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load configuration
    try:
        config = Config.from_yaml('configs/default_config.yaml')
    except:
        # Fallback to programmatic config
        from src.utils.config import (ModelConfig, TrainingConfig, GaLoreConfig, 
                                     FederatedConfig, KnowledgeDistillationConfig)
        config = Config(
            model=ModelConfig(),
            training=TrainingConfig(batch_size=2),
            galore=GaLoreConfig(rank=128),
            federated=FederatedConfig(num_rounds=3),
            kd=KnowledgeDistillationConfig()
        )
    
    print(f"Configuration: {config.federated.num_clients} clients, {config.federated.num_rounds} rounds")
    
    # Run federated training
    try:
        comm_manager = CommunicationManager(config)
        results = comm_manager.run_federated_training(device=device)
        
        # Display results
        print("\n🎉 Training completed!")
        print(f"Final loss: {results[-1]['avg_client_loss']:.4f}")
        
        # Track metrics
        metrics = TrainingMetrics()
        for result in results:
            metrics.update_round_metrics(result)
        
        summary = metrics.get_summary()
        print(f"Training summary: {summary}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
