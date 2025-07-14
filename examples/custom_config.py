#!/usr/bin/env python3
"""Advanced configuration example for SplitFedLLM"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import (Config, ModelConfig, TrainingConfig, 
                             GaLoreConfig, FederatedConfig, KnowledgeDistillationConfig)
from src.communication.communication_manager import CommunicationManager

def main():
    print("🔧 SplitFedLLM Custom Configuration Example")
    
    # Custom configuration for larger scale
    config = Config(
        model=ModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32
        ),
        training=TrainingConfig(
            batch_size=8,          # Larger batch size
            learning_rate=5e-5,    # Lower learning rate
            max_seq_length=1024    # Longer sequences
        ),
        federated=FederatedConfig(
            num_clients=5,         # More clients
            num_rounds=20,         # More rounds
            local_epochs=2         # More local training
        ),
        galore=GaLoreConfig(
            rank=2048,             # Higher rank for better approximation
            update_proj_gap=200,   # More frequent updates
            scale=0.5              # Different scaling
        ),
        kd=KnowledgeDistillationConfig(
            temperature=4.0,       # Higher temperature
            alpha=0.7              # More emphasis on distillation
        )
    )
    
    print(f"Custom config: {config.federated.num_clients} clients, {config.federated.num_rounds} rounds")
    print(f"GaLore rank: {config.galore.rank}, Temperature: {config.kd.temperature}")
    
    # Run training with custom config
    comm_manager = CommunicationManager(config)
    results = comm_manager.run_federated_training(device='cuda')
    
    print(f"Final result: {results[-1]['avg_client_loss']:.4f}")

if __name__ == "__main__":
    main()
