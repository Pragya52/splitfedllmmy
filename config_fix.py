#!/usr/bin/env python3
"""Fix vocabulary size mismatch in config"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import Config, ModelConfig, TrainingConfig, GaLoreConfig, FederatedConfig, KnowledgeDistillationConfig

def create_fixed_config():
    """Create configuration with correct vocabulary size"""
    
    # DialoGPT-small has vocab_size=50257
    return Config(
        model=ModelConfig(
            model_name="llama-7b",
            vocab_size=50257,  # Fixed: Match DialoGPT tokenizer
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            max_position_embeddings=2048
        ),
        training=TrainingConfig(
            batch_size=1,  # Reduced for debugging
            learning_rate=1e-4,
            num_epochs=1,
            max_seq_length=256  # Reduced for debugging
        ),
        galore=GaLoreConfig(
            rank=64,  # Reduced for debugging
            update_proj_gap=50,
            scale=0.25
        ),
        federated=FederatedConfig(
            num_clients=2,  # Reduced for debugging
            num_rounds=2,   # Reduced for debugging
            local_epochs=1
        ),
        kd=KnowledgeDistillationConfig(
            temperature=3.0,
            alpha=0.5
        )
    )

def save_fixed_config():
    """Save fixed config to YAML file"""
    config = create_fixed_config()
    
    import yaml
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    os.makedirs("configs", exist_ok=True)
    with open("configs/fixed_config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print("✅ Saved fixed config to configs/fixed_config.yaml")
    print(f"✅ Vocabulary size set to: {config.model.vocab_size}")
    print("✅ Reduced batch size and dimensions for debugging")

if __name__ == "__main__":
    save_fixed_config()
