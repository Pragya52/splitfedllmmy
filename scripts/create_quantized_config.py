#!/usr/bin/env python3
"""Create quantized federated learning configuration with 10 clients and 50 rounds"""

import sys
import os
import yaml
from dataclasses import asdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import (Config, ModelConfig, TrainingConfig, GaLoreConfig, 
                             FederatedConfig, KnowledgeDistillationConfig, QuantizationConfig)

def create_quantized_config_10_clients():
    """Create configuration with 10 clients, 50 rounds, and quantization support"""
    
    print("🔧 Creating quantized federated learning configuration...")
    print("📊 Settings: 10 clients, 50 rounds, quantization enabled")
    
    # Use smaller model dimensions for feasibility with 10 clients
    config = Config(
        model=ModelConfig(
            model_name="llama-7b",
            vocab_size=50257,  # Match DialoGPT tokenizer
            hidden_size=512,   # Manageable size for 10 clients (512/8 = 64 head_dim)
            intermediate_size=1024,  # 2x hidden_size
            num_hidden_layers=4,     # Reasonable depth for federated learning
            num_attention_heads=8,   # Clean division: 512/8 = 64
            num_key_value_heads=8,   # Same as attention heads
            max_position_embeddings=512,  # Match max_seq_length
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_dropout=0.1   # Slight dropout for regularization
        ),
        training=TrainingConfig(
            batch_size=2,      # Slightly larger batch for 10 clients
            learning_rate=5e-5,  # Lower learning rate for stability
            num_epochs=1,
            max_seq_length=128,  # Reasonable sequence length
            warmup_steps=20,     # More warmup steps
            weight_decay=0.01,
            gradient_clipping=1.0
        ),
        galore=GaLoreConfig(
            rank=32,           # Balanced rank for efficiency
            update_proj_gap=10,  # More frequent updates
            scale=0.25,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"]
        ),
        federated=FederatedConfig(
            num_clients=10,     # ✅ YOUR REQUIREMENT: 10 clients
            num_rounds=50,      # ✅ YOUR REQUIREMENT: 50 rounds
            client_fraction=1.0,  # All clients participate
            local_epochs=1,     # Keep local epochs at 1 for stability
            server_address="localhost",
            server_port=8080
        ),
        kd=KnowledgeDistillationConfig(
            temperature=4.0,    # Slightly higher temperature for softer targets
            alpha=0.4          # Balanced distillation weight
        ),
        quantization=QuantizationConfig(  # ✅ QUANTIZATION SUPPORT
            enabled=True,
            k=12.0,            # Slightly higher sharpness for 10 clients
            learnable_scale=True,
            auto_calibrate=True,
            privacy_noise_std=0.015,  # Slightly more noise for more clients
            log_quantization_stats=True
        )
    )
    
    # Verify configuration
    head_dim = config.model.hidden_size // config.model.num_attention_heads
    print(f"\n📊 Configuration verification:")
    print(f"   Model: {config.model.model_name}")
    print(f"   Hidden size: {config.model.hidden_size}")
    print(f"   Attention heads: {config.model.num_attention_heads}")
    print(f"   Head dimension: {head_dim}")
    print(f"   Layers: {config.model.num_hidden_layers}")
    print(f"   Clients: {config.federated.num_clients}")
    print(f"   Rounds: {config.federated.num_rounds}")
    print(f"   Quantization: {'Enabled' if config.quantization.enabled else 'Disabled'}")
    print(f"   Quantization K: {config.quantization.k}")
    
    # Assertions
    assert config.model.hidden_size % config.model.num_attention_heads == 0, \
        f"hidden_size ({config.model.hidden_size}) must be divisible by num_attention_heads ({config.model.num_attention_heads})"
    
    assert head_dim >= 8, f"head_dim ({head_dim}) should be at least 8"
    
    print("✅ Configuration validated successfully!")
    return config

def create_memory_efficient_config():
    """Create a more memory-efficient version for testing"""
    
    print("\n🔧 Creating memory-efficient configuration for testing...")
    
    config = Config(
        model=ModelConfig(
            model_name="llama-7b",
            vocab_size=50257,
            hidden_size=256,   # Smaller for memory efficiency
            intermediate_size=512,
            num_hidden_layers=2,  # Minimal layers
            num_attention_heads=8,  # 256/8 = 32 head_dim
            num_key_value_heads=8,
            max_position_embeddings=256,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=1,      # Minimal batch size
            learning_rate=1e-4,
            num_epochs=1,
            max_seq_length=64,   # Short sequences
            warmup_steps=10,
            weight_decay=0.01,
            gradient_clipping=1.0
        ),
        galore=GaLoreConfig(
            rank=16,           # Small rank
            update_proj_gap=5,
            scale=0.25,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"]
        ),
        federated=FederatedConfig(
            num_clients=10,     # Still 10 clients
            num_rounds=50,      # Still 50 rounds
            client_fraction=1.0,
            local_epochs=1,
            server_address="localhost",
            server_port=8080
        ),
        kd=KnowledgeDistillationConfig(
            temperature=3.0,
            alpha=0.5
        ),
        quantization=QuantizationConfig(
            enabled=True,
            k=10.0,
            learnable_scale=True,
            auto_calibrate=True,
            privacy_noise_std=0.01,
            log_quantization_stats=True
        )
    )
    
    print("✅ Memory-efficient configuration created!")
    return config

def save_configs():
    """Save both configurations to YAML files"""
    
    # Create configs directory
    os.makedirs("configs", exist_ok=True)
    
    # Save main quantized config (10 clients, 50 rounds)
    main_config = create_quantized_config_10_clients()
    main_config_dict = {
        'model': asdict(main_config.model),
        'training': asdict(main_config.training),
        'galore': asdict(main_config.galore),
        'federated': asdict(main_config.federated),
        'kd': asdict(main_config.kd),
        'quantization': asdict(main_config.quantization)
    }
    
    with open("configs/quantized_10clients_50rounds.yaml", "w") as f:
        yaml.dump(main_config_dict, f, default_flow_style=False, indent=2)
    
    print(f"\n💾 Saved main config to: configs/quantized_10clients_50rounds.yaml")
    
    # Save memory-efficient config for testing
    mem_config = create_memory_efficient_config()
    mem_config_dict = {
        'model': asdict(mem_config.model),
        'training': asdict(mem_config.training),
        'galore': asdict(mem_config.galore),
        'federated': asdict(mem_config.federated),
        'kd': asdict(mem_config.kd),
        'quantization': asdict(mem_config.quantization)
    }
    
    with open("configs/quantized_memory_efficient.yaml", "w") as f:
        yaml.dump(mem_config_dict, f, default_flow_style=False, indent=2)
    
    print(f"💾 Saved memory-efficient config to: configs/quantized_memory_efficient.yaml")
    
    return main_config, mem_config

def estimate_memory_requirements(config):
    """Estimate GPU memory requirements"""
    
    # Rough estimation for LLaMA-like model
    hidden_size = config.model.hidden_size
    num_layers = config.model.num_hidden_layers
    vocab_size = config.model.vocab_size
    seq_length = config.training.max_seq_length
    batch_size = config.training.batch_size
    
    # Model parameters (in millions)
    embedding_params = vocab_size * hidden_size / 1e6
    layer_params = num_layers * (
        4 * hidden_size * hidden_size +  # Attention layers
        3 * hidden_size * config.model.intermediate_size  # MLP layers
    ) / 1e6
    
    total_params = embedding_params + layer_params
    
    # Memory estimation (GB) - rough approximation
    # Parameters + gradients + optimizer states + activations
    model_memory = total_params * 4 / 1000  # FP32 parameters in GB
    gradient_memory = model_memory  # Gradients
    optimizer_memory = model_memory * 2  # Adam optimizer states
    activation_memory = batch_size * seq_length * hidden_size * num_layers * 4 / 1e9  # Activations
    
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    print(f"\n📊 Memory Estimation for {config.federated.num_clients} clients:")
    print(f"   Model parameters: {total_params:.1f}M")
    print(f"   Estimated GPU memory per client: {total_memory:.2f} GB")
    print(f"   Model memory: {model_memory:.2f} GB")
    print(f"   Gradient memory: {gradient_memory:.2f} GB") 
    print(f"   Optimizer memory: {optimizer_memory:.2f} GB")
    print(f"   Activation memory: {activation_memory:.2f} GB")
    
    if total_memory > 24:
        print("⚠️  WARNING: Estimated memory > 24GB - consider using memory-efficient config")
    elif total_memory > 12:
        print("⚠️  CAUTION: Estimated memory > 12GB - monitor GPU usage closely")
    else:
        print("✅ Memory requirements look reasonable")
    
    return total_memory

if __name__ == "__main__":
    print("🚀 Creating Quantized Federated Learning Configurations")
    print("=" * 60)
    
    # Save configurations
    main_config, mem_config = save_configs()
    
    # Estimate memory requirements
    print("\n" + "=" * 60)
    print("💻 MEMORY REQUIREMENTS ANALYSIS")
    print("=" * 60)
    
    print("\n1️⃣  MAIN CONFIG (10 clients, 50 rounds):")
    main_memory = estimate_memory_requirements(main_config)
    
    print("\n2️⃣  MEMORY-EFFICIENT CONFIG:")
    mem_memory = estimate_memory_requirements(mem_config)
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDED USAGE")
    print("=" * 60)
    
    if main_memory <= 12:
        print("✅ Use main config - memory requirements are reasonable:")
        print("   python scripts/run_quantized_federated.py --config configs/quantized_10clients_50rounds.yaml")
    else:
        print("⚠️  Main config may require high-end GPU. Consider:")
        
    print("\n🧪 For testing/development:")
    print("   python scripts/run_quantized_federated.py --config configs/quantized_memory_efficient.yaml")
    
    print("\n🔬 For comparison experiments:")
    print("   python scripts/run_quantized_federated.py --config configs/quantized_memory_efficient.yaml --compare_modes")
    
    print("\n📊 For full-scale training:")
    print("   python scripts/run_quantized_federated.py --config configs/quantized_10clients_50rounds.yaml --save_plots")
    
    print(f"\n✅ Configuration files created successfully!")
    print("📁 Next steps:")
    print("   1. Choose appropriate config based on your GPU memory")
    print("   2. Run the quantized training script")
    print("   3. Monitor logs/training/ for results")
