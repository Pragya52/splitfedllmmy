#!/usr/bin/env python3
"""Main script to run federated training with enhanced metrics for multiple clients"""

import argparse
import sys
import os
import torch
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.communication.communication_manager import CommunicationManager

def create_sample_config():
    """Create a sample configuration with RoPE-compatible dimensions for 3 clients"""
    from src.utils.config import (ModelConfig, TrainingConfig, GaLoreConfig, 
                                 FederatedConfig, KnowledgeDistillationConfig, Config)
    
    # Use dimensions that guarantee clean division for RoPE
    # hidden_size must be divisible by num_attention_heads
    # Result: 128/4 = 32 head_dim (clean division)
    config = Config(
        model=ModelConfig(
            model_name="llama-7b",
            vocab_size=50257,  # Match DialoGPT tokenizer
            hidden_size=128,   # 128/4 = 32 head_dim ✅
            intermediate_size=256,  # 2x hidden_size
            num_hidden_layers=2,    # Minimal layers for testing
            num_attention_heads=4,  # Clean division with hidden_size
            num_key_value_heads=4,  # Same as attention heads
            max_position_embeddings=128,  # Match max_seq_length
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_dropout=0.0
        ),
        training=TrainingConfig(
            batch_size=1,  # Start with batch_size=1
            learning_rate=1e-4,
            num_epochs=1,
            max_seq_length=64,  # Small sequence length
            warmup_steps=10,
            weight_decay=0.01,
            gradient_clipping=1.0
        ),
        galore=GaLoreConfig(
            rank=16,  # Small rank for testing
            update_proj_gap=5,
            scale=0.25,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ),
        federated=FederatedConfig(
            num_clients=3,  # CHANGED TO 3 CLIENTS
            num_rounds=5,   # 5 rounds for testing
            client_fraction=1.0,
            local_epochs=1,
            server_address="localhost",
            server_port=8080
        ),
        kd=KnowledgeDistillationConfig(
            temperature=3.0,
            alpha=0.5
        )
    )
    
    # Verify configuration compatibility
    head_dim = config.model.hidden_size // config.model.num_attention_heads
    print(f"📊 Config verification for {config.federated.num_clients} clients:")
    print(f"  hidden_size: {config.model.hidden_size}")
    print(f"  num_attention_heads: {config.model.num_attention_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  max_seq_length: {config.training.max_seq_length}")
    print(f"  vocab_size: {config.model.vocab_size}")
    print(f"  num_clients: {config.federated.num_clients}")
    
    # Ensure clean division
    assert config.model.hidden_size % config.model.num_attention_heads == 0, \
        f"hidden_size ({config.model.hidden_size}) must be divisible by num_attention_heads ({config.model.num_attention_heads})"
    
    # Ensure reasonable head dimension for RoPE
    assert head_dim >= 8, f"head_dim ({head_dim}) should be at least 8 for proper RoPE functionality"
    
    print(f"✅ Configuration validated successfully!")
    return config

def main():
    parser = argparse.ArgumentParser(description="FL-LLaMA Federated Training with Enhanced Metrics")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--experiment_name", type=str, default="fl_llama_3clients", help="Experiment name")
    parser.add_argument("--save_plots", action="store_true", help="Save training plots")
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs("logs/training", exist_ok=True)
    logger = setup_logger(
        __name__, 
        log_file=f"logs/training/{args.experiment_name}.log",
        level=getattr(logging, args.log_level.upper())
    )
    
    # Load configuration
    if args.config:
        try:
            config = Config.from_yaml(args.config)
            logger.info(f"Loaded configuration from {args.config}")
            logger.info(f"Configuration loaded for {config.federated.num_clients} clients")
        except Exception as e:
            logger.warning(f"Failed to load config from {args.config}: {e}")
            logger.info("Falling back to sample configuration")
            config = create_sample_config()
    else:
        logger.info("No config file provided, using sample configuration for 3 clients")
        config = create_sample_config()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Log system info
    if device == 'cuda':
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize ENHANCED metrics tracker
    try:
        from src.utils.enhanced_metrics import TrainingMetrics
        metrics = TrainingMetrics()
        logger.info("Enhanced metrics tracker initialized successfully")
    except ImportError as e:
        logger.warning(f"Enhanced metrics not available: {e}")
        logger.info("Falling back to basic metrics tracking")
        # Try basic metrics as fallback
        try:
            from src.utils.metrics import TrainingMetrics
            metrics = TrainingMetrics()
            logger.info("Basic metrics tracker initialized")
        except ImportError as e2:
            logger.warning(f"No metrics tracker available: {e2}")
            logger.info("Continuing without metrics tracking")
            metrics = None
    
    try:
        # Run federated training
        logger.info("Initializing communication manager...")
        comm_manager = CommunicationManager(config)
        
        logger.info(f"Starting federated training with {config.federated.num_clients} clients...")
        logger.info(f"Training for {config.federated.num_rounds} rounds")
        
        results = comm_manager.run_federated_training(device)
        
        # Update metrics if available
        if metrics:
            for result in results:
                metrics.update_round_metrics(result)
            
            # Print detailed summary with individual client performance
            if hasattr(metrics, 'print_detailed_summary'):
                metrics.print_detailed_summary()
            else:
                # Fallback summary for basic metrics
                summary = metrics.get_summary()
                logger.info("Training completed successfully!")
                logger.info(f"Training Summary: {summary}")
            
            # Save plots with individual client curves
            if args.save_plots:
                try:
                    plot_path = f"logs/training/{args.experiment_name}_plots.png"
                    if hasattr(metrics, 'plot_training_curves'):
                        # Enhanced metrics with individual client support
                        if 'show_individual_clients' in metrics.plot_training_curves.__code__.co_varnames:
                            logger.info("Generating plots with individual client curves...")
                            metrics.plot_training_curves(plot_path, show_individual_clients=True)
                            logger.info(f"Training plots with individual client curves saved to {plot_path}")
                        else:
                            logger.info("Generating basic plots...")
                            metrics.plot_training_curves(plot_path)
                            logger.info(f"Training plots saved to {plot_path}")
                    else:
                        logger.warning("Plotting not available with current metrics class")
                except Exception as e:
                    logger.error(f"Could not save plots: {e}")
                    logger.info("Continuing without plots...")
        else:
            # Fallback: manually print client information
            logger.info("Training completed successfully!")
            logger.info(f"Total rounds completed: {len(results)}")
            
            for i, result in enumerate(results):
                avg_loss = result.get('avg_client_loss', 'N/A')
                client_losses = result.get('client_losses', [])
                server_loss = result.get('server_losses', 'N/A')
                
                logger.info(f"Round {i}: Avg Loss: {avg_loss:.4f}, Client Losses: {client_losses}, Server Loss: {server_loss}")
            
            # Calculate improvement
            if len(results) >= 2:
                initial_loss = results[0].get('avg_client_loss', 0)
                final_loss = results[-1].get('avg_client_loss', 0)
                improvement = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
                logger.info(f"Overall improvement: {improvement:.1f}% ({initial_loss:.4f} → {final_loss:.4f})")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("For debugging, try running with:")
        logger.error("  CUDA_LAUNCH_BLOCKING=1 python scripts/run_federated.py --device cpu")
        raise

if __name__ == "__main__":
    main()
