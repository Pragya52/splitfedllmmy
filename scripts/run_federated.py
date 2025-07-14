#!/usr/bin/env python3
"""Main script to run federated training"""

import argparse
import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.metrics import TrainingMetrics
from src.communication.communication_manager import CommunicationManager

def create_sample_config():
    """Create a sample configuration for testing"""
    from src.utils.config import (ModelConfig, TrainingConfig, GaLoreConfig, 
                                 FederatedConfig, KnowledgeDistillationConfig, Config)
    
    return Config(
        model=ModelConfig(
            model_name="llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            max_position_embeddings=2048
        ),
        training=TrainingConfig(
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            max_seq_length=512
        ),
        galore=GaLoreConfig(
            rank=128,  # Reduced for demo
            update_proj_gap=50,
            scale=0.25
        ),
        federated=FederatedConfig(
            num_clients=3,
            num_rounds=5,
            local_epochs=1
        ),
        kd=KnowledgeDistillationConfig(
            temperature=3.0,
            alpha=0.5
        )
    )

def main():
    parser = argparse.ArgumentParser(description="FL-LLaMA Federated Training")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--experiment_name", type=str, default="fl_llama_exp", help="Experiment name")
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
        config = Config.from_yaml(args.config)
    else:
        logger.info("No config file provided, using sample configuration")
        config = create_sample_config()
    
    # Setup device
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize metrics tracker
    metrics = TrainingMetrics()
    
    try:
        # Run federated training
        comm_manager = CommunicationManager(config)
        results = comm_manager.run_federated_training(device)
        
        # Update metrics
        for result in results:
            metrics.update_round_metrics(result)
        
        # Log results
        summary = metrics.get_summary()
        logger.info("Training completed successfully!")
        logger.info(f"Training Summary: {summary}")
        
        # Save plots if requested
        if args.save_plots:
            plot_path = f"logs/training/{args.experiment_name}_plots.png"
            metrics.plot_training_curves(plot_path)
            logger.info(f"Training plots saved to {plot_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    import logging
    main()
