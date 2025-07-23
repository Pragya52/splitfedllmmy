#!/usr/bin/env python3
"""Updated main script to run quantized federated training with enhanced metrics"""

import argparse
import sys
import os
import torch
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.communication.communication_manager import QuantizedCommunicationManager

def create_quantized_sample_config():
    """Create a sample configuration optimized for quantized federated learning"""
    
    # Use the new quantized config creation method
    config = Config.create_quantized_config(num_clients=3, quantization_k=10.0)
    
    print(f"📊 Quantized FL-LLaMA Config Summary:")
    print(f"  Model: {config.model.model_name} ({config.model.hidden_size}d, {config.model.num_hidden_layers} layers)")
    print(f"  Clients: {config.federated.num_clients}")
    print(f"  Rounds: {config.federated.num_rounds}")
    print(f"  Quantization: {'Enabled' if config.quantization.enabled else 'Disabled'}")
    print(f"  Quantization K: {config.quantization.k}")
    print(f"  Learnable Scales: {config.quantization.learnable_scale}")
    print(f"  Privacy Noise: {config.quantization.privacy_noise_std}")
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Quantized FL-LLaMA Federated Training")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--experiment_name", type=str, default="quantized_fl_llama", help="Experiment name")
    parser.add_argument("--save_plots", action="store_true", help="Save training plots")
    parser.add_argument("--disable_quantization", action="store_true", help="Disable quantization for comparison")
    parser.add_argument("--quantization_k", type=float, default=10.0, help="Quantization sharpness parameter")
    parser.add_argument("--privacy_noise", type=float, default=0.01, help="Privacy noise standard deviation")
    parser.add_argument("--compare_modes", action="store_true", help="Compare quantized vs standard training")
    
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
            
            # Override quantization settings from command line
            if args.disable_quantization:
                config.quantization.enabled = False
                logger.info("Quantization disabled via command line")
            else:
                config.quantization.k = args.quantization_k
                config.quantization.privacy_noise_std = args.privacy_noise
                logger.info(f"Quantization settings: k={args.quantization_k}, noise_std={args.privacy_noise}")
                
        except Exception as e:
            logger.warning(f"Failed to load config from {args.config}: {e}")
            logger.info("Falling back to sample quantized configuration")
            config = create_quantized_sample_config()
    else:
        logger.info("No config file provided, using sample quantized configuration")
        config = create_quantized_sample_config()
        
        # Apply command line overrides
        if args.disable_quantization:
            config.quantization.enabled = False
            logger.info("Quantization disabled via command line")
        else:
            config.quantization.k = args.quantization_k
            config.quantization.privacy_noise_std = args.privacy_noise
    
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
    
    # Log quantization status
    if config.quantization.enabled:
        logger.info("🔢 QUANTIZED FEDERATED LEARNING ENABLED")
        logger.info(f"   Quantization method: INT4 with sigmoid-based differentiable quantization")
        logger.info(f"   Sharpness parameter (k): {config.quantization.k}")
        logger.info(f"   Learnable scales: {config.quantization.learnable_scale}")
        logger.info(f"   Auto calibration: {config.quantization.auto_calibrate}")
        logger.info(f"   Privacy noise std: {config.quantization.privacy_noise_std}")
    else:
        logger.info("📊 STANDARD FEDERATED LEARNING (No Quantization)")
    
    # Initialize enhanced metrics tracker
    try:
        from src.utils.enhanced_metrics import QuantizedTrainingMetrics
        metrics = QuantizedTrainingMetrics(track_quantization=config.quantization.enabled)
        logger.info("Enhanced quantized metrics tracker initialized successfully")
    except ImportError as e:
        logger.warning(f"Enhanced quantized metrics not available: {e}")
        try:
            from src.utils.enhanced_metrics import TrainingMetrics
            metrics = TrainingMetrics()
            logger.info("Basic metrics tracker initialized")
        except ImportError as e2:
            logger.warning(f"No metrics tracker available: {e2}")
            metrics = None

    try:
        # Initialize communication manager
        logger.info("Initializing communication manager...")
        comm_manager = QuantizedCommunicationManager(config)
        
        # Run training based on mode
        if args.compare_modes:
            logger.info("Running comparison between quantized and standard training...")
            results = comm_manager.compare_quantized_vs_standard(device)
            
            # Process comparison results
            if metrics and isinstance(results, dict):
                # Process both sets of results
                for mode, mode_results in results.items():
                    logger.info(f"Processing {mode} results for metrics...")
                    for result in mode_results:
                        if hasattr(metrics, 'update_round_metrics'):
                            metrics.update_round_metrics(result, mode_prefix=mode)
            
        else:
            # Run single mode training
            logger.info(f"Starting {'quantized' if config.quantization.enabled else 'standard'} federated training...")
            logger.info(f"Training configuration:")
            logger.info(f"  Clients: {config.federated.num_clients}")
            logger.info(f"  Rounds: {config.federated.num_rounds}")
            logger.info(f"  Local epochs per round: {config.federated.local_epochs}")
            logger.info(f"  Batch size: {config.training.batch_size}")
            logger.info(f"  Learning rate: {config.training.learning_rate}")
            
            # Run federated training
            if config.quantization.enabled:
                results = comm_manager.run_quantized_federated_training(device)
            else:
                results = comm_manager.run_standard_federated_training(device)
        
        # Process results and update metrics
        if metrics and not args.compare_modes:
            for result in results:
                if hasattr(metrics, 'update_round_metrics'):
                    metrics.update_round_metrics(result)
                
                # Update quantization metrics if available
                if config.quantization.enabled and 'server_quantization_mse' in result:
                    if hasattr(metrics, 'update_quantization_metrics'):
                        metrics.update_quantization_metrics(
                            server_mse=result['server_quantization_mse'],
                            client_stats=result.get('client_quantization_stats', [])
                        )
            
            # Print detailed summary
            if hasattr(metrics, 'print_quantized_summary') and config.quantization.enabled:
                metrics.print_quantized_summary()
            elif hasattr(metrics, 'print_detailed_summary'):
                metrics.print_detailed_summary()
            else:
                # Fallback summary
                summary = metrics.get_summary() if hasattr(metrics, 'get_summary') else "No summary available"
                logger.info("Training completed successfully!")
                logger.info(f"Training Summary: {summary}")
            
            # Save plots
            if args.save_plots:
                try:
                    plot_path = f"logs/training/{args.experiment_name}_plots.png"
                    
                    if hasattr(metrics, 'plot_quantized_training_curves') and config.quantization.enabled:
                        logger.info("Generating quantized training plots...")
                        metrics.plot_quantized_training_curves(plot_path, show_individual_clients=True)
                        logger.info(f"Quantized training plots saved to {plot_path}")
                    elif hasattr(metrics, 'plot_training_curves'):
                        logger.info("Generating training plots...")
                        if 'show_individual_clients' in metrics.plot_training_curves.__code__.co_varnames:
                            metrics.plot_training_curves(plot_path, show_individual_clients=True)
                        else:
                            metrics.plot_training_curves(plot_path)
                        logger.info(f"Training plots saved to {plot_path}")
                    else:
                        logger.warning("Plotting not available with current metrics class")
                        
                except Exception as e:
                    logger.error(f"Could not save plots: {e}")
        
        # Manual result processing without metrics (fallback)
        if not metrics:
            if args.compare_modes:
                logger.info("Comparison completed successfully!")
                if isinstance(results, dict):
                    for mode, mode_results in results.items():
                        logger.info(f"{mode.capitalize()} mode: {len(mode_results)} rounds completed")
                        if mode_results:
                            final_loss = mode_results[-1].get('avg_client_loss', 'N/A')
                            logger.info(f"  Final loss: {final_loss}")
            else:
                logger.info("Training completed successfully!")
                logger.info(f"Total rounds completed: {len(results)}")
                
                # Print round-by-round results
                for i, result in enumerate(results):
                    avg_loss = result.get('avg_client_loss', 'N/A')
                    client_losses = result.get('client_losses', [])
                    
                    log_msg = f"Round {i+1}: Avg Loss: {avg_loss:.4f}, Client Losses: {client_losses}"
                    
                    # Add quantization info if available
                    if config.quantization.enabled and 'server_quantization_mse' in result:
                        server_mse = result['server_quantization_mse']
                        log_msg += f", Server Quantization MSE: {server_mse:.6f}"
                    
                    logger.info(log_msg)
                
                # Calculate improvement
                if len(results) >= 2:
                    initial_loss = results[0].get('avg_client_loss', 0)
                    final_loss = results[-1].get('avg_client_loss', 0)
                    improvement = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
                    logger.info(f"Overall improvement: {improvement:.1f}% ({initial_loss:.4f} → {final_loss:.4f})")
        
        # Log quantization summary if enabled
        if config.quantization.enabled and not args.compare_modes and results:
            logger.info("\n" + "="*50)
            logger.info("QUANTIZATION PERFORMANCE SUMMARY")
            logger.info("="*50)
            
            # Calculate average quantization MSE across all rounds
            quantization_mses = [r.get('server_quantization_mse', 0) for r in results if 'server_quantization_mse' in r]
            if quantization_mses:
                avg_quantization_mse = sum(quantization_mses) / len(quantization_mses)
                logger.info(f"Average Server Quantization MSE: {avg_quantization_mse:.6f}")
                logger.info(f"Quantization MSE Range: [{min(quantization_mses):.6f}, {max(quantization_mses):.6f}]")
            
            # Log final quantization parameters from last round
            if results and 'client_quantization_stats' in results[-1]:
                client_stats = results[-1]['client_quantization_stats']
                for i, stats in enumerate(client_stats):
                    if stats.get('quantization_enabled', False):
                        logger.info(f"Client {i} Final Quantization Scales:")
                        logger.info(f"  Client->Server: scale={stats.get('client_to_server_scale', 'N/A'):.6f}")
                        logger.info(f"  Server->Client: scale={stats.get('server_to_client_scale', 'N/A'):.6f}")
            
            logger.info("="*50)
        
        # Save final configuration for reproducibility
        try:
            config_save_path = f"logs/training/{args.experiment_name}_final_config.yaml"
            config.to_yaml(config_save_path)
            logger.info(f"Final configuration saved to {config_save_path}")
        except Exception as e:
            logger.warning(f"Could not save final configuration: {e}")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("For debugging, try running with:")
        logger.error("  CUDA_LAUNCH_BLOCKING=1 python scripts/run_quantized_federated.py --device cpu")
        logger.error("  Or disable quantization with: --disable_quantization")
        raise

def run_quantization_comparison():
    """Standalone function to run quantization comparison"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔬 RUNNING QUANTIZATION COMPARISON EXPERIMENT")
    logger.info("="*60)
    
    # Create base config
    base_config = create_quantized_sample_config()
    
    # Initialize communication manager
    comm_manager = QuantizedCommunicationManager(base_config)
    
    # Run comparison
    results = comm_manager.compare_quantized_vs_standard('cuda')
    
    return results

if __name__ == "__main__":
    # Check for special comparison mode
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        run_quantization_comparison()
    else:
        main()
