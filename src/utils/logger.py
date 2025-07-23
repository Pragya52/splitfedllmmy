import logging
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logger(name: str, log_file: str = None, level=logging.INFO, 
                 include_quantization_filter: bool = False):
    """
    Setup logger with file and console handlers, with optional quantization-aware filtering
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        include_quantization_filter: Add filter for quantization-specific logging
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter with more detailed format for quantization experiments
    if include_quantization_filter:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Add quantization filter if requested
    if include_quantization_filter:
        console_handler.addFilter(QuantizationLogFilter())
    
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        
        # Add quantization filter to file handler too
        if include_quantization_filter:
            file_handler.addFilter(QuantizationLogFilter())
        
        logger.addHandler(file_handler)
        
        # Log the start of a new session
        logger.info(f"="*60)
        logger.info(f"NEW LOGGING SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Log level: {logging.getLevelName(level)}")
        logger.info(f"Quantization filtering: {'Enabled' if include_quantization_filter else 'Disabled'}")
        logger.info(f"="*60)
    
    return logger

class QuantizationLogFilter(logging.Filter):
    """Custom log filter for quantization-related logging"""
    
    def __init__(self):
        super().__init__()
        self.quantization_keywords = [
            'quantization', 'quantized', 'quantizer', 'mse', 'scale', 'sigmoid',
            'int4', 'compression', 'privacy', 'noise'
        ]
    
    def filter(self, record):
        """
        Filter log records to highlight quantization-related messages
        """
        # Always allow non-quantization messages
        message = record.getMessage().lower()
        
        # Check if this is a quantization-related message
        is_quantization_msg = any(keyword in message for keyword in self.quantization_keywords)
        
        if is_quantization_msg:
            # Add a prefix to quantization messages for easy identification
            if not record.getMessage().startswith('🔢'):
                record.msg = f"🔢 {record.msg}"
        
        return True  # Allow all messages through

def setup_quantized_logger(name: str, experiment_name: str = "quantized_experiment", 
                          level=logging.INFO) -> logging.Logger:
    """
    Convenience function to setup a logger specifically for quantized federated learning experiments
    
    Args:
        name: Logger name
        experiment_name: Name of the experiment (used for log file naming)
        level: Logging level
    
    Returns:
        Configured logger
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"logs/training/{experiment_name}_{timestamp}.log"
    
    return setup_logger(
        name=name, 
        log_file=log_file, 
        level=level, 
        include_quantization_filter=True
    )

def setup_comparison_logger(name: str, comparison_name: str = "quantization_comparison", 
                           level=logging.INFO) -> logging.Logger:
    """
    Setup logger for quantization comparison experiments
    
    Args:
        name: Logger name
        comparison_name: Name of the comparison experiment
        level: Logging level
    
    Returns:
        Configured logger
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"logs/comparisons/{comparison_name}_{timestamp}.log"
    
    logger = setup_logger(
        name=name, 
        log_file=log_file, 
        level=level, 
        include_quantization_filter=True
    )
    
    # Add special markers for comparison experiments
    logger.info("🔬 COMPARISON EXPERIMENT LOGGER INITIALIZED")
    logger.info("This log will contain both quantized and standard training results")
    
    return logger

class LoggingContextManager:
    """Context manager for experiment-specific logging"""
    
    def __init__(self, experiment_name: str, quantization_enabled: bool = True):
        self.experiment_name = experiment_name
        self.quantization_enabled = quantization_enabled
        self.logger = None
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        
        if self.quantization_enabled:
            self.logger = setup_quantized_logger("experiment", self.experiment_name)
            self.logger.info(f"🔢 QUANTIZED EXPERIMENT STARTED: {self.experiment_name}")
        else:
            self.logger = setup_logger("experiment", 
                                     f"logs/training/{self.experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log")
            self.logger.info(f"📊 STANDARD EXPERIMENT STARTED: {self.experiment_name}")
        
        self.logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"✅ EXPERIMENT COMPLETED SUCCESSFULLY: {self.experiment_name}")
        else:
            self.logger.error(f"❌ EXPERIMENT FAILED: {self.experiment_name}")
            self.logger.error(f"Error: {exc_val}")
        
        self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Duration: {duration}")
        self.logger.info("="*60)

def log_quantization_config(logger: logging.Logger, config):
    """Log quantization configuration details"""
    if hasattr(config, 'quantization') and config.quantization:
        logger.info("🔢 QUANTIZATION CONFIGURATION:")
        logger.info(f"   Enabled: {config.quantization.enabled}")
        logger.info(f"   Sharpness (k): {config.quantization.k}")
        logger.info(f"   Learnable scales: {config.quantization.learnable_scale}")
        logger.info(f"   Auto calibration: {config.quantization.auto_calibrate}")
        logger.info(f"   Privacy noise std: {config.quantization.privacy_noise_std}")
        logger.info(f"   Log quantization stats: {config.quantization.log_quantization_stats}")
    else:
        logger.info("📊 STANDARD CONFIGURATION (No Quantization)")

def log_system_info(logger: logging.Logger, device: str = "cuda"):
    """Log system information for debugging"""
    import torch
    import platform
    
    logger.info("💻 SYSTEM INFORMATION:")
    logger.info(f"   Platform: {platform.platform()}")
    logger.info(f"   Python version: {platform.python_version()}")
    logger.info(f"   PyTorch version: {torch.__version__}")
    logger.info(f"   Device: {device}")
    
    if device == 'cuda' and torch.cuda.is_available():
        logger.info(f"   CUDA version: {torch.version.cuda}")
        logger.info(f"   GPU: {torch.cuda.get_device_name()}")
        logger.info(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"   CUDA devices: {torch.cuda.device_count()}")
    elif device == 'cuda':
        logger.warning("⚠️  CUDA requested but not available!")

def log_model_info(logger: logging.Logger, config):
    """Log model configuration information"""
    logger.info("🤖 MODEL CONFIGURATION:")
    logger.info(f"   Model name: {config.model.model_name}")
    logger.info(f"   Hidden size: {config.model.hidden_size}")
    logger.info(f"   Attention heads: {config.model.num_attention_heads}")
    logger.info(f"   Layers: {config.model.num_hidden_layers}")
    logger.info(f"   Vocab size: {config.model.vocab_size}")
    logger.info(f"   Max sequence length: {config.training.max_seq_length}")

def log_federated_config(logger: logging.Logger, config):
    """Log federated learning configuration"""
    logger.info("🌐 FEDERATED LEARNING CONFIGURATION:")
    logger.info(f"   Number of clients: {config.federated.num_clients}")
    logger.info(f"   Number of rounds: {config.federated.num_rounds}")
    logger.info(f"   Local epochs: {config.federated.local_epochs}")
    logger.info(f"   Client fraction: {config.federated.client_fraction}")
    logger.info(f"   Batch size: {config.training.batch_size}")
    logger.info(f"   Learning rate: {config.training.learning_rate}")

def log_training_progress(logger: logging.Logger, round_num: int, total_rounds: int, 
                         avg_loss: float, client_losses: list, quantization_mse: float = None):
    """Log training progress for a round"""
    progress_percentage = (round_num / total_rounds) * 100
    
    base_msg = f"📊 Round {round_num}/{total_rounds} ({progress_percentage:.1f}%): "
    base_msg += f"Avg Loss: {avg_loss:.4f}, Client Losses: {[f'{loss:.4f}' for loss in client_losses]}"
    
    if quantization_mse is not None:
        base_msg += f", Quantization MSE: {quantization_mse:.6f}"
    
    logger.info(base_msg)

def log_experiment_results(logger: logging.Logger, results: list, quantization_enabled: bool = False):
    """Log final experiment results"""
    if not results:
        logger.warning("No results to log")
        return
    
    initial_loss = results[0].get('avg_client_loss', 0)
    final_loss = results[-1].get('avg_client_loss', 0)
    improvement = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
    
    logger.info("📈 EXPERIMENT RESULTS SUMMARY:")
    logger.info(f"   Total rounds: {len(results)}")
    logger.info(f"   Initial loss: {initial_loss:.4f}")
    logger.info(f"   Final loss: {final_loss:.4f}")
    logger.info(f"   Improvement: {improvement:.1f}%")
    logger.info(f"   Convergence rate: {improvement/len(results):.3f}% per round")
    
    if quantization_enabled:
        # Log quantization-specific results
        quantization_mses = [r.get('server_quantization_mse', 0) for r in results 
                           if 'server_quantization_mse' in r]
        if quantization_mses:
            avg_mse = sum(quantization_mses) / len(quantization_mses)
            logger.info(f"🔢 Average quantization MSE: {avg_mse:.6f}")
            logger.info(f"🔢 MSE range: [{min(quantization_mses):.6f}, {max(quantization_mses):.6f}]")

# Convenience functions for common logging scenarios
def create_experiment_logger(experiment_name: str, quantization_enabled: bool = False) -> logging.Logger:
    """Create a logger for a specific experiment"""
    if quantization_enabled:
        return setup_quantized_logger("experiment", experiment_name)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/training/{experiment_name}_{timestamp}.log"
        return setup_logger("experiment", log_file)

def create_debug_logger(name: str = "debug") -> logging.Logger:
    """Create a debug logger with verbose output"""
    return setup_logger(name, level=logging.DEBUG, include_quantization_filter=True)

# Example usage functions
def example_quantized_logging():
    """Example of how to use quantized logging"""
    
    # Setup quantized logger
    logger = setup_quantized_logger("example", "test_quantization")
    
    # Log configuration (example)
    class MockConfig:
        class MockQuantization:
            enabled = True
            k = 10.0
            learnable_scale = True
            auto_calibrate = True
            privacy_noise_std = 0.01
            log_quantization_stats = True
        
        class MockModel:
            model_name = "llama-7b"
            hidden_size = 128
            num_attention_heads = 4
            num_hidden_layers = 2
            vocab_size = 50257
        
        class MockTraining:
            max_seq_length = 64
            batch_size = 1
            learning_rate = 1e-4
        
        class MockFederated:
            num_clients = 3
            num_rounds = 5
            local_epochs = 1
            client_fraction = 1.0
        
        quantization = MockQuantization()
        model = MockModel()
        training = MockTraining()
        federated = MockFederated()
    
    config = MockConfig()
    
    # Log system info
    log_system_info(logger, "cuda")
    
    # Log configuration
    log_quantization_config(logger, config)
    log_model_info(logger, config)
    log_federated_config(logger, config)
    
    # Log training progress (example)
    log_training_progress(logger, 1, 5, 2.5, [2.3, 2.7, 2.4], 0.001234)
    
    # Log final results (example)
    example_results = [
        {'avg_client_loss': 3.0, 'server_quantization_mse': 0.002},
        {'avg_client_loss': 2.5, 'server_quantization_mse': 0.0015},
        {'avg_client_loss': 2.0, 'server_quantization_mse': 0.0012}
    ]
    log_experiment_results(logger, example_results, quantization_enabled=True)

if __name__ == "__main__":
    # Test the logging system
    print("Testing quantized logging system...")
    
    # Test basic logger
    logger = setup_logger("test", "logs/test.log")
    logger.info("Basic logger test")
    
    # Test quantized logger
    quant_logger = setup_quantized_logger("test_quant", "test_quantization")
    quant_logger.info("Quantized logger test")
    quant_logger.info("Testing quantization MSE: 0.001234")
    
    # Test context manager
    with LoggingContextManager("test_context", quantization_enabled=True) as context_logger:
        context_logger.info("Testing context manager")
        context_logger.info("This is a quantization-related message with scale=0.5")
    
    print("Logging system test completed!")
    
    # Run example
    # example_quantized_logging()
