# Updated configuration to support quantization parameters

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class QuantizationConfig:
    """Configuration for INT4 quantization"""
    enabled: bool = True
    k: float = 10.0  # Sharpness parameter for sigmoid quantization
    learnable_scale: bool = True  # Whether quantization scales are learnable
    auto_calibrate: bool = True  # Whether to auto-calibrate scales based on data
    privacy_noise_std: float = 0.01  # Standard deviation of privacy noise
    log_quantization_stats: bool = True  # Whether to log quantization statistics

@dataclass 
class ModelConfig:
    """Model configuration with quantization support"""
    model_name: str = "llama-7b"
    vocab_size: int = 50257
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    max_position_embeddings: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_epochs: int = 1
    max_seq_length: int = 64
    warmup_steps: int = 10
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0

@dataclass
class GaLoreConfig:
    """GaLore optimizer configuration"""
    rank: int = 16
    update_proj_gap: int = 5
    scale: float = 0.25
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    num_clients: int = 3
    num_rounds: int = 5
    client_fraction: float = 1.0
    local_epochs: int = 1
    server_address: str = "localhost"
    server_port: int = 8080

@dataclass
class KnowledgeDistillationConfig:
    """Knowledge distillation configuration"""
    temperature: float = 3.0
    alpha: float = 0.5

@dataclass
class Config:
    """Main configuration class with quantization support"""
    model: ModelConfig
    training: TrainingConfig
    galore: GaLoreConfig
    federated: FederatedConfig
    kd: KnowledgeDistillationConfig
    quantization: QuantizationConfig = None  # ✅ NEW: Quantization config
    
    def __post_init__(self):
        # Initialize quantization config if not provided
        if self.quantization is None:
            self.quantization = QuantizationConfig()
    
    @classmethod
    def create_quantized_config(cls, num_clients: int = 3, quantization_k: float = 10.0):
        """Create a sample configuration optimized for quantized federated learning"""
        
        config = cls(
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
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                              "gate_proj", "up_proj", "down_proj"]
            ),
            federated=FederatedConfig(
                num_clients=num_clients,
                num_rounds=5,   # 5 rounds for testing
                client_fraction=1.0,
                local_epochs=1,
                server_address="localhost",
                server_port=8080
            ),
            kd=KnowledgeDistillationConfig(
                temperature=3.0,
                alpha=0.5
            ),
            quantization=QuantizationConfig(  # ✅ NEW: Quantization settings
                enabled=True,
                k=quantization_k,
                learnable_scale=True,
                auto_calibrate=True,
                privacy_noise_std=0.01,
                log_quantization_stats=True
            )
        )
        
        # Verify configuration compatibility
        head_dim = config.model.hidden_size // config.model.num_attention_heads
        print(f"📊 Quantized Config verification for {config.federated.num_clients} clients:")
        print(f"  hidden_size: {config.model.hidden_size}")
        print(f"  num_attention_heads: {config.model.num_attention_heads}")
        print(f"  head_dim: {head_dim}")
        print(f"  max_seq_length: {config.training.max_seq_length}")
        print(f"  vocab_size: {config.model.vocab_size}")
        print(f"  num_clients: {config.federated.num_clients}")
        print(f"  quantization_enabled: {config.quantization.enabled}")
        print(f"  quantization_k: {config.quantization.k}")
        
        # Ensure clean division
        assert config.model.hidden_size % config.model.num_attention_heads == 0, \
            f"hidden_size ({config.model.hidden_size}) must be divisible by num_attention_heads ({config.model.num_attention_heads})"
        
        # Ensure reasonable head dimension for RoPE
        assert head_dim >= 8, f"head_dim ({head_dim}) should be at least 8 for proper RoPE functionality"
        
        print(f"✅ Quantized configuration validated successfully!")
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file with quantization support"""
        import yaml
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to dataclass instances
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        galore_config = GaLoreConfig(**config_dict.get('galore', {}))
        federated_config = FederatedConfig(**config_dict.get('federated', {}))
        kd_config = KnowledgeDistillationConfig(**config_dict.get('kd', {}))
        
        # ✅ NEW: Handle quantization config
        quantization_dict = config_dict.get('quantization', {})
        quantization_config = QuantizationConfig(**quantization_dict) if quantization_dict else QuantizationConfig()
        
        return cls(
            model=model_config,
            training=training_config,
            galore=galore_config,
            federated=federated_config,
            kd=kd_config,
            quantization=quantization_config  # ✅ NEW
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file including quantization settings"""
        import yaml
        from dataclasses import asdict
        
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'galore': asdict(self.galore),
            'federated': asdict(self.federated),
            'kd': asdict(self.kd),
            'quantization': asdict(self.quantization)  # ✅ NEW
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to {yaml_path}")


# ✅ NEW: Example YAML configuration with quantization
SAMPLE_QUANTIZED_CONFIG_YAML = """
model:
  model_name: "llama-7b"
  vocab_size: 50257
  hidden_size: 128
  intermediate_size: 256
  num_hidden_layers: 2
  num_attention_heads: 4
  num_key_value_heads: 4
  max_position_embeddings: 128
  rms_norm_eps: 1.0e-6
  rope_theta: 10000.0
  attention_dropout: 0.0

training:
  batch_size: 1
  learning_rate: 1.0e-4
  num_epochs: 1
  max_seq_length: 64
  warmup_steps: 10
  weight_decay: 0.01
  gradient_clipping: 1.0

galore:
  rank: 16
  update_proj_gap: 5
  scale: 0.25
  target_modules:
    - "q_proj"
    - "k_proj" 
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

federated:
  num_clients: 3
  num_rounds: 5
  client_fraction: 1.0
  local_epochs: 1
  server_address: "localhost"
  server_port: 8080

kd:
  temperature: 3.0
  alpha: 0.5

quantization:
  enabled: true
  k: 10.0
  learnable_scale: true
  auto_calibrate: true
  privacy_noise_std: 0.01
  log_quantization_stats: true
"""

def save_sample_quantized_config(path: str = "configs/quantized_config.yaml"):
    """Save a sample quantized configuration to file"""
    import os
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        f.write(SAMPLE_QUANTIZED_CONFIG_YAML)
    
    print(f"Sample quantized configuration saved to {path}")

if __name__ == "__main__":
    # Create and test quantized configuration
    config = Config.create_quantized_config(num_clients=3, quantization_k=10.0)
    
    # Save sample config
    save_sample_quantized_config()
    
    # Test loading from YAML
    try:
        loaded_config = Config.from_yaml("configs/quantized_config.yaml")
        print("✅ Configuration loading test passed!")
        print(f"Quantization enabled: {loaded_config.quantization.enabled}")
        print(f"Quantization k: {loaded_config.quantization.k}")
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
