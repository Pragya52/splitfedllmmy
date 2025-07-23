from dataclasses import dataclass, asdict
from typing import List
import yaml

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "llama-7b"
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 3
    max_seq_length: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0

@dataclass
class GaLoreConfig:
    """GaLore optimization configuration"""
    rank: int = 1024
    update_proj_gap: int = 500
    scale: float = 0.25
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    num_clients: int = 3
    num_rounds: int = 10
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
    """Main configuration class"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    galore: GaLoreConfig = GaLoreConfig()
    federated: FederatedConfig = FederatedConfig()
    kd: KnowledgeDistillationConfig = KnowledgeDistillationConfig()
    
    @classmethod
    @classmethod
    def from_yaml(cls, path: str):
      with open(path, 'r') as f:
         data = yaml.safe_load(f)
    
      # Ensure numeric types are correct
      model_data = data.get('model', {})
      training_data = data.get('training', {})
    
      # Convert string numbers to proper types
      if 'learning_rate' in training_data:
        training_data['learning_rate'] = float(training_data['learning_rate'])
      if 'vocab_size' in model_data:
        model_data['vocab_size'] = int(model_data['vocab_size'])
      if 'hidden_size' in model_data:
        model_data['hidden_size'] = int(model_data['hidden_size'])
    
      return cls(
        model=ModelConfig(**model_data),
        training=TrainingConfig(**training_data),
        galore=GaLoreConfig(**data.get('galore', {})),
        federated=FederatedConfig(**data.get('federated', {})),
        kd=KnowledgeDistillationConfig(**data.get('kd', {}))
      )
