import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class DifferentiableINT4Quantizer(nn.Module):
    """
    Differentiable INT4 quantization using sigmoid-based soft quantization.
    Formula: sum(n=0 to 15) n * (sigmoid(k*(s-n)) - sigmoid(k*(s-n-1)))
    where s is the scaled input and k is the sharpness parameter.
    """
    
    def __init__(self, k: float = 10.0, learnable_scale: bool = True):
        super().__init__()
        self.k = k  # Sharpness parameter (sufficiently large)
        self.num_levels = 16  # INT4 has 16 levels (0-15)
        
        # Learnable scale and zero-point parameters
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.zero_point = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('scale', torch.tensor(1.0))
            self.register_buffer('zero_point', torch.tensor(0.0))
    
    def calibrate_scale_zero_point(self, x: torch.Tensor) -> None:
        """
        Calibrate scale and zero_point based on input statistics
        """
        with torch.no_grad():
            x_min = x.min()
            x_max = x.max()
            
            # Calculate scale to map [x_min, x_max] to [0, 15]
            scale = (x_max - x_min) / (self.num_levels - 1)
            zero_point = -x_min / scale
            
            # Clamp zero_point to valid range
            zero_point = torch.clamp(zero_point, 0, self.num_levels - 1)
            
            self.scale.data = scale
            self.zero_point.data = zero_point
            
            logger.debug(f"Calibrated quantizer: scale={scale:.6f}, zero_point={zero_point:.6f}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply differentiable INT4 quantization
        
        Args:
            x: Input tensor in FP32
            
        Returns:
            Quantized tensor (still in FP32 but quantized values)
        """
        # Scale input to [0, 15] range approximately
        s = (x - self.zero_point * self.scale) / self.scale + self.zero_point
        
        # Apply the differentiable quantization formula
        quantized = self._sigmoid_quantization(s)
        
        # Scale back to original range
        quantized_scaled = (quantized - self.zero_point) * self.scale + self.zero_point * self.scale
        
        return quantized_scaled
    
    def _sigmoid_quantization(self, s: torch.Tensor) -> torch.Tensor:
        """
        Apply the sigmoid-based quantization formula:
        sum(n=0 to 15) n * (sigmoid(k*(s-n)) - sigmoid(k*(s-n-1)))
        """
        result = torch.zeros_like(s)
        
        for n in range(self.num_levels):
            if n == 0:
                # For n=0: sigmoid(k*(s-0)) - sigmoid(k*(s-(-1))) = sigmoid(k*s) - sigmoid(k*(s+1))
                term = torch.sigmoid(self.k * s) - torch.sigmoid(self.k * (s + 1))
            else:
                # For n>0: sigmoid(k*(s-n)) - sigmoid(k*(s-n-1))
                term = torch.sigmoid(self.k * (s - n)) - torch.sigmoid(self.k * (s - n - 1))
            
            result += n * term
        
        return result

class QuantizedCommunication(nn.Module):
    """
    Handles quantized communication between client and server
    """
    
    def __init__(self, k: float = 10.0, auto_calibrate: bool = True):
        super().__init__()
        self.client_to_server_quantizer = DifferentiableINT4Quantizer(k=k, learnable_scale=True)
        self.server_to_client_quantizer = DifferentiableINT4Quantizer(k=k, learnable_scale=True)
        self.auto_calibrate = auto_calibrate
        self.calibrated = False
    
    def quantize_client_to_server(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Quantize hidden states from client to server
        """
        if self.auto_calibrate and not self.calibrated:
            self.client_to_server_quantizer.calibrate_scale_zero_point(hidden_states)
        
        quantized = self.client_to_server_quantizer(hidden_states)
        
        # Add some noise for privacy (optional)
        noise_std = 0.01
        noise = torch.randn_like(quantized) * noise_std
        quantized_with_noise = quantized + noise
        
        logger.debug(f"Client->Server quantization: {hidden_states.shape}, "
                    f"range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}] -> "
                    f"[{quantized.min():.4f}, {quantized.max():.4f}]")
        
        return quantized_with_noise
    
    def quantize_server_to_client(self, processed_states: torch.Tensor) -> torch.Tensor:
        """
        Quantize processed states from server to client
        """
        if self.auto_calibrate and not self.calibrated:
            self.server_to_client_quantizer.calibrate_scale_zero_point(processed_states)
            self.calibrated = True
        
        quantized = self.server_to_client_quantizer(processed_states)
        
        logger.debug(f"Server->Client quantization: {processed_states.shape}, "
                    f"range: [{processed_states.min():.4f}, {processed_states.max():.4f}] -> "
                    f"[{quantized.min():.4f}, {quantized.max():.4f}]")
        
        return quantized


# Test function to verify quantization
def test_quantization():
    """Test the quantization implementation"""
    print("Testing INT4 Quantization...")
    
    # Create test data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_tensor = torch.randn(2, 10, 128, device=device)  # Batch, seq, hidden
    
    # Initialize quantizer
    quantizer = DifferentiableINT4Quantizer(k=10.0).to(device)
    quantizer.calibrate_scale_zero_point(test_tensor)
    
    # Test forward pass
    quantized = quantizer(test_tensor)
    
    print(f"Original range: [{test_tensor.min():.4f}, {test_tensor.max():.4f}]")
    print(f"Quantized range: [{quantized.min():.4f}, {quantized.max():.4f}]")
    print(f"MSE loss: {F.mse_loss(test_tensor, quantized):.6f}")
    
    # Test gradient flow
    test_tensor.requires_grad_(True)
    quantized = quantizer(test_tensor)
    loss = quantized.sum()
    loss.backward()
    
    print(f"Gradient flow check: {test_tensor.grad is not None}")
    print(f"Gradient norm: {test_tensor.grad.norm():.6f}")
    
    print("✅ Quantization test completed successfully!")

if __name__ == "__main__":
    test_quantization()
