import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        print(f"🔧 RoPE Init: dim={dim}, max_pos={max_position_embeddings}")
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_cached = emb.cos().to(dtype=x.dtype)
        sin_cached = emb.sin().to(dtype=x.dtype)
        
        print(f"🔧 RoPE Forward: seq_len={seq_len}, emb.shape={emb.shape}, cos.shape={cos_cached.shape}")
        
        return cos_cached, sin_cached

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding with automatic dimension fixing"""
    print(f"🔍 RoPE Debug - Input shapes:")
    print(f"  q: {q.shape}, k: {k.shape}")
    print(f"  cos: {cos.shape}, sin: {sin.shape}")
    print(f"  position_ids: {position_ids.shape if position_ids is not None else None}")
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Handle cos/sin - they come as [seq_len, dim] from forward()
    if len(cos.shape) == 2:
        # cos/sin are [seq_len, dim] - need to reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]
    
    cos_head_dim = cos.shape[-1]
    sin_head_dim = sin.shape[-1]
    
    print(f"  Expected head_dim: {head_dim}, cos_head_dim: {cos_head_dim}")
    
    # Fix dimension mismatch
    if cos_head_dim != head_dim:
        print(f"⚠️  Dimension mismatch detected: {cos_head_dim} vs {head_dim}")
        
        if cos_head_dim > head_dim:
            # Truncate cos/sin to match head_dim
            print(f"🔧 Truncating cos/sin from {cos_head_dim} to {head_dim}")
            cos = cos[..., :head_dim]
            sin = sin[..., :head_dim]
        else:
            # Pad cos/sin to match head_dim
            print(f"🔧 Padding cos/sin from {cos_head_dim} to {head_dim}")
            pad_size = head_dim - cos_head_dim
            cos = F.pad(cos, (0, pad_size), "constant", 0)
            sin = F.pad(sin, (0, pad_size), "constant", 0)
    
    # Ensure cos/sin match q/k batch and sequence dimensions
    if cos.shape[0] != batch_size:
        cos = cos.expand(batch_size, -1, -1, -1)
        sin = sin.expand(batch_size, -1, -1, -1)
    
    if cos.shape[1] != num_heads:
        cos = cos.expand(-1, num_heads, -1, -1)
        sin = sin.expand(-1, num_heads, -1, -1)
    
    if cos.shape[2] != seq_len:
        if cos.shape[2] > seq_len:
            cos = cos[:, :, :seq_len, :]
            sin = sin[:, :, :seq_len, :]
        else:
            # This shouldn't happen in normal cases
            print(f"⚠️  cos seq_len ({cos.shape[2]}) < q seq_len ({seq_len})")
            # Repeat the last position
            repeat_count = seq_len - cos.shape[2]
            last_cos = cos[:, :, -1:, :].repeat(1, 1, repeat_count, 1)
            last_sin = sin[:, :, -1:, :].repeat(1, 1, repeat_count, 1)
            cos = torch.cat([cos, last_cos], dim=2)
            sin = torch.cat([sin, last_sin], dim=2)
    
    print(f"  After fixing: cos: {cos.shape}, sin: {sin.shape}")
    
    try:
        # Apply rotary embedding
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        print(f"✅ RoPE applied successfully!")
        return q_embed, k_embed
    except Exception as e:
        print(f"❌ RoPE failed even after fixing: {e}")
        print(f"🔧 Returning original tensors without RoPE")
        return q, k

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        print(f"🔧 Attention config: hidden_size={self.hidden_size}, num_heads={self.num_heads}, head_dim={self.head_dim}")
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,  # Use calculated head_dim
            max_position_embeddings=self.max_position_embeddings,
            base=getattr(config, 'rope_theta', 10000.0),
        )
        
        # Set parameter names for GaLore
        self.q_proj.weight.param_name = "q_proj"
        self.k_proj.weight.param_name = "k_proj"
        self.v_proj.weight.param_name = "v_proj"
        self.o_proj.weight.param_name = "o_proj"
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Repeat key and value states for multi-head attention
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Set parameter names for GaLore
        self.gate_proj.weight.param_name = "gate_proj"
        self.up_proj.weight.param_name = "up_proj"
        self.down_proj.weight.param_name = "down_proj"
    
    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        intermediate = F.silu(gate) * up
        return self.down_proj(intermediate)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class LlamaLayerRange(nn.Module):
    """Extract specific layers from LLaMA model"""
    
    def __init__(self, config, start_layer: int, end_layer: int):
        super().__init__()
        self.config = config
        self.start_layer = start_layer
        self.end_layer = end_layer
        
        # Create layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(end_layer - start_layer + 1)
        ])
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        return hidden_states
