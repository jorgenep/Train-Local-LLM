"""
Llama-style Decoder-Only Transformer Architecture
Built with:
- RoPE (Rotary Position Embeddings)
- RMSNorm (Pre-normalization)
- SwiGLU Feed-Forward Network
- Grouped-Query Attention (GQA)
- Tied Embeddings (Weight sharing between input embedding & output head)
- Configurable context length (default 2048 tokens)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm calculation: x * rsqrt(mean(x^2) + eps) * weight
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    # xq shape: (batch, seq, n_q_heads, head_dim)
    # xk shape: (batch, seq, n_kv_heads, head_dim)
    # freqs_cos/sin shape: (seq, head_dim // 2)
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
    
    # Broadcast across batch and head dimensions: (1, seq, 1, head_dim // 2)
    cos = freqs_cos.unsqueeze(0).unsqueeze(2).to(xq.device)
    sin = freqs_sin.unsqueeze(0).unsqueeze(2).to(xq.device)
    
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos
    
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Standard Llama SwiGLU uses gate (w1), up (w3), and down (w2) linear projections without bias
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class GroupedQueryAttention(nn.Module):
    def __init__(self, emb_dim: int = 1024, n_heads: int = 16, n_kv_heads: int = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = emb_dim // n_heads
        self.num_rep = n_heads // n_kv_heads
        
        self.q_proj = nn.Linear(emb_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, emb_dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape
        
        xq = self.q_proj(x).view(b, seq, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(b, seq, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(b, seq, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        
        # Expand KV heads for Grouped-Query Attention if n_kv_heads < n_heads
        if self.num_rep > 1:
            xk = xk.repeat_interleave(self.num_rep, dim=2)
            xv = xv.repeat_interleave(self.num_rep, dim=2)
            
        # Reshape for PyTorch scaled dot-product attention: (b, n_heads, seq, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Flash / Efficient Causal Attention
        output = F.scaled_dot_product_attention(
            xq, xk, xv, is_causal=True
        )
        
        # Transpose back: (b, seq, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(b, seq, -1)
        return self.o_proj(output)

class LlamaBlock(nn.Module):
    def __init__(self, emb_dim: int = 1024, n_heads: int = 16, n_kv_heads: int = 4):
        super().__init__()
        self.attn_norm = RMSNorm(emb_dim)
        self.attn = GroupedQueryAttention(emb_dim=emb_dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
        
        self.ffn_norm = RMSNorm(emb_dim)
        # Standard Llama SwiGLU dimension calculation (2/3 * 4 * emb_dim, rounded to multiple of 256)
        hidden_dim = int(2 * (4 * emb_dim) / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.ffn = SwiGLUFFN(emb_dim, hidden_dim)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        # Pre-normalization residual blocks
        x = x + self.attn(self.attn_norm(x), freqs_cos, freqs_sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class LlamaCodeLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16384,
        emb_dim: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        context_length: int = 2048
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.head_dim = emb_dim // n_heads
        
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.layers = nn.ModuleList([
            LlamaBlock(emb_dim=emb_dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        
        # TIED EMBEDDINGS: Share weights between token embedding & output classification head
        self.lm_head.weight = self.tok_emb.weight
        
        # Precompute RoPE frequencies for up to max context_length
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, context_length)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, seq = idx.shape
        assert seq <= self.context_length, f"Sequence length {seq} exceeds max context length {self.context_length}"
        
        x = self.tok_emb(idx)
        freqs_cos = self.freqs_cos[:seq]
        freqs_sin = self.freqs_sin[:seq]
        
        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        # Since lm_head is tied with tok_emb, PyTorch count already handles shared memory pointers correctly
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

if __name__ == "__main__":
    # Quick instantiation test
    model = LlamaCodeLM(vocab_size=16384, emb_dim=1024, n_layers=24, context_length=2048)
    params = model.count_parameters()
    print(f"Model initialized! Total Parameters: {params['total_parameters']:,}")
