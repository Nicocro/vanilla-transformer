from typing import Dict, Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F

### Model Blocks ###

class FeedForward(nn.Module):
  def __init__(self, d_model: int, ff_hidden_ratio: int=4):
    super().__init__()
    self.hidden_dim = d_model * ff_hidden_ratio
    self.embed_dim = d_model

    self.net = nn.Sequential(
        nn.Linear(d_model, self.hidden_dim),
        nn.GELU(),
        nn.Linear(self.hidden_dim, d_model)
    )

  def forward(self, x):
    return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int): 
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding
    

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int=1, mask: bool=False):
        super().__init__()
        assert d_model % n_heads == 0 # d_model must be divisible by n_heads

        self.n_heads = n_heads
        self.mask = mask
        self.d_model = d_model
        self.d_k = d_model // n_heads

        # create multi heads attention projections
        self.Wq = nn.Linear(d_model, self.d_k * n_heads)
        self.Wk = nn.Linear(d_model, self.d_k * n_heads)
        self.Wv = nn.Linear(d_model, self.d_k * n_heads)

        # Final projection back to d_model
        self.Wo = nn.Linear(self.d_k * n_heads, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch_size, seq_len, _ = x.size()

        Q = self.Wq(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, device=x.device))

        # add masking here for all the heads
        if self.mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        O = self.Wo(attn_output)

        return O, {'attn_weights' : attn_weights} 
    

class CrossAttention(nn.Module):
    def __init__(self, encoder_d_model: int, decoder_d_model:int, n_heads: int=1):
        super().__init__()
        assert decoder_d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.decoder_d_model = decoder_d_model
        self.encoder_d_model = encoder_d_model
        self.d_k = decoder_d_model // n_heads

        # Create multi-head projections
        self.Wq = nn.Linear(decoder_d_model, self.d_k * n_heads)
        self.Wk = nn.Linear(encoder_d_model, self.d_k * n_heads)
        self.Wv = nn.Linear(encoder_d_model, self.d_k * n_heads)

        # Final projection back to d_model
        self.Wo = nn.Linear(self.d_k * n_heads, decoder_d_model)
    
    def forward(self, enc_out: torch.Tensor, dec_in: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch_size, enc_len, _ = enc_out.size()
        _, dec_len, _ = dec_in.size()

        # Project and reshape to [batch_size, seq_len, n_heads, d_k]
        Q = self.Wq(dec_in).view(batch_size, dec_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(enc_out).view(batch_size, enc_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(enc_out).view(batch_size, enc_len, self.n_heads, self.d_k).transpose(1, 2)

        # Calculate attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, device=dec_in.device))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = (attn_weights @ V).transpose(1, 2).contiguous().view(batch_size, dec_len, self.n_heads * self.d_k)
        
        # Final projection
        O = self.Wo(attn_output)

        return O, {'attn_weights' : attn_weights} 

        