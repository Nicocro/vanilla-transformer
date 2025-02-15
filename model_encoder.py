import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple
from model_blocks import FeedForward, SelfAttention, PositionalEncoding
    

class EncoderBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 ff_hidden_ratio: int=4,
                 n_heads: int=1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = SelfAttention(d_model=d_model, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model=d_model, ff_hidden_ratio=ff_hidden_ratio)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
      # firs part with residual connectino
      x_norm = self.norm1(x)
      x_att, self_attn_dict = self.self_attention(x_norm)
      x = x + x_att # residual connection

      # feed forward block with resituals
      x_norm = self.norm2(x)
      x_ff = self.ff(x_norm)
      x_out = x + x_ff # residual connection

      return x_out, self_attn_dict
    

class Encoder(nn.Module):
  def __init__(self,
               d_model: int,
               seq_len: int,
               n_heads: int=1,
               ff_hidden_ratio: int=4,
               num_enc_blocks: int=1,  
               ):
      super().__init__()
      
      self.pos_encoding = PositionalEncoding(d_model, seq_len)
      
      self.blocks = nn.ModuleList([
          EncoderBlock(d_model=d_model,
                       n_heads=n_heads,
                       ff_hidden_ratio=ff_hidden_ratio)
          for _ in range(num_enc_blocks)
      ])

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
      x = self.pos_encoding(x)
      attention = {}

      for block in self.blocks:
          x, attention = block(x)

      return x, {'self_attention': attention} # returns the last attention dict