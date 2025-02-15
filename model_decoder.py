import torch
import torch.nn as nn
import torch.nn.functional as F
from model_blocks import FeedForward, CrossAttention, SelfAttention, PositionalEncoding
from typing import Tuple

class DecoderBlock(nn.Module):
    def __init__(self, encoder_d_model: int, decoder_d_model, n_heads:int=1, ff_hidden_ratio: int=4):
        super().__init__()
        self.encoder_d_model = encoder_d_model
        self.decoder_d_model = decoder_d_model
        self.n_heads = n_heads

        self.attn1 = SelfAttention(d_model=decoder_d_model, n_heads=n_heads, mask=True)
        self.norm1 = nn.LayerNorm(decoder_d_model)
        self.cross_attn1 = CrossAttention(encoder_d_model=self.decoder_d_model, decoder_d_model=self.decoder_d_model, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(decoder_d_model)
        self.ff = FeedForward(decoder_d_model, ff_hidden_ratio)
        self.norm3 = nn.LayerNorm(decoder_d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x_att1, self_attn_dict = self.attn1(x)
        x_norm1 = self.norm1(x + x_att1)
        x_cross_att, cross_attn_dict = self.cross_attn1(enc_out=encoder_out, dec_in=x_norm1)
        x_norm2 = self.norm2(x_norm1 + x_cross_att)
        x_ff = self.ff(x_norm2)
        x_out = self.norm3(x_norm2 + x_ff)

        attention_info = {
            "self_attention": self_attn_dict["attn_weights"],
            "cross_attention": cross_attn_dict["attn_weights"]
        }

        return x_out, attention_info
    

class Decoder(nn.Module):
    def __init__(self, encoder_d_model:int, decoder_d_model: int, num_dec_blocks: int, seq_len: int, n_heads: int=1, ff_hidden_ratio: int=4):
        super().__init__()

        self.pos_encoding = PositionalEncoding(d_model=decoder_d_model, seq_len=seq_len)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                         encoder_d_model=encoder_d_model,
                         decoder_d_model=decoder_d_model,
                         n_heads=n_heads,
                         ff_hidden_ratio=ff_hidden_ratio)
            for _ in range(num_dec_blocks)
        ])
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x = self.pos_encoding(x)
        attention = {}
        
        for block in self.blocks:
            x, attention = block(x, encoder_out)

        return x, attention 