import torch
import torch.nn as nn
import torch.nn.functional as F
from model_encoder import Encoder
from model_decoder import Decoder
from typing import List


class Transformer(nn.Module):
    def __init__(
                 self, 
                 encoder_d_model: int, 
                 decoder_d_model: int,
                 num_enc_blocks: int, 
                 num_dec_blocks: int,
                 enc_seq_len: int,
                 dec_seq_len: int, 
                 n_heads: int=1,
                 ff_hidden_ratio: int=4,
                 ):
        super().__init__()

        self.encoder_d_model = encoder_d_model
        self.decoder_d_model = decoder_d_model
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len

        self.encoder = Encoder(
                               d_model=decoder_d_model,
                               ff_hidden_ratio=ff_hidden_ratio,
                               seq_len=enc_seq_len,
                               num_enc_blocks=num_enc_blocks
                               )

        self.decoder = Decoder(
                               encoder_d_model=encoder_d_model,
                               decoder_d_model=decoder_d_model,
                               num_dec_blocks=num_dec_blocks,
                               seq_len=dec_seq_len,
                               ff_hidden_ratio=ff_hidden_ratio,
                               n_heads=n_heads)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        encoder_out, encoder_attention = self.encoder(x)
        output, decoder_attention = self.decoder(y, encoder_out)
        
        return output, {
            'encoder': encoder_attention,
            'decoder': decoder_attention
        }   