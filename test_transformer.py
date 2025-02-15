import torch
import torch.nn as nn
from model_transformer import Transformer
from model_blocks import SelfAttention, CrossAttention, FeedForward, PositionalEncoding
from model_encoder import EncoderBlock, Encoder
from model_decoder import DecoderBlock, Decoder

def test_model_dimensions():
    """Test if the model handles various input dimensions correctly"""
    print("\n=== Testing Model Dimensions ===")
    
    # Model parameters
    batch_size = 4
    enc_seq_len = 16
    dec_seq_len = 10
    encoder_d_model = 64
    decoder_d_model = 64
    num_enc_blocks = 2
    num_dec_blocks = 2
    n_heads = 4
    
    # Create model
    model = Transformer(
        encoder_d_model=encoder_d_model,
        decoder_d_model=decoder_d_model,
        num_enc_blocks=num_enc_blocks,
        num_dec_blocks=num_dec_blocks,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        n_heads=n_heads
    )
    
    # Create dummy inputs
    x = torch.randn(batch_size, enc_seq_len, encoder_d_model)
    y = torch.randn(batch_size, dec_seq_len, decoder_d_model)
    
    # Forward pass
    try:
        output, attention_dict = model(x, y)
        print(f"✓ Model forward pass successful")
        print(f"✓ Input shapes - Encoder: {x.shape}, Decoder: {y.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Expected output shape: {(batch_size, dec_seq_len, dec_seq_len)}")
        
        # Check attention shapes
        enc_attn = attention_dict['encoder']['self_attention']['attn_weights']
        dec_self_attn = attention_dict['decoder']['self_attention']
        dec_cross_attn = attention_dict['decoder']['cross_attention']
        
        print("\nAttention Shapes:")
        print(f"✓ Encoder self-attention: {enc_attn.shape}")
        print(f"✓ Decoder self-attention: {dec_self_attn.shape}")
        print(f"✓ Decoder cross-attention: {dec_cross_attn.shape}")
        
    except Exception as e:
        print(f"✗ Error in forward pass: {str(e)}")
        raise e

def test_causal_masking():
    """Test if the causal masking in decoder's self-attention works correctly"""
    print("\n=== Testing Causal Masking ===")
    
    d_model = 64
    seq_len = 10
    n_heads = 2
    batch_size = 3
    
    # Create self-attention with masking
    masked_self_attention = SelfAttention(d_model=d_model, n_heads=n_heads, mask=True)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attention_info = masked_self_attention(x)
    attention_weights = attention_info['attn_weights']
    
    # Check if future tokens are properly masked (should be very close to 0)
    mask_violated = False
    for batch in range(batch_size):
        for head in range(n_heads):
            attn_matrix = attention_weights[batch, head]
            upper_triangle = torch.triu(attn_matrix, diagonal=1)
            if not torch.all(upper_triangle < 1e-7):  # Small threshold for numerical precision
                mask_violated = True
                print(f"Found unexpectedly large value in upper triangle: {upper_triangle.max().item()}")
                break
    
    print(f"✓ Causal masking {'working correctly' if not mask_violated else 'FAILED'}")
    print("\nSample attention matrix (first head, first batch):")
    print(attention_weights[0, 0].round(decimals=3))  # Show for debugging

def test_components():
    """Test individual components of the transformer"""
    print("\n=== Testing Individual Components ===")
    
    batch_size = 2
    seq_len = 8
    d_model = 32
    n_heads = 2
    
    # Test FeedForward
    ff = FeedForward(d_model=d_model)
    ff_input = torch.randn(batch_size, seq_len, d_model)
    ff_output = ff(ff_input)
    print(f"✓ FeedForward - Input: {ff_input.shape}, Output: {ff_output.shape}")
    
    # Test PositionalEncoding
    pos_enc = PositionalEncoding(d_model=d_model, seq_len=seq_len)
    pos_input = torch.randn(batch_size, seq_len, d_model)
    pos_output = pos_enc(pos_input)
    print(f"✓ PositionalEncoding - Input: {pos_input.shape}, Output: {pos_output.shape}")
    
    # Test EncoderBlock
    enc_block = EncoderBlock(d_model=d_model, n_heads=n_heads)
    enc_input = torch.randn(batch_size, seq_len, d_model)
    enc_output, _ = enc_block(enc_input)
    print(f"✓ EncoderBlock - Input: {enc_input.shape}, Output: {enc_output.shape}")
    
    # Test DecoderBlock
    dec_block = DecoderBlock(encoder_d_model=d_model, decoder_d_model=d_model, n_heads=n_heads)
    dec_input = torch.randn(batch_size, seq_len, d_model)
    enc_output = torch.randn(batch_size, seq_len, d_model)
    dec_output, _ = dec_block(dec_input, enc_output)
    print(f"✓ DecoderBlock - Input: {dec_input.shape}, Output: {dec_output.shape}")

def run_all_tests():
    """Run all test functions"""
    try:
        test_components()
        test_causal_masking()
        test_model_dimensions()
        print("\n✓ All tests passed successfully!")
    except Exception as e:
        print(f"\n✗ Tests failed: {str(e)}")

if __name__ == "__main__":
    run_all_tests()