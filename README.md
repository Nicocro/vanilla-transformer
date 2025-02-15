# vanilla-transformer

A clean PyTorch implementation of the Transformer architecture from "Attention Is All You Need", designed for learning and experimentation.
The code prioritizes readability and understanding over optimization. Each component is self-contained with relevant comments explaining the key concepts.

## Architecture
The implementation starts assuming embedded sequences (vectors) as inputs. The first implemented encoder/decoder layers are learnable positional encodings that are added to the input embeddings. From there:

Encoder: Self-attention followed by feed-forward networks
Decoder: Masked self-attention and cross-attention with feed-forward networks
Both include standard layer normalization and residual connections