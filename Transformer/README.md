# Transformer Implementation

This folder contains a PyTorch implementation of the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

## Implementation Details

The implementation focuses on a sequence-to-sequence neural machine translation model with the following components:

### Model Architecture
- Complete encoder-decoder Transformer architecture
- Multi-head self-attention mechanism
- Positional encoding using sine and cosine functions
- Feed-forward networks
- Residual connections and layer normalization
- Greedy decoding for inference

### Dataset Processing
- Uses the `opus_books` dataset from Hugging Face for translation tasks
- Implements tokenization using word-level tokenizers
- Creates separate tokenizers for source and target languages
- Handles padding and special tokens ([SOS], [EOS], [PAD])
- Implements causal masking for autoregressive decoding

### Training
- Configurable batch size, learning rate, and other hyperparameters
- Support for checkpointing and loading pre-trained models
- Validation metrics (Character Error Rate, Word Error Rate, BLEU)
- TensorBoard integration for monitoring training progress
- Label smoothing for regularization

The implementation is designed for machine translation between languages, with English (en) and Italian (it) as the default source and target languages.

