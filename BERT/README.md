# BERT Implementation

This folder contains a PyTorch implementation of BERT (Bidirectional Encoder Representations from Transformers) based on the paper ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805).

## Implementation Details

The implementation focuses on a simplified version of BERT with both pre-training objectives:

### Model Architecture
- Bidirectional Transformer encoder
- Token, positional, and segment embeddings
- Multi-head self-attention mechanism
- Feed-forward networks with GELU activation
- Layer normalization and residual connections
- MLM (Masked Language Modeling) head
- NSP (Next Sentence Prediction) head

### Data Processing
- Custom text processing from local text files
- Sentence splitting using spaCy
- Word-level tokenization and vocabulary building
- Implementation of MLM with 15% random token masking:
  - 80% replaced with [MASK]
  - 10% replaced with random words
  - 10% left unchanged
- Next sentence prediction with 50% positive/negative pairs

### Training
- Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) losses
- AdamW optimizer with learning rate scheduling
- Linear warmup followed by linear decay
- Gradient clipping for stability
- Inference demo for validating model predictions

The implementation works with a simple text file (animal_story.txt) provided in the folder for demonstration purposes.