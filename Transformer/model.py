"""
Attention Is All You Need — Implementation

Paper: https://arxiv.org/abs/1706.03762
Notes: https://github.com/hkproj/transformer-from-scratch-notes/tree/main

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

import torch
import torch.nn as nn
import math


# -------------------- Embedding Layers --------------------

# Step 1: Input Embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model  # size of the embedding vector
        self.vocab_size = vocab_size  # size of the vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: (batch, seq_len)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper (pg.5)
        return self.embedding(x) * math.sqrt(self.d_model)  # (batch, seq_len, d_model)


# Step 2: Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (10000 ** (-i / dmodel)) --> inverse of an original form in paper
        # apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sin ( position * (10000 ** (-i / d_model))
        # apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos ( position * (10000 ** (-i / d_model))
        # add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer, not a parameter.
        self.register_buffer('pe', pe)  # Registered pe, not a parameter

    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


# -------------------- Transformer Encoder --------------------

# Step 3: Layer Normalization
class LayerNormalization(nn.Module):
    # Applies layer normalization over the last dimension of the input tensor.
    def __init__(self, eps:float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Learnable parameters alpha and bias
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative (scale)
        self.bias = nn.Parameter(torch.zeros(1)) # Additive (shift)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1) and keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias  # scale and shift


# Step 4: Feed Forward Network - paper eq. 2
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch, seq_len, d_model)
        x = self.dropout(self.relu(self.fc1(x)))  # (batch, seq_len, d_ff)
        return self.fc2(x)  # (batch, seq_len, d_model)


# Step 5: Multi-head attention
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model  # size of the embedding vector
        self.h = h  # number of heads
        # imp: d_model must be divisible by h
        assert d_model % h ==0, "d_model is not divisible by h"

        self.d_k = d_model // h  # dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_0 = nn.Linear(d_model, d_model)  #Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    # paper eq.1
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]  # last dim of q, k, and v

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # @ means matrix multiplication
        if mask is not None:
            # write a very low value like '-1e9' indicating -inf to the positions where mask ==0
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim = -1)  # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_score = dropout(attention_score)
        # (batch, h, seq_len, seq_len) --> # (batch, h, seq_len, d_k)
        return (attention_score @ value), attention_score  # output, attention score for visualization after softmax

    def forward(self, q, k, v, mask):
        # (batch, seq_len, d_model --> (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # transpose swap
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply the last matrix Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_0(x)


# Step 6: Residual Block
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sub_layer):
        # sub_layer is a function/layer to apply to normalized x
        return x + self.dropout(sub_layer(self.norm(x)))


# Step 7: One Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, self_attn_block: MultiHeadAttentionBlock, feed_fwd_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attn_block = self_attn_block
        self.feed_fwd_block = feed_fwd_block
        self.residuals = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residuals[0] (x, lambda x: self.self_attn_block(x, x, x, src_mask))  # first skip-connection
        # and first inp from x. other x is coming from attn block. sub_layer is defined using lambda fn.
        x = self.residuals[1] (x, self.feed_fwd_block)  # second skip-connection - combine output from the previous
        # one and add it to the ffb
        return x

# Step 8: Define Encoder Blocks -  complete Encoder (stack of layers)
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# -------------------- Transformer Decoder --------------------

# Step 9: One Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, self_attn_block: MultiHeadAttentionBlock, cross_attn_block: MultiHeadAttentionBlock,
                 feed_fwd_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attn_block = self_attn_block
        self.cross_attn_block = cross_attn_block
        self.feed_fwd_block = feed_fwd_block
        self.residuals = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )
    def forward(self, x, encoder_output, src_mask, trg_mask):  # src_mask for source lang and trg_mask for target lang
        x = self.residuals[0](x, lambda x: self.self_attn_block(x, x, x, trg_mask))  # res_block 1. src mask
        x = self.residuals[1](x, lambda x: self.cross_attn_block(x, encoder_output, encoder_output, src_mask))  # res_block 2.
        # here the query is coming from the decoder, but key and value are coming from the decoder. use encoder mask.
        x = self.residuals[2](x, self.feed_fwd_block)  # res_block 3
        return x

# Step 10: Define Decoder Blocks - Stacks multiple decoder blocks
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)

# Step 11: Projection layer - final layer above the decoder
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        x = torch.log_softmax(self.proj(x), dim = -1)
        return x


# -------------------- Complete Transformer --------------------

# Step 12: Transformer Blocks
class Transformer(nn.Module):
    # The full Transformer model: encoder, decoder, embeddings, positional encodings, and projection layer.
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, trg_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, trg_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trg, trg_mask):
        # (batch, seq_len, d_model)
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

# Step 13: Final Transformer
# This function builds the full transformer model by stacking encoder and decoder blocks, and connecting all components.
def build_transformer(src_vocab_size: int, trg_vocab_size: int, src_seq_len: int, trg_seq_len: int, d_model: int=512,
                      N: int=6, h:int = 8, dropout: float=0.1, d_ff: int = 2048) -> Transformer:

    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    trg_embed = InputEmbeddings(d_model, trg_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attn_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_fwd_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attn_block, feed_fwd_block, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attn_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attn_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_fwd_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attn_block, decoder_cross_attn_block, feed_fwd_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, trg_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


# -------------------- Testing --------------------

# def test_transformer_implementation():
#
#     # configuration
#     src_vocab_size = 100
#     trg_vocab_size = 120
#     src_seq_len = 10
#     trg_seq_len = 12
#     d_model = 512
#
#     # Build transformer
#     transformer = build_transformer(
#         src_vocab_size=src_vocab_size,
#         trg_vocab_size=trg_vocab_size,
#         src_seq_len=src_seq_len,
#         trg_seq_len=trg_seq_len,
#         d_model=d_model,
#         N=2,
#         h=8,
#         dropout=0.0,
#         d_ff=1024
#     )
#
#     # Create source and target batches (batch_size = 2)
#     batch_size = 2
#     src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))   # (batch, src_seq_len)
#     trg = torch.randint(0, trg_vocab_size, (batch_size, trg_seq_len))   # (batch, trg_seq_len)
#
#     # masks (all ones, no padding for simplicity)
#     src_mask = torch.ones((batch_size, 1, 1, src_seq_len))  # (batch, 1, 1, src_seq_len)
#     trg_mask = torch.ones((batch_size, 1, trg_seq_len, trg_seq_len))  # (batch, 1, trg_seq_len, trg_seq_len)
#
#     # Forward pass
#     enc_out = transformer.encode(src, src_mask)  # (batch, src_seq_len, d_model)
#     dec_out = transformer.decode(enc_out, src_mask, trg, trg_mask)  # (batch, trg_seq_len, d_model)
#     out = transformer.project(dec_out)  # (batch, trg_seq_len, trg_vocab_size)
#
#     # Check output shapes
#     assert enc_out.shape == (batch_size, src_seq_len, d_model), f"Encoder output shape mismatch: {enc_out.shape}"
#     assert dec_out.shape == (batch_size, trg_seq_len, d_model), f"Decoder output shape mismatch: {dec_out.shape}"
#     assert out.shape == (batch_size, trg_seq_len, trg_vocab_size), f"Projection output shape mismatch: {out.shape}"
#     print("Transformer passes shape tests.")
#     print(f"Encoder output shape: {enc_out.shape}")
#     print(f"Decoder output shape: {dec_out.shape}")
#     print(f"Final output shape:   {out.shape}")
#
#
# if __name__ == "__main__":
#     test_transformer_implementation()























