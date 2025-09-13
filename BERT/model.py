"""
BERT - Implementation from scratch

Paper: https://arxiv.org/pdf/1810.04805
Notes: https://github.com/hkproj/bert-from-scratch

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

import torch
import torch.nn as nn
import math


# -------------------- Embedding Layers --------------------

class Embeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int, seq_len: int, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(seq_len, d_model)
        self.segment_embed = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids):
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(input_ids.size(1), device=input_ids.device).expand_as(input_ids)

        # Get embeddings
        token_emb = self.token_embed(input_ids)
        pos_emb = self.position_embed(positions)
        seg_emb = self.segment_embed(segment_ids)

        # Combine and normalize
        x = token_emb + pos_emb + seg_emb
        x = self.norm(x)
        return self.dropout(x)


# -------------------- Transformer Encoder --------------------

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()  # BERT uses GELU activation

    def forward(self, x):
        # (batch, seq_len, d_model)
        x = self.dropout(self.gelu(self.fc1(x)))  # (batch, seq_len, d_ff)
        return self.fc2(x)  # (batch, seq_len, d_model)


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


class ResidualConnection(nn.Module):
    def __init__(self, d_model:int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))    # Post-LN

class EncoderBlock(nn.Module):
    def __init__(self, d_model:int, self_attn_block: MultiHeadAttentionBlock, feed_fwd_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attn_block = self_attn_block
        self.feed_fwd_block = feed_fwd_block
        self.residuals = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, mask):
        x = self.residuals[0] (x, lambda x: self.self_attn_block(x, x, x, mask))  # first skip-connection
        x = self.residuals[1] (x, self.feed_fwd_block)  # second skip-connection
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# -------------------- BERT Model --------------------

class BERT(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model=768, N=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # Unified embedding module
        self.embeddings = Embeddings(d_model, vocab_size, seq_len, dropout)

        # Encoder stack
        encoder_blocks = []
        for _ in range(N):
            self_attn = MultiHeadAttentionBlock(d_model, h, dropout)
            ffn = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_blocks.append(EncoderBlock(d_model, self_attn, ffn, dropout))
        self.encoder = Encoder(nn.ModuleList(encoder_blocks))

        # For MLM Head (Linear -> GELU -> LayerNorm -> Linear)
        self.mlm_dense = nn.Linear(d_model, d_model)
        self.mlm_activation = nn.GELU()
        self.mlm_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.mlm_decoder = nn.Linear(d_model, vocab_size)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

        # Tie MLM decoder weights to input token embeddings
        self.mlm_decoder.weight = self.embeddings.token_embed.weight
        self.mlm_decoder.bias = self.mlm_bias  # Also tie the bias

        # For NSP Head
        self.nsp_classifier = nn.Linear(d_model, 2)  # IsNext, NotNext

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, segment_ids):
        # Create padding mask (batch, seq_len) -> (batch, 1, 1, seq_len)
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
        mask = mask.to(input_ids.device)

        # Get embeddings
        x = self.embeddings(input_ids, segment_ids)

        # Encoder processing
        x = self.encoder(x, mask)

        # NSP: Use [CLS] token (first token's output) for classification
        nsp_logits = self.nsp_classifier(x[:, 0])

        # MLM: Process the ENTIRE sequence output through the MLM head.
        # The loss function will be responsible for selecting the masked positions.
        mlm_hidden = self.mlm_activation(self.mlm_dense(x))
        mlm_hidden = self.mlm_norm(mlm_hidden)
        mlm_logits = self.mlm_decoder(mlm_hidden)

        # mlm_logits has shape (batch_size, seq_len, vocab_size)
        return mlm_logits, nsp_logits


# -------------------- Testing BERT Model --------------------

# def test_bert():
#     vocab_size = 30522   # standard BERT vocab
#     seq_len = 16         # keep it short for test
#     batch_size = 2
#
#     model = BERT(vocab_size=vocab_size, seq_len=seq_len, d_model=128, N=2, h=4, d_ff=256)
#
#     input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))   # (batch, seq_len)
#     segment_ids = torch.randint(0, 2, (batch_size, seq_len))          # (batch, seq_len)
#
#     mlm_logits, nsp_logits = model(input_ids, segment_ids)
#
#     print("Input IDs shape:", input_ids.shape)
#     print("MLM logits shape:", mlm_logits.shape)   # (batch, seq_len, vocab_size)
#     print("NSP logits shape:", nsp_logits.shape)   # (batch, 2)
#
#     # Quick check
#     assert mlm_logits.shape == (batch_size, seq_len, vocab_size)
#     assert nsp_logits.shape == (batch_size, 2)
#
#
# # Run test
# if __name__ == "__main__":
#     test_bert()
#
