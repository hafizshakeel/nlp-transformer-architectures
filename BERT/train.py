import torch
import torch.nn as nn
from model import BERT
from data_preprocess import get_data_and_vocab
from config import get_config
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -------------------- CONFIG and DATA --------------------
config = get_config()
make_batch, word2id, id2word, vocab_size = get_data_and_vocab(config['data_path'])

# Update config with dynamic vocab size
config['vocab_size'] = vocab_size

# -------------------- MODEL and OPTIMIZER --------------------
model = BERT(
    vocab_size=config['vocab_size'],
    seq_len=config['max_len'],
    d_model=config['hidden_dim'],
    N=config['n_layers'],
    h=config['n_heads'],
    d_ff=config['ff_dim'],
    dropout=config['dropout']
)

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token index for MLM loss
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

# Learning rate scheduler with linear warmup
total_steps = config['epochs'] * 100  # Approximate total steps
warmup_steps = int(total_steps * 0.1)  # 10% warmup


def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))


scheduler = LambdaLR(optimizer, lr_lambda)

# -------------------- TRAINING LOOP --------------------
print("Starting training...")
for epoch in range(config['epochs']):
    model.train()

    # Get batch
    input_ids, segment_ids, masked_tokens, masked_pos, is_next = make_batch(
        batch_size=config['batch_size'],
        max_len=config['max_len'],
        max_mask=config['max_mask']
    )

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    mlm_logits, nsp_logits = model(input_ids, segment_ids)

    # ---------------- MLM Loss ----------------
    # Gather predictions only for masked positions
    batch_size, seq_len, vocab_size = mlm_logits.shape
    max_mask = masked_pos.shape[1]

    # Create a mask for valid masked positions (non-zero)
    valid_mask = (masked_pos != 0)

    # Gather the logits for the masked positions
    masked_logits = mlm_logits.gather(
        1,
        masked_pos.unsqueeze(2).expand(-1, -1, vocab_size)
    )

    # Calculate loss only for valid masked positions
    loss_lm = criterion(
        masked_logits[valid_mask].view(-1, vocab_size),
        masked_tokens[valid_mask].view(-1)
    )

    # ---------------- NSP Loss ----------------
    loss_nsp = criterion(nsp_logits, is_next.long())

    # ---------------- Total Loss ----------------
    loss = loss_lm + loss_nsp

    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d} | Loss: {loss.item():.6f} '
              f'(MLM: {loss_lm.item():.6f}, NSP: {loss_nsp.item():.6f})')

    # Backpropagation
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

# -------------------- INFERENCE --------------------
print("\n--- Starting Inference Demo ---")
# Get evaluation batch
input_ids, segment_ids, masked_tokens, masked_pos, is_next = make_batch(
    batch_size=1,
    max_len=config['max_len'],
    max_mask=config['max_mask']
)

# Reconstruct the input sentence text using id2word
input_tokens = [id2word[w.item()] for w in input_ids[0] if w.item() != word2id['[PAD]']]
print("\nInput sentence (w/o [PAD]):")
print(' '.join(input_tokens))

# Model forward pass
model.eval()
with torch.no_grad():
    mlm_logits, nsp_logits = model(input_ids, segment_ids)

# ---------------- MLM PREDICTION ----------------
# Get the predictions for all tokens
predicted_indices = mlm_logits.argmax(-1)[0]  # Shape: (max_len,)

# Get original and predicted tokens only for the masked positions
original_masked_ids = [t.item() for t in masked_tokens[0] if t.item() != 0]
predicted_masked_ids = []
valid_masked_pos = [p.item() for p in masked_pos[0] if p.item() != 0]

for pos in valid_masked_pos:
    predicted_masked_ids.append(predicted_indices[pos].item())

original_tokens = [id2word[t_id] for t_id in original_masked_ids]
predicted_tokens = [id2word[t_id] for t_id in predicted_masked_ids]

print("\nOriginal masked tokens:", original_tokens)
print("Predicted masked tokens:", predicted_tokens)

# ---------------- NSP PREDICTION ----------------
nsp_prediction = nsp_logits.argmax(-1)[0].item()
print("\nOriginal isNext:", bool(is_next[0]))
print("Predicted isNext:", bool(nsp_prediction))