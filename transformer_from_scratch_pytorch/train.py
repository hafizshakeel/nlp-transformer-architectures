import torch
import torch.nn as nn
from pathlib import Path
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import get_ds, causal_mask
from model import build_transformer


# -------------------- Model Loading  --------------------

def get_model(config, vocab_src_len, vocal_trg_len):
    model = build_transformer(vocab_src_len, vocal_trg_len, config["seq_len"], config["seq_len"], d_model=config["d_model"])
    return model

# -------------------- Greedy Decoding  --------------------

# Generate translation step-by-step for inference / validation.
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_trg, max_len, device):
    sos_idx = tokenizer_trg.token_to_id('[SOS]')
    eos_idx = tokenizer_trg.token_to_id('[EOS]')

    # Pass the source sentence through the encoder only once to get its representation.
    encoder_output = model.encode(source, source_mask)
    # encoder_output: (1, src_seq_len, d_model), contains encoded information about the source sentence

    # Start the decoder input with only the [SOS] token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    # decoder_input: shape (1, 1), starts with [SOS] token only

    # Start generating tokens one-by-one
    while True:
        # Stop if max length reached (to avoid infinite loops in case of mistakes)
        if decoder_input.size(1) == max_len:
            break

        # Build causal mask for current decoder input length
        # Mask ensures decoder can't look at future tokens (autoregressive behavior)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # decoder_mask: shape (1, seq_len, seq_len)

        # Pass through decoder to get hidden states for current decoder input
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # out: (1, seq_len, d_model)

        # Project decoder output to vocab size to get logits for the next token
        prob = model.project(out[:, -1])
        # out[:, -1]: get the last time step's output (latest token)
        # prob: (1, vocab_size), logits over the vocabulary for the next token

        # Pick the most likely next token (greedy choice, no sampling)
        _, next_word = torch.max(prob, dim=1)
        # next_word: shape (1,), contains index of the most probable next token

        # Append the predicted token to decoder input for the next iteration
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        # decoder_input grows by 1 token every iteration

        # If the predicted token is EOS, stop early
        if next_word == eos_idx:
            break

    # Return the generated tokens excluding the batch dimension
    # Shape becomes (seq_len,)
    return decoder_input.squeeze(0)


# -------------------- Validation --------------------

def run_validation(model, validation_ds, tokenizer_src, tokenizer_trg, max_len, device, print_msg, global_step, writer,
                   num_examples=2):
    # This function runs validation on a few examples, prints source/target/predicted, and logs metrics to TensorBoard.
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_trg, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["trg_text"][0]
            model_out_text = tokenizer_trg.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


# -------------------- Training --------------------

def train_model(config):
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # make sure weight folder is created
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloder, val_dataloader, tokenizer_src, tokenizer_trg = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()).to(device)

    # tensorboard
    writer = SummaryWriter(config["experiment_name"])

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # resume training if model crashes
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    # Label smoothing is a regularization technique commonly used in training transformers for NLP tasks, especially for
    # sequence-to-sequence tasks like translation. Label smoothing helps produce softer, more robust probabilities.

    for epoch in range(initial_epoch, config['num_epochs']):

        batch_iterator = tqdm(train_dataloder, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            model.train()

            # Move tensors to a device
            encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (batch, 1, seq_len, seq_len)

            # Forward pass-through model - run the tensors though the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, trg_vocab_size)

            label = batch['label'].to(device)  # (batch, seq_len)

            # (batch, seq_len, trg_vocab_size) --> (batch * seq_len, trg_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item()}: .6.3f"})  # updates the tqdm progress bar

            # log the loss to tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()  #  Write all pending logs to disk immediately

            # Backward and update wights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step +=1

        # Validate and show predictions at the end of each epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_trg, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # save the model checkpoints the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    train_model(config=get_config())







