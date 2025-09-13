import torch
from pathlib import Path
from tokenizers import Tokenizer  # HuggingFace library to load tokenizers
from tokenizers.models import WordLevel  # Word-level tokenizer model
from tokenizers.pre_tokenizers import Whitespace  # Pre-tokenizer to split text by whitespace
from tokenizers.trainers import WordLevelTrainer  # Trainer for the tokenizer

from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split


# -------------------- Tokenization --------------------

# This function is a generator that yields all sentences in a dataset for a given language.
def get_all_sentences(ds, lang):
    for item in ds:  # item --> pair of sentences like { "en": "CHAPTER I", "it": "PARTE PRIMA" }
        yield item['translation'][lang]  # extract the particular language sentences


def get_or_build_tokenizer(config, ds, lang):
    # Loads a tokenizer from a file if it exists, otherwise train a new one on the dataset and save it.
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) #  # e.g., tokenizer_en.json
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel({}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer(Whitespace)
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer


# -------------------- Dataset --------------------

class BilingualDataset(Dataset):
    def __init__(self, ds, src_tokenizer, trg_tokenizer, src_lang, trg_lang, seq_len):
        super().__init__()

        # save the values
        self.seq_len = seq_len

        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_lang = src_lang
        self.trg_lang = trg_lang

        # also, save the special tokens required to create tensor for our model,
        # these tensors can be created using src or trg as they are the same for both the src and trg.
        # also make them long because vocab size could be larger than 32-bit

        # extract ids for the special tokens
        self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        # first, extract the original pairs from the huggingface dataset and extract src and trg text
        # see Helsinki-NLP/opus_books to get the idea,
        # for instance at id -->3 and translation --> { "en": "CHAPTER I", "it": "PARTE PRIMA" }

        src_target_pair = self.ds[index]  # get one row
        src_text = src_target_pair['translation'][self.src_lang]  # Src Text ("CHAPTER I")
        trg_text = src_target_pair['translation'][self.trg_lang]  # Trg Text ("PARTE PRIMA")

        # now transform the text into tokens and then into input ids - in one pass
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.trg_tokenizer.encode(trg_text).ids

        # Add sos, eos, and padding tokens to each sentence to make the length of fixed size.
        # compute number of pad tokens to reach the seq length
        enc_num_pad_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 because <sos> and <eos> will be added later.
        # For example to reach the seq_len of 10: [SOS] + 5 tokens + [EOS] + 3 paddings = 1 + 5 + 1 + 3 = 10

        # For decoder input, only <sos> is needed, and for label only <eos> is required. That's why -1.
        dec_num_pad_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative.
        # Meaning: If the sentence is longer than seq_len - 2,
        # it's too big to fit in the model's expected fixed size (after adding [SOS] and [EOS]).
        if enc_num_pad_tokens < 0 or dec_num_pad_tokens < 0:
            raise ValueError("Sentence is too long")

        # build tensor for the encoder and decoder input and for the label.
        # add <sos> and <eos> tokens
        encoder_input = torch.cat(
            [
                self.sos_token,  # [SOS]
                torch.tensor(enc_input_tokens, dtype=torch.int64),  # token IDs of the sentence
                self.eos_token,  # [EOS]
                torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64),  # padding to reach seq_len
            ],
            dim=0,
        )  # input will be something like [SOS] 12 45 67 [EOS] [PAD] [PAD] ...  (up to seq_len)


        # add only <sos> token for decoder input
        decoder_input = torch.cat(
            [
                self.sos_token, # [SOS]
                torch.tensor(dec_input_tokens, dtype=torch.int64),  # token IDs of the sentence
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64), # padding to reach seq_len
            ],
            dim=0,
        )

        # Add only <eos> token to the label to match the decoder's output
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),  # token IDs of the sentence
                self.eos_token,  # [EOS]
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64), # padding to reach seq_len
            ],
            dim=0,
        )

        # Double-check the size of the tensors to make sure they're all seq_len long
        assert  encoder_input.size(0) == self.seq_len  # tokens along the first dimension (seq_len)
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # return all the tensors to be used in training
        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)

            # then build the mask which we don't want to be seen in self-attention
            "encoder_mask": (encoder_input  != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len) --> Only Mask [PAD] tokens
            # Creates a mask of True / False where True means "this token is not padding." .int(): Convert True/False to 1/0
            # Purpose of encoder mask is to tell the encoder which positions are real tokens (1) and which are padding (0).

            # then the decoder mask is built using a special function to make sure it only looks at the current and past tokens and also ignores the padding tokens
            # first part: Same idea: 1 = real token, 0 = padding. and combines padding mask with causal mask
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len) --> Mask [PAD] + future tokens
            "label": label, # (seq_len)
            "src_text" : src_text,
            "trg_text" : trg_text,
        }


# -------------------- Masking --------------------

# Creates a mask to block future tokens (causal/autoregressive masking).
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)  # Creates upper-triangle matrix of 1s
    return mask == 0  # Converts 0 → True (allowed), 1 → False (masked) before returning (clear grammar)


# -------------------- Data Loading --------------------

def get_ds(config):
    # Load dataset and prepare dataloaders
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_trg"]}', split='train')
    # Give the name of dataset from the huggingface library and choose subset - here we'll subset dynamically --> en-it

    # build source and target tokenizers --> tokenizer_en.json
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_trg = get_or_build_tokenizer(config, ds_raw, config['lang_trg'])

    # split into train and validation sets
    # The above data only comes with training, so we can manually divide the dataset into training and validation
    # keep 90% data for training and rest for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])  # split randomly according
    # to the given size


    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'], config['seq_len'])

    # Find max sentence length in source and target and set seq len slightly larger to cover all sentences and special tokens
    max_len_src = 0
    max_len_trg = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        trg_ids = tokenizer_trg.encode(item['translation'][config['lang_trg']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trg = max(max_len_trg, len(trg_ids))

    print(f'Max length of the source sentence: {max_len_src}')
    print(f'Max length of the target sentence: {max_len_trg}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)  # batch_size 1 to process one sentence at a time during inference

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg









