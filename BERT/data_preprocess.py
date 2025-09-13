# ---------------------------------
# DATA PREPROCESSING FOR BERT
# ---------------------------------
import re      # for regular expressions
import random  # for random picking mask
import torch
import spacy

def get_data_and_vocab(file_path="animal_story.txt"):
    """
    Reads a text file, processes it, builds a vocabulary,
    and returns a batch-making function along with vocab details.
    """
    # Load raw text and split into sentences
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(raw_text)

    # Clean and normalize sentences
    sentences = [sent.text.strip().lower() for sent in doc.sents]

    def clean_sentence(sent):
        # Remove all non-alphabetical characters except spaces
        sent = re.sub(r"[^a-zA-Z\s]", "", sent)
        # Replace multiple spaces with a single space
        sent = re.sub(r"\s+", " ", sent)
        return sent.strip().lower()

    sentences = [clean_sentence(sent) for sent in sentences if sent.strip()]

    # Build Vocabulary (word2id)
    word_list = list(set(" ".join(sentences).split()))
    word2id = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(word_list):
        word2id[w] = i + 4
    id2word = {v: k for k, v in word2id.items()}
    vocab_size = len(word2id)

    # Tokenize all sentences to lists of ids
    token_list = []
    for sentence in sentences:
        token_ids = [word2id[word] for word in sentence.split()]
        token_list.append(token_ids)

    # -------------------------
    # MAKE BATCH FUNCTION
    # -------------------------
    def make_batch(batch_size=6, max_len=128, max_mask=20):
        batch = []
        positive, negative = 0, 0

        # Build batch with 50% next sentence and 50% random
        while positive < batch_size / 2 or negative < batch_size / 2:
            # Ensure we don't pick the last sentence as sentence_a to have a valid next sentence
            idx_a = random.randrange(len(sentences) - 1)
            # 50% chance of being the next sentence, 50% chance of being a random one
            if random.random() < 0.5:
                idx_b = idx_a + 1  # Next sentence
                is_next_label = True
            else:
                idx_b = random.randrange(len(sentences))  # Random sentence
                is_next_label = False

            tokens_a, tokens_b = token_list[idx_a], token_list[idx_b]

            # [CLS] A [SEP] B [SEP]
            input_ids = [word2id['[CLS]']] + tokens_a + [word2id['[SEP]']] + tokens_b + [word2id['[SEP]']]
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

            # Check if sequence is too long
            if len(input_ids) > max_len:
                continue  # Skip this pair

            # ---------------
            # MLM masking
            # ---------------
            n_pred = min(max_mask, max(1, int(round(len(input_ids) * 0.15))))
            candidate_pos = [i for i, token in enumerate(input_ids)
                             if token not in [word2id['[CLS]'], word2id['[SEP]']]]
            random.shuffle(candidate_pos)

            masked_tokens, masked_pos = [], []
            for pos in candidate_pos[:n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(input_ids[pos])
                prob = random.random()
                if prob < 0.8:
                    input_ids[pos] = word2id['[MASK]']
                elif prob < 0.9:
                    # Pick a random word (excluding special tokens)
                    input_ids[pos] = random.randint(4, vocab_size - 1)
                # else keep unchanged

            # Pad input to max_len
            n_pad = max_len - len(input_ids)
            input_ids.extend([word2id['[PAD]']] * n_pad)
            segment_ids.extend([0] * n_pad)

            # Pad masked tokens and positions
            # Use 0 for padding, which will be ignored by the loss function
            if len(masked_tokens) < max_mask:
                masked_tokens.extend([0] * (max_mask - len(masked_tokens)))
            if len(masked_pos) < max_mask:
                masked_pos.extend([0] * (max_mask - len(masked_pos)))

            if is_next_label and positive < batch_size / 2:
                batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
                positive += 1
            elif not is_next_label and negative < batch_size / 2:
                batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
                negative += 1

        # Convert to tensors
        input_ids, segment_ids, masked_tokens, masked_pos, is_next = map(torch.LongTensor, zip(*batch))
        return input_ids, segment_ids, masked_tokens, masked_pos, is_next

    return make_batch, word2id, id2word, vocab_size


# if __name__ == '__main__':
#     make_batch_fn, word2id, id2word, vocab_size = get_data_and_vocab()
#     print(f"Vocabulary Size: {vocab_size}")
#     input_ids, segment_ids, masked_tokens, masked_pos, is_next = make_batch_fn(batch_size=6, max_len=128, max_mask=20)
#     print("Input IDs shape:", input_ids.shape)
#     print("Segment IDs shape:", segment_ids.shape)
#     print("Masked Tokens shape:", masked_tokens.shape)
#     print("Masked Positions shape:", masked_pos.shape)
#     print("Is Next shape:", is_next.shape)