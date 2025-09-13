import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="BERT Configuration for educational purposes"
    )

    # Model Architecture
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension size")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=2048, help="Feedforward hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--max_mask", type=int, default=20, help="Maximum number of tokens to mask")

    # Data
    parser.add_argument("--data_path", type=str, default="animal_story.txt", help="Path to training data")

    return parser.parse_args()

def get_config():
    return vars(parse_args())  # convert Namespace -> dict