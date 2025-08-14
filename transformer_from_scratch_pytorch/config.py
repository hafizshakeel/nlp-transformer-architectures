import argparse
from pathlib import Path

def get_config():
    parser = argparse.ArgumentParser(description="Transformer Training Config")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=320, help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="Embedding/hidden dimension")

    # Language settings
    parser.add_argument("--lang_src", type=str, default="en", help="Source language")
    parser.add_argument("--lang_trg", type=str, default="it", help="Target language")

    # Paths & filenames
    parser.add_argument("--model_folder", type=str, default="weights", help="Folder to save model weights")
    parser.add_argument("--model_basename", type=str, default="tmodel_", help="Base name for model checkpoint files")
    parser.add_argument("--tokenizer_file", type=str, default="tokenizer_{0}.json", help="Tokenizer file pattern")
    parser.add_argument("--experiment_name", type=str, default="runs/tmodel", help="TensorBoard experiment folder")

    # Checkpoint loading
    parser.add_argument(
        "--preload",
        type=str,
        default=None,
        help="Checkpoint to preload ('latest', epoch number like '05', or None)"
    )

    args = parser.parse_args()
    return vars(args)  # return as a dict just like before


def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    weights_files = list(Path(model_folder).glob(f"{model_basename}*.pt"))
    if len(weights_files) == 0:
        return None
    weights_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return str(weights_files[-1])
