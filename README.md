# NLP Transformer Architectures

This repository contains implementations of various transformer-based architectures for natural language processing (NLP).

## Included Implementations

The repository currently includes the following implementations:

1. [**Transformer**](./Transformer/) - A complete implementation of the original Transformer architecture from the "Attention Is All You Need" paper, designed for machine translation.

2. [**BERT**](./BERT/) - An implementation of BERT (Bidirectional Encoder Representations from Transformers) with MLM and NSP pre-training objectives.

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/hafizshakeel/nlp-transformer-architectures.git
cd nlp-transformer-architectures
```

### Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### Running Different Models

Navigate to the specific model folder and follow the instructions in its README:

#### Transformer

```bash
cd Transformer
python train.py
```

#### BERT

```bash
cd BERT
python train.py
```

## Repository Structure

```
nlp-transformer-architectures/
├── Transformer/        # Transformer implementation for machine translation
├── BERT/               # BERT implementation for language modeling
└── requirements.txt    # Project dependencies
```

Each implementation folder contains its own detailed README with specific information about the model architecture, training procedure, and usage instructions.

## License
This project is licensed under the MIT License.

## Contact
Email: hafizshakeel1997@gmail.com