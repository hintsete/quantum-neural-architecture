# src/experiments/run_benchmark_mini.py

import torch
from torch.utils.data import DataLoader
from ..data.datasets import TextClassificationDataset, get_tokenizer, download_or_load_agnews
from ..models.q_transformer import QTransformer
from ..models.classical_transformer import ClassicalTransformer
from ..training.trainer import Trainer
from ..utils.config import Config, set_seed

# -----------------------------
# Configuration tweaks for CPU-safe mini benchmark
# -----------------------------
set_seed(Config.seed)

SUBSET_SIZE = 200       # Small subset to run quickly on CPU
MAX_SEQ_LEN = 32        # Short sequences for faster computation
BATCH_SIZE = 8          # Small batch size to reduce computation
N_QUBITS = 4            # Quantum circuit qubits
N_LAYERS = 1            # Ansätze layers
NUM_BLOCKS = 1          # Transformer blocks
EPOCHS = 3              # Small number of epochs

# -----------------------------
# Load small subset of AG_NEWS
# -----------------------------
tokenizer = get_tokenizer(max_seq_len=MAX_SEQ_LEN)
dataset = download_or_load_agnews(max_seq_len=MAX_SEQ_LEN, subset=SUBSET_SIZE)

train_texts = list(dataset["train"]["text"])
train_labels = list(dataset["train"]["label"])
test_texts = list(dataset["test"]["text"])
test_labels = list(dataset["test"]["label"])
num_classes = len(set(train_labels + test_labels))

train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_len=MAX_SEQ_LEN)
test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_seq_len=MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Helper function to run experiments
# -----------------------------
def run_experiment(model_type: str = "quantum"):
    print(f"\n--- Running {model_type.capitalize()} Transformer ---")

    if model_type.lower() == "quantum":
        model = QTransformer(
            vocab_size=Config.vocab_size,
            embed_dim=Config.embed_dim,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            device=Config.device,
            q_device=Config.q_device,
            num_blocks=NUM_BLOCKS,
            num_classes=num_classes,
            ff_hidden_dim=128,
            max_seq_len=MAX_SEQ_LEN,
            dropout=Config.dropout
        )
    else:
        model = ClassicalTransformer(
            vocab_size=Config.vocab_size,
            embed_dim=Config.embed_dim,
            num_blocks=NUM_BLOCKS,
            num_classes=num_classes,
            ff_hidden_dim=128,
            max_seq_len=MAX_SEQ_LEN,
            dropout=Config.dropout
        )

    model.to(Config.device)

    trainer = Trainer(
        model=model,
        device=Config.device,
        lr=3e-4,
        weight_decay=0.0
    )

    # Train
    history = trainer.fit(train_loader, test_loader, epochs=EPOCHS, print_every=1)

    # Evaluate
    metrics = trainer.evaluate(test_loader)
    print(f"{model_type.capitalize()} Transformer | Test Metrics: {metrics}")
    return metrics

# -----------------------------
# Run mini benchmark
# -----------------------------
if __name__ == "__main__":
    results = {}
    results["AGNEWS_quantum"] = run_experiment("quantum")
    results["AGNEWS_classical"] = run_experiment("classical")

    print("\n--- Mini Benchmark Summary ---")
    for key, metrics in results.items():
        print(f"{key}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")