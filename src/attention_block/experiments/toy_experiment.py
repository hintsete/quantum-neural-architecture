import torch
from torch.utils.data import DataLoader
from ..data.datasets import TextClassificationDataset, get_tokenizer
from ..models.q_transformer import QTransformer
from ..training.trainer import Trainer
from ..utils.config import Config, set_seed

# -----------------------------
# Set seeds
# -----------------------------
set_seed(Config.seed)

# -----------------------------
# Tiny toy dataset
# -----------------------------
texts = ["hello world", "quantum transformer", "test sentence", "another example"]
labels = [0, 1, 0, 1]
tokenizer = get_tokenizer(max_seq_len=8)

train_dataset = TextClassificationDataset(texts, labels, tokenizer, max_seq_len=8)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)  # reuse train as test

# -----------------------------
# Initialize Quantum Transformer
# -----------------------------
model = QTransformer(
    vocab_size=Config.vocab_size,
    embed_dim=Config.embed_dim,
    n_qubits=2,       # smaller for CPU
    n_layers=1,       # single ansatz layer
    device=Config.device,
    q_device=Config.q_device,
    num_blocks=1,     # single transformer block for testing
    num_classes=2,
    ff_hidden_dim=32,
    max_seq_len=8,
    dropout=0.1
)

model.to(Config.device)

# -----------------------------
# Initialize Trainer
# -----------------------------
trainer = Trainer(model=model, device=Config.device, lr=1e-3, weight_decay=0.0)

# -----------------------------
# Train for 1 epoch (quick test)
# -----------------------------
history = trainer.fit(train_loader, test_loader, epochs=10, print_every=1)

# -----------------------------
# Evaluate
# -----------------------------
metrics = trainer.evaluate(test_loader)
print("\nToy Dataset Test Metrics:", metrics)