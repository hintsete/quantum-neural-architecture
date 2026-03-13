import torch
from torch.utils.data import Dataset, DataLoader


class ToyTextDataset(Dataset):
    """
    Very small synthetic dataset to test the training pipeline.

    Each sample is a random token sequence with a random label.
    """

    def __init__(self, num_samples=200, seq_len=10, vocab_size=100, num_classes=2):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_toy_dataset(batch_size=8):
    """
    Returns train/val/test loaders for the toy dataset.
    """

    train_dataset = ToyTextDataset(num_samples=200)
    val_dataset = ToyTextDataset(num_samples=50)
    test_dataset = ToyTextDataset(num_samples=50)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    num_classes = 2

    return train_loader, val_loader, test_loader, num_classes