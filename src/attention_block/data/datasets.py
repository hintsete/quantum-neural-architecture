# # src/data/datasets.py

# import torch
# from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset
# from typing import List, Tuple, Callable, Optional
# from transformers import AutoTokenizer


# class TextClassificationDataset(Dataset):
#     """
#     Generic text classification dataset for PyTorch.

#     Args:
#         texts (List[str]): List of text samples
#         labels (List[int]): Corresponding labels
#         tokenizer (Callable): Tokenizer function returning token IDs
#         max_seq_len (int): Maximum sequence length for padding/truncation
#     """
#     def __init__(
#         self,
#         texts: List[str],
#         labels: List[int],
#         tokenizer: Callable,
#         max_seq_len: int = 128
#     ):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_seq_len = max_seq_len

#     def __len__(self) -> int:
#         return len(self.texts)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Tokenizer must return numeric IDs
#         tokens = self.tokenizer(self.texts[idx])
        
#         # Some tokenizers return dict (like HuggingFace), take 'input_ids'
#         if isinstance(tokens, dict) and "input_ids" in tokens:
#             tokens = tokens["input_ids"]

#         # Truncate/pad to max_seq_len
#         tokens = tokens[:self.max_seq_len]
#         padding_len = self.max_seq_len - len(tokens)
#         tokens = tokens + [0] * padding_len  # 0 is PAD token

#         return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# # -----------------------------
# # Simple whitespace tokenizer fallback
# # -----------------------------
# def simple_tokenizer(text: str) -> List[int]:
#     """
#     Basic whitespace tokenizer that maps words to integers on-the-fly.
#     """
#     tokens = text.lower().split()
#     # Simple word2idx mapping
#     word2idx = {"<PAD>": 0}
#     idxs = []
#     for token in tokens:
#         if token not in word2idx:
#             word2idx[token] = len(word2idx)
#         idxs.append(word2idx[token])
#     return idxs


# # -----------------------------
# # Dataset loader functions
# # -----------------------------
# def load_agnews(
#     tokenizer: Optional[Callable] = None,
#     max_seq_len: int = 128,
#     batch_size: int = 64,
#     subset: int = 1000
# ) -> Tuple[DataLoader, DataLoader, int]:
#     """
#     Load AG_NEWS dataset (HuggingFace) as PyTorch DataLoaders.
#     Only loads first `subset` samples for quick testing.

#     Returns:
#         train_loader, test_loader, num_classes
#     """
#     # Load dataset from HuggingFace
#     dataset = load_dataset("ag_news")

#     # Use HuggingFace BERT tokenizer if none provided
#     if tokenizer is None:
#         hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#         tokenizer = lambda text: hf_tokenizer(text, truncation=True, max_length=max_seq_len, padding=False)

#     # Extract texts and labels
#     train_texts = dataset["train"]["text"][:subset]
#     train_labels = dataset["train"]["label"][:subset]
#     test_texts = dataset["test"]["text"][:subset]
#     test_labels = dataset["test"]["label"][:subset]

#     # Number of classes
#     num_classes = len(set(train_labels + test_labels))

#     # Create PyTorch datasets
#     train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_len)
#     test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_seq_len)

#     # DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader, num_classes


# # Optionally, IMDB loader can be added similarly
# def load_imdb(
#     tokenizer: Optional[Callable] = None,
#     max_seq_len: int = 128,
#     batch_size: int = 64,
#     subset: int = 1000
# ) -> Tuple[DataLoader, DataLoader, int]:
#     """
#     Load IMDB dataset (HuggingFace) as PyTorch DataLoaders.
#     Only loads first `subset` samples for quick testing.

#     Returns:
#         train_loader, test_loader, num_classes
#     """
#     dataset = load_dataset("imdb")

#     if tokenizer is None:
#         hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#         tokenizer = lambda text: hf_tokenizer(text, truncation=True, max_length=max_seq_len, padding=False)

#     train_texts = dataset["train"]["text"][:subset]
#     train_labels = dataset["train"]["label"][:subset]
#     test_texts = dataset["test"]["text"][:subset]
#     test_labels = dataset["test"]["label"][:subset]

#     num_classes = len(set(train_labels + test_labels))

#     train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_len)
#     test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_seq_len)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader, num_classes

# src/data/datasets.py
# src/attention_block/data/datasets.py
import os
import pickle
from typing import List, Tuple, Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# -----------------------------
# Cache directory
# -----------------------------
CACHE_DIR = "./cached_datasets"
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# Dataset class
# -----------------------------
class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: Callable, max_seq_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(self.texts[idx])

        # Ensure it's a Python list of ints
        if hasattr(tokens, "ids"):  # fast tokenizer
            tokens = tokens.ids
        elif isinstance(tokens, dict) and "input_ids" in tokens:
            tokens = tokens["input_ids"]
        tokens = list(tokens)

        # Pad/truncate
        tokens = tokens[:self.max_seq_len]
        padding_len = self.max_seq_len - len(tokens)
        tokens = tokens + [0] * padding_len  # 0 = PAD token

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# -----------------------------
# HuggingFace tokenizer wrapper
# -----------------------------
def get_tokenizer(max_seq_len: int) -> Callable:
    hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def _tokenize(text: str):
        # return numeric token IDs
        enc = hf_tokenizer(text, truncation=True, max_length=max_seq_len, padding=False)
        if hasattr(enc, "ids"):
            return enc.ids
        return enc["input_ids"]

    return _tokenize


# -----------------------------
# Download or load cached dataset
# -----------------------------
def download_or_load_agnews(max_seq_len: int = 128, subset: Optional[int] = None) -> DatasetDict:
    cache_path = os.path.join(CACHE_DIR, f"agnews_{max_seq_len}.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached AG_NEWS from {cache_path}")
        with open(cache_path, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("Downloading AG_NEWS from HuggingFace...")
        dataset = load_dataset("ag_news")
        if subset is not None:
            dataset["train"] = dataset["train"].select(range(subset))
            dataset["test"] = dataset["test"].select(range(subset))
        with open(cache_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved dataset to cache: {cache_path}")
    return dataset


# -----------------------------
# Load DataLoader
# -----------------------------
def load_agnews(tokenizer: Optional[Callable] = None,
                max_seq_len: int = 128,
                batch_size: int = 64,
                subset: Optional[int] = None) -> Tuple[DataLoader, DataLoader, int]:
    if tokenizer is None:
        tokenizer = get_tokenizer(max_seq_len)

    dataset = download_or_load_agnews(max_seq_len=max_seq_len, subset=subset)

    # Convert Columns to Python lists
    train_texts = list(dataset["train"]["text"])
    train_labels = list(dataset["train"]["label"])
    test_texts = list(dataset["test"]["text"])
    test_labels = list(dataset["test"]["label"])

    num_classes = len(set(train_labels + test_labels))

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_len)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_classes