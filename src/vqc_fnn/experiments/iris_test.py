# # =======================
# # Reproducibility utils
# # =======================
# import random
# import numpy as np
# import torch

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     # Make PyTorch deterministic
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# # =======================
# # Imports
# # =======================
# from torch.utils.data import TensorDataset, DataLoader, random_split
# from torch import nn, optim
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import time

# from models.hybrid_qnn import HybridQNN


# # =======================
# # Data loading
# # =======================
# def get_iris_loaders(batch_size=16, val_ratio=0.2, seed=42):
#     """
#     Load the Iris dataset and return deterministic PyTorch DataLoaders.
#     """
#     iris = load_iris()
#     X = iris.data
#     y = iris.target

#     # Standardize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
#     y_tensor = torch.tensor(y, dtype=torch.long)

#     dataset = TensorDataset(X_tensor, y_tensor)

#     # Dataset split sizes
#     val_size = int(len(dataset) * val_ratio)
#     test_size = int(len(dataset) * 0.2)
#     train_size = len(dataset) - val_size - test_size

#     # Seeded generator for deterministic splits & shuffling
#     generator = torch.Generator().manual_seed(seed)

#     train_ds, val_ds, test_ds = random_split(
#         dataset,
#         [train_size, val_size, test_size],
#         generator=generator
#     )

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,
#         generator=generator
#     )
#     val_loader = DataLoader(val_ds, batch_size=batch_size)
#     test_loader = DataLoader(test_ds, batch_size=batch_size)

#     return train_loader, val_loader, test_loader


# # =======================
# # Evaluation
# # =======================
# def evaluate_model(model, loader, device):
#     """
#     Evaluate the model on a given DataLoader.
#     """
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     total_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for X, y in loader:
#             X, y = X.to(device), y.to(device)
#             outputs = model(X)
#             loss = criterion(outputs, y)

#             total_loss += loss.item() * y.size(0)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == y).sum().item()
#             total += y.size(0)

#     avg_loss = total_loss / total
#     accuracy = correct / total
#     return avg_loss, accuracy


# # =======================
# # Training
# # =======================
# def train_hybrid_qnn(
#     n_epochs=10,
#     batch_size=16,
#     learning_rate=0.01,
#     seed=42
# ):
#     """
#     Train the HybridQNN on the Iris dataset (deterministic).
#     """
#     set_seed(seed)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     train_loader, val_loader, test_loader = get_iris_loaders(
#         batch_size=batch_size,
#         seed=seed
#     )

#     num_features = train_loader.dataset[0][0].shape[0]
#     num_classes = 3

#     model = HybridQNN(
#         num_features=num_features,
#         num_classes=num_classes,
#         n_vqc_layers=4
#     ).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.CrossEntropyLoss()

#     history = {
#         'train_loss': [],
#         'val_loss': [],
#         'train_acc': [],
#         'val_acc': []
#     }

#     start_time = time.time()

#     for epoch in range(1, n_epochs + 1):
#         model.train()
#         total_loss = 0.0
#         correct = 0
#         total = 0

#         for X, y in train_loader:
#             X, y = X.to(device), y.to(device)

#             optimizer.zero_grad()
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item() * y.size(0)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == y).sum().item()
#             total += y.size(0)

#         train_loss = total_loss / total
#         train_acc = correct / total
#         val_loss, val_acc = evaluate_model(model, val_loader, device)

#         history['train_loss'].append(train_loss)
#         history['val_loss'].append(val_loss)
#         history['train_acc'].append(train_acc)
#         history['val_acc'].append(val_acc)

#         print(
#             f"Epoch {epoch:02d}/{n_epochs} | "
#             f"Train Loss: {train_loss:.4f} | "
#             f"Train Acc: {train_acc:.4f} | "
#             f"Val Loss: {val_loss:.4f} | "
#             f"Val Acc: {val_acc:.4f}"
#         )

#     end_time = time.time()
#     print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

#     test_loss, test_acc = evaluate_model(model, test_loader, device)
#     print(f"\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

#     return model, history, n_epochs


# # =======================
# # Plotting
# # =======================
# def plot_history(history, n_epochs):
#     """
#     Plot training and validation accuracy.
#     """
#     epochs = range(1, n_epochs + 1)
#     plt.figure(figsize=(10, 4))

#     plt.plot(epochs, history['train_acc'], label='Train Acc')
#     plt.plot(epochs, history['val_acc'], label='Val Acc')

#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy vs Epoch')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# # =======================
# # Main
# # =======================
# if __name__ == "__main__":
#     model, history, n_epochs = train_hybrid_qnn(
#         n_epochs=5,
#         batch_size=8,
#         learning_rate=0.01,
#         seed=42
#     )
#     plot_history(history, n_epochs)

# import torch
# from torch import nn, optim
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# from models.hybrid_qnn import HybridQNN  # assume your HybridQNN is in models/hybrid_qnn.py
# import numpy as np

# # -------------------------------
# # 0. Set seeds for reproducibility
# # -------------------------------
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# # -------------------------------
# # 1. Load and preprocess dataset
# # -------------------------------
# iris = load_iris()
# X = iris.data  # shape (150, 4)
# y = iris.target  # shape (150,)

# # Scale features to [0, pi] for angle embedding
# scaler = MinMaxScaler(feature_range=(0, np.pi))
# X_scaled = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
# )
# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)
# y_test = torch.tensor(y_test, dtype=torch.long)

# # -------------------------------
# # 2. Define HybridQNN
# # -------------------------------
# num_features = X_train.shape[1]
# num_classes = len(set(y_train.numpy()))
# model = HybridQNN(
#     num_features=num_features,
#     num_classes=num_classes,
#     n_vqc_layers=8,
#     embedding_type="angle",  # angle embedding for classical data
#     gate_type="Y",
#     # device_type="default.qubit",  # QNode device
# )

# # -------------------------------
# # 3. Training setup
# # -------------------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# epochs = 20

# train_acc_list = []
# test_acc_list = []

# # -------------------------------
# # 4. Training loop
# # -------------------------------
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     logits = model(X_train)
#     loss = criterion(logits, y_train)
#     loss.backward()
#     optimizer.step()

#     # Training accuracy
#     preds_train = torch.argmax(logits, dim=1)
#     train_acc = (preds_train == y_train).float().mean().item()
#     train_acc_list.append(train_acc)

#     # Test accuracy
#     model.eval()
#     with torch.no_grad():
#         logits_test = model(X_test)
#         preds_test = torch.argmax(logits_test, dim=1)
#         test_acc = (preds_test == y_test).float().mean().item()
#         test_acc_list.append(test_acc)

#     print(
#         f"Epoch {epoch+1:02d}/{epochs} | Loss: {loss.item():.4f} "
#         f"| Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%"
#     )

# # -------------------------------
# # 5. Visualize accuracy
# # -------------------------------
# plt.figure(figsize=(8,5))
# plt.plot(train_acc_list, label="Train Accuracy")
# plt.plot(test_acc_list, label="Test Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Hybrid QNN on Iris Dataset")
# plt.legend()
# plt.grid(True)
# plt.show()

# # -------------------------------
# # 6. Test a single sample output
# # -------------------------------
# sample_idx = 0
# model.eval()
# with torch.no_grad():
#     sample_logits = model(X_test[sample_idx])
#     predicted_class = torch.argmax(sample_logits).item()
#     true_class = y_test[sample_idx].item()
#     print(f"\nSample {sample_idx} prediction -> Predicted: {predicted_class}, True: {true_class}")

# # -------------------------------
# # 7. Final test accuracy in percentage
# # -------------------------------
# model.eval()
# with torch.no_grad():
#     logits_test = model(X_test)
#     preds_test = torch.argmax(logits_test, dim=1)
#     final_accuracy_percent = (preds_test == y_test).float().mean().item() * 100

# print(f"\nFinal Test Accuracy: {final_accuracy_percent:.2f}%")


# =======================
# Reproducibility utils
# =======================
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =======================
# Imports
# =======================
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn, optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

from models.hybrid_qnn import HybridQNN


# =======================
# Data loading
# =======================
def get_iris_loaders(batch_size=16, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Load the Iris dataset and return deterministic PyTorch DataLoaders.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    # Dataset split sizes
    val_size = int(len(dataset) * val_ratio)
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - val_size - test_size

    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# =======================
# Evaluation
# =======================
def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item() * y.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# =======================
# Training
# =======================
def train_hybrid_qnn(
    n_epochs=10,
    batch_size=16,
    learning_rate=0.01,
    seed=42
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_iris_loaders(batch_size=batch_size, seed=seed)

    num_features = train_loader.dataset.dataset[0][0].shape[0]  # Access underlying dataset
    num_classes = 3

    model = HybridQNN(
        num_features=num_features,
        num_classes=num_classes,
        n_vqc_layers=6,  # Reasonable for Iris dataset
        embedding_type="angle",
        gate_type="Y",
        encoding_strategy="reupload",
        device_type="default.qubit"
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    return model, history, n_epochs


# =======================
# Plotting
# =======================
def plot_history(history, n_epochs):
    epochs = range(1, n_epochs + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =======================
# Main
# =======================
if __name__ == "__main__":
    model, history, n_epochs = train_hybrid_qnn(
        n_epochs=10,
        batch_size=8,
        learning_rate=0.01,
        seed=42
    )
    plot_history(history, n_epochs)
