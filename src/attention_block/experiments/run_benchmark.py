# src/experiments/run_benchmark.py

import torch
from ..utils.config import Config, set_seed
from ..utils.logging import setup_logger
from ..data.datasets import load_agnews
from ..models.q_transformer import QTransformer
from ..models.classical_transformer import ClassicalTransformer
from ..training.trainer import Trainer


def run_experiment(model_type: str = "quantum", subset: int = None):
    """
    Run a single experiment with Quantum or Classical Transformer on AG_NEWS.

    Args:
        model_type (str): "quantum" or "classical"
        subset (int): Number of samples to use from the training dataset (None = full dataset)
    Returns:
        metrics (dict): Dictionary containing 'accuracy' and 'f1' scores
    """
    # -----------------------------
    # Set seeds for reproducibility
    # -----------------------------
    set_seed(Config.seed)

    # -----------------------------
    # Setup logger
    # -----------------------------
    logger = setup_logger(f"logs/agnews_{model_type}.log")
    logger.info(f"Running experiment with {model_type.capitalize()} Transformer on AG_NEWS")

    # -----------------------------
    # Load AG_NEWS dataset
    # -----------------------------
    train_loader, test_loader, num_classes = load_agnews(
        max_seq_len=Config.max_seq_len,
        batch_size=Config.batch_size,
        subset=subset  # Use None for full dataset
    )
    val_loader = test_loader  # For simplicity, use test as validation

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # -----------------------------
    # Initialize model
    # -----------------------------
    if model_type.lower() == "quantum":
        model = QTransformer(
            vocab_size=Config.vocab_size,
            embed_dim=Config.embed_dim,
            n_qubits=Config.n_qubits,
            n_layers=Config.n_layers,
            device=Config.device,
            q_device=Config.q_device,
            num_blocks=2,
            num_classes=num_classes,
            ff_hidden_dim=128,
            max_seq_len=Config.max_seq_len,
            dropout=Config.dropout
        )
    else:
        model = ClassicalTransformer(
            vocab_size=Config.vocab_size,
            embed_dim=Config.embed_dim,
            num_blocks=2,
            num_classes=num_classes,
            ff_hidden_dim=128,
            max_seq_len=Config.max_seq_len,
            dropout=Config.dropout
        )

    model.to(Config.device)

    # -----------------------------
    # Initialize trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        device=Config.device,
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )

    # -----------------------------
    # Train model
    # -----------------------------
    logger.info(f"Training {model_type.capitalize()} Transformer")
    trainer.fit(train_loader, val_loader, epochs=Config.epochs, print_every=1)

    # -----------------------------
    # Evaluate model
    # -----------------------------
    metrics = trainer.evaluate(test_loader)
    logger.info(
        f"{model_type.capitalize()} Transformer | Test Accuracy: {metrics['accuracy']:.4f} | "
        f"F1 Score: {metrics['f1']:.4f}"
    )

    return metrics


if __name__ == "__main__":
    results = {}

    # -----------------------------
    # USER-SELECTED SUBSET FOR CPU TESTING
    # -----------------------------
    # Set subset=None to run full dataset (120k for AG_NEWS)
    # Recommended on CPU: small subset like 1000–2000
    SUBSET_SIZE = 1000

    results["AGNEWS_quantum"] = run_experiment(model_type="quantum", subset=SUBSET_SIZE)
    results["AGNEWS_classical"] = run_experiment(model_type="classical", subset=SUBSET_SIZE)

    # -----------------------------
    # Print benchmark summary
    # -----------------------------
    print("\n--- Benchmark Summary ---")
    for key, metrics in results.items():
        print(f"{key}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

# # experiments/run_benchmark.py

# import torch
# from utils.config import Config, set_seed
# from utils.metrics import accuracy_score, f1_score
# from utils.logging import setup_logger
# from data.datasets import load_dataset
# from models.q_transformer import QTransformer
# from models.classical_transformer import ClassicalTransformer
# from training.trainer import Trainer

# def run_experiment(dataset_name: str, model_type: str = "quantum"):
#     """
#     Run a full benchmark on a dataset with either quantum or classical transformer.

#     Args:
#         dataset_name (str): Name of the dataset ("AGNEWS", "IMDB", "SST-2", "SST-5")
#         model_type (str): "quantum" or "classical"

#     Returns:
#         dict: Dictionary with evaluation metrics (accuracy, f1)
#     """
#     # Set seeds for reproducibility
#     set_seed(Config.seed)

#     # Setup logger
#     logger = setup_logger(f"logs/{dataset_name}_{model_type}.log")

#     logger.info(f"Loading dataset: {dataset_name}")
#     train_loader, val_loader, test_loader, num_classes = load_dataset(
#         dataset_name, batch_size=Config.batch_size
#     )

#     # Initialize model
#     logger.info(f"Initializing {model_type} transformer")
#     if model_type.lower() == "quantum":
#         model = QTransformer(
#             embed_dim=64, 
#             n_qubits=Config.n_qubits, 
#             n_layers=Config.n_layers, 
#             max_seq_len=Config.max_seq_len,
#             num_classes=num_classes,
#             device=Config.device
#         )
#     else:
#         model = ClassicalTransformer(
#             embed_dim=64,
#             num_heads=4,  # can match quantum param budget
#             num_layers=2,
#             num_classes=num_classes,
#             max_seq_len=Config.max_seq_len,
#             device=Config.device
#         )

#     model.to(Config.device)

#     # Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         test_loader=test_loader,
#         epochs=Config.epochs,
#         lr=Config.learning_rate,
#         device=Config.device,
#         logger=logger
#     )

#     # Train model
#     logger.info(f"Training {model_type} transformer on {dataset_name}")
#     trainer.train()

#     # Evaluate on test set
#     logger.info(f"Evaluating {model_type} transformer on {dataset_name} test set")
#     y_true, y_pred = trainer.evaluate(test_loader)

#     acc = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average='macro')

#     logger.info(f"{model_type.capitalize()} Transformer | Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

#     return {"accuracy": acc, "f1": f1}


# if __name__ == "__main__":
#     datasets = ["AGNEWS", "IMDB", "SST-2", "SST-5"]

#     results = {}
#     for dataset in datasets:
#         # Quantum Transformer
#         results[f"{dataset}_quantum"] = run_experiment(dataset, model_type="quantum")
#         # Classical Transformer
#         results[f"{dataset}_classical"] = run_experiment(dataset, model_type="classical")

#     # Print summary
#     print("\n--- Benchmark Summary ---")
#     for key, metrics in results.items():
#         print(f"{key}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")


# # experiments/run_benchmark.py
# # src/attention_block/experiments/run_benchmark.py

# import torch
# import pennylane as qml

# from ..utils.config import Config, set_seed
# from ..utils.metrics import accuracy_score, f1_score
# from ..utils.logging import setup_logger

# from ..data.datasets import load_dataset
# from ..data.toy_dataset import load_toy_dataset

# from ..models.q_transformer import QTransformer
# from ..models.classical_transformer import ClassicalTransformer

# from ..training.trainer import Trainer


# def run_experiment(dataset_name: str, model_type: str = "quantum"):
#     """
#     Runs a full training + evaluation pipeline.

#     Args:
#         dataset_name (str):
#             "TOY", "AGNEWS", "IMDB", "SST-2", "SST-5"

#         model_type (str):
#             "quantum" or "classical"

#     Returns:
#         dict with accuracy and f1 score
#     """

#     # ----------------------------
#     # Reproducibility
#     # ----------------------------
#     set_seed(Config.seed)

#     logger = setup_logger(f"logs/{dataset_name}_{model_type}.log")

#     logger.info(f"Loading dataset: {dataset_name}")

#     # ----------------------------
#     # Dataset selection
#     # ----------------------------
#     if dataset_name == "TOY":

#         train_loader, val_loader, test_loader, num_classes = load_toy_dataset(
#             batch_size=Config.batch_size
#         )

#         vocab_size = 100

#     else:

#         train_loader, val_loader, test_loader, num_classes = load_dataset(
#             dataset_name,
#             batch_size=Config.batch_size,
#         )

#         vocab_size = Config.vocab_size

#     # ----------------------------
#     # Model initialization
#     # ----------------------------
#     logger.info(f"Initializing {model_type} transformer")

#     if model_type == "quantum":

#         # Create PennyLane quantum device
#         q_device = qml.device("default.qubit", wires=Config.n_qubits)

#         model = QTransformer(
#             vocab_size=vocab_size,
#             embed_dim=Config.embed_dim,
#             n_qubits=Config.n_qubits,
#             n_layers=Config.n_layers,
#             device=q_device,
#             num_blocks=2,
#             num_classes=num_classes,
#             max_seq_len=Config.max_seq_len,
#             dropout=Config.dropout,
#         )

#     else:

#         model = ClassicalTransformer(
#             vocab_size=vocab_size,
#             embed_dim=Config.embed_dim,
#             num_blocks=2,
#             num_classes=num_classes,
#             max_seq_len=Config.max_seq_len,
#         )

#     model = model.to(Config.device)

#     # ----------------------------
#     # Trainer
#     # ----------------------------
#     trainer = Trainer(
#         model=model,
#         device=Config.device,
#         lr=Config.learning_rate,
#         weight_decay=Config.weight_decay,
#     )

#     # ----------------------------
#     # Training
#     # ----------------------------
#     logger.info("Starting training")

#     trainer.fit(
#         train_loader=train_loader,
#         val_loader=val_loader,
#         epochs=Config.epochs,
#     )

#     # ----------------------------
#     # Evaluation
#     # ----------------------------
#     logger.info("Evaluating on test set")

#     test_metrics = trainer.evaluate(test_loader)

#     acc = test_metrics["accuracy"]
#     f1 = test_metrics["f1"]

#     logger.info(
#         f"{model_type.capitalize()} Transformer | "
#         f"Accuracy: {acc:.4f} | F1: {f1:.4f}"
#     )

#     return {"accuracy": acc, "f1": f1}


# if __name__ == "__main__":

#     datasets = ["TOY"]

#     results = {}

#     for dataset in datasets:

#         results[f"{dataset}_quantum"] = run_experiment(
#             dataset,
#             model_type="quantum",
#         )

#         results[f"{dataset}_classical"] = run_experiment(
#             dataset,
#             model_type="classical",
#         )

#     print("\n--- Benchmark Summary ---")

#     for key, metrics in results.items():

#         print(
#             f"{key}: "
#             f"Accuracy={metrics['accuracy']:.4f}, "
#             f"F1={metrics['f1']:.4f}"
#         )