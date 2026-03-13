# src/tests/test_dataset_loading.py

from ..data.datasets import load_agnews

def test_agnews_loader():
    train_loader, test_loader, num_classes = load_agnews(max_seq_len=16, batch_size=2, subset=10)

    # Check first batch
    batch = next(iter(train_loader))
    inputs, labels = batch

    print("Input batch shape:", inputs.shape)
    print("Label batch shape:", labels.shape)
    print("Number of classes:", num_classes)
    print("Sample tokens:", inputs[0])
    print("Sample label:", labels[0])

if __name__ == "__main__":
    test_agnews_loader()