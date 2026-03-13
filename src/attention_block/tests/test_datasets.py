from ..data.datasets import load_agnews, load_imdb

def test_datasets():
    datasets = {
        "AGNEWS": load_agnews(subset=100),  # small sample for quick test
        # "IMDB": load_imdb(subset=100)
    }

    for name, (train_loader, test_loader) in datasets.items():
        print(f"Dataset: {name}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Take one batch and check shape
        for batch in train_loader:
            x, y = batch
            print(f"  Sample batch X shape: {x.shape}")
            print(f"  Sample batch y shape: {y.shape}")
            break  # only first batch

if __name__ == "__main__":
    test_datasets()