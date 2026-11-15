from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch


class IMDbDataset(Dataset):

    def __init__(self, split, tokenizer, max_length=256):

        self.dataset = load_dataset(
            "stanfordnlp/imdb",
            split=split
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)

        attention_mask = encoding["attention_mask"].squeeze(0)

        X = torch.stack([input_ids, attention_mask], dim=0)

        y = torch.tensor(sample["label"], dtype=torch.long)

        return X, y


def get_imdb_loaders(batch_size=64, max_length=256):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = IMDbDataset("train", tokenizer, max_length)
    test_dataset = IMDbDataset("test", tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, tokenizer