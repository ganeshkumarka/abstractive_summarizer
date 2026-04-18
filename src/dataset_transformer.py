"""
src/dataset_transformer_v2.py
----------------------------
Dataset for Transformer models (mT5, MuRIL, etc.)

Uses SAME preprocessed pickle files as dataset.py
but converts them back to text for tokenizer-based models.

No conflict with word2vec pipeline.
"""

import os
import pickle
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TransformerDataset(Dataset):
    def __init__(self, samples, tokenizer, max_src=128, max_tgt=40):
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

        # IMPORTANT: convert token IDs back to text
        # because transformer expects text input
        self.data = []

        for s in samples:
            if 'src_tokens' in s and 'tgt_tokens' in s:
                src_text = " ".join(s['src_tokens'])
                tgt_text = " ".join(s['tgt_tokens'])
            else:
                # fallback (if only ids exist)
                continue

            if src_text.strip() and tgt_text.strip():
                self.data.append({
                    "src": src_text,
                    "tgt": tgt_text
                })

        print(f"Transformer dataset: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        model_inputs = self.tokenizer(
            "summarize: " + item["src"],
            max_length=self.max_src,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        labels = self.tokenizer(
            text_target=item["tgt"],
            max_length=self.max_tgt,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0)
        }


def load_transformer_datasets(tokenizer):
    """
    Load train/test datasets for transformer models
    """
    train_path = os.path.join(config.DATA_PROCESSED, "train.pkl")
    test_path  = os.path.join(config.DATA_PROCESSED, "test.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError("Run preprocess.py first")

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)

    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    train_ds = TransformerDataset(train_data, tokenizer)
    test_ds  = TransformerDataset(test_data, tokenizer)

    return train_ds, test_ds