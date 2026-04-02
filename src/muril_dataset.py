"""
src/muril_dataset.py
--------------------
Dataset wrapper that handles MuRIL tokenization alongside the standard
vocab-index encoding.

The problem with the previous approach: we were passing our vocab indices
(range 0-8000) as MuRIL input_ids. MuRIL's vocabulary has ~200k tokens
so this produced completely wrong embeddings.

Correct approach:
  - Keep original preprocessed data (src_ids, src_pos, tgt_ids) for LSTM vocab
  - Store raw source text and tokenize it with MuRIL tokenizer separately
  - MuRIL outputs 768-dim contextual vectors → project to 256-dim
  - These replace Word2Vec embeddings in the encoder

Usage:
    from src.muril_dataset import get_muril_dataloaders
    train_loader, test_loader = get_muril_dataloaders(tokenizer, batch_size=32)
"""

import os, sys, pickle
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MuRILDataset(Dataset):
    """Dataset that provides both vocab indices and MuRIL-tokenized input."""

    def __init__(self, samples, tokenizer, muril_max_len=128):
        self.samples     = samples
        self.tokenizer   = tokenizer
        self.muril_max   = muril_max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        src_ids = torch.tensor(s['src_ids'], dtype=torch.long)
        src_pos = torch.tensor(s['src_pos'], dtype=torch.float)
        tgt_ids = torch.tensor(s['tgt_ids'], dtype=torch.long)
        src_len = int((src_ids != 0).sum().item())
        src_len = max(src_len, 1)

        # MuRIL tokenization of the raw source text
        src_text = ' '.join(s['src_tokens'])
        enc = self.tokenizer(
            src_text,
            padding='max_length',
            truncation=True,
            max_length=self.muril_max,
            return_tensors='pt',
        )
        muril_ids  = enc['input_ids'].squeeze(0)       # (muril_max,)
        muril_mask = enc['attention_mask'].squeeze(0)  # (muril_max,)

        return src_ids, src_pos, tgt_ids, src_len, muril_ids, muril_mask


def get_muril_dataloaders(tokenizer, batch_size=32, muril_max_len=256):
    """
    Load preprocessed data and return MuRIL-aware DataLoaders.

    Args:
        tokenizer   : MuRIL/BERT tokenizer from transformers
        batch_size  : batch size (keep ≤32 for GPU memory with BERT)
        muril_max_len: max tokens for MuRIL (128 covers p95 of sources)
    """
    train_path = os.path.join(config.DATA_PROCESSED, 'train.pkl')
    test_path  = os.path.join(config.DATA_PROCESSED, 'test.pkl')

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # MuRIL max_len: source p95 = 74 tokens, Malayalam subword tokenization
    # adds ~3x subwords (agglutinative language), so 256 needed to cover ~95%.
    train_ds = MuRILDataset(train_data, tokenizer, muril_max_len)
    test_ds  = MuRILDataset(test_data,  tokenizer, muril_max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(config.DEVICE == 'cuda')
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(config.DEVICE == 'cuda')
    )
    print(f"MuRIL DataLoaders | train={len(train_ds)}, "
          f"test={len(test_ds)}, batch={batch_size}, "
          f"muril_max_len={muril_max_len}")
    return train_loader, test_loader