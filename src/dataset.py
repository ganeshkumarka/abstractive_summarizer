"""
src/dataset.py
--------------
PyTorch Dataset and DataLoader wrappers for the preprocessed Malayalam data.
Each sample yields:
    src_ids  : (MAX_INPUT_LEN,)         — source token indices
    src_pos  : (MAX_INPUT_LEN, POS_DIM) — POS one-hot vectors
    tgt_ids  : (MAX_SUMMARY_LEN,)       — target token indices (with START/END)
    src_len  : int                       — actual (non-padded) source length
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MalayalamSumDataset(Dataset):
    """Dataset for Malayalam abstractive summarization."""

    def __init__(self, samples: list):
        """
        Args:
            samples: list of dicts from preprocess.py, each containing:
                     src_ids, src_pos, tgt_ids
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        src_ids = torch.tensor(sample['src_ids'], dtype=torch.long)
        src_pos = torch.tensor(sample['src_pos'], dtype=torch.float)
        tgt_ids = torch.tensor(sample['tgt_ids'], dtype=torch.long)

        # Actual (non-padded) source length — used to pack sequences
        src_len = int((src_ids != 0).sum().item())
        src_len = max(src_len, 1)  # at least 1

        return src_ids, src_pos, tgt_ids, src_len


def get_dataloaders(batch_size: int = config.BATCH_SIZE):
    """
    Load preprocessed pickle files and return train/test DataLoaders.
    
    Returns:
        train_loader, test_loader
    """
    train_path = os.path.join(config.DATA_PROCESSED, 'train.pkl')
    test_path  = os.path.join(config.DATA_PROCESSED, 'test.pkl')

    if not os.path.exists(train_path):
        raise FileNotFoundError("Run src/preprocess.py first to generate processed data.")

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    train_dataset = MalayalamSumDataset(train_data)
    test_dataset  = MalayalamSumDataset(test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,    # set to 2-4 if using GPU
        pin_memory=(config.DEVICE == 'cuda'),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(config.DEVICE == 'cuda'),
    )

    print(f"DataLoaders ready | train={len(train_dataset)}, "
          f"test={len(test_dataset)}, batch_size={batch_size}")
    return train_loader, test_loader