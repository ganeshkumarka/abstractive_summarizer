"""
src/train.py
------------
Training loop for the three model variants.
Implements:
  - Teacher forcing (ratio decays over epochs)
  - Gradient clipping (paper uses clip=5.0)
  - Checkpoint saving (best model by val loss)
  - Loss logging per epoch

Usage:
    python src/train.py --variant ptf_attention
    python src/train.py --variant attention
    python src/train.py --variant seq2seq
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import get_dataloaders
from src.model import build_model
from src.preprocess import Vocabulary
from src.embedding import load_embedding_matrix


def train_epoch(model, loader, optimizer, criterion, teacher_forcing_ratio, device):
    model.train()
    total_loss = 0
    total_tokens = 0

    for src_ids, src_pos, tgt_ids, src_lengths in tqdm(loader, desc="  train", leave=False):
        src_ids    = src_ids.to(device)
        src_pos    = src_pos.to(device)
        tgt_ids    = tgt_ids.to(device)
        src_lengths = src_lengths.to(device)

        optimizer.zero_grad()

        # Forward pass — outputs: (batch, tgt_len-1, vocab_size)
        outputs = model(src_ids, src_pos, tgt_ids, src_lengths, teacher_forcing_ratio)

        # Reshape for loss: (batch * (tgt_len-1), vocab_size)
        batch_size, tgt_len_minus1, vocab_size = outputs.shape
        outputs_flat = outputs.reshape(-1, vocab_size)
        targets_flat = tgt_ids[:, 1:].reshape(-1)         # shift left by 1

        loss = criterion(outputs_flat, targets_flat)
        loss.backward()

        # Gradient clipping (paper §4.2)
        nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
        optimizer.step()

        # Count non-PAD tokens for accurate loss reporting
        n_tokens = (targets_flat != 0).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for src_ids, src_pos, tgt_ids, src_lengths in tqdm(loader, desc="  eval ", leave=False):
            src_ids    = src_ids.to(device)
            src_pos    = src_pos.to(device)
            tgt_ids    = tgt_ids.to(device)
            src_lengths = src_lengths.to(device)

            # At eval time, no teacher forcing
            outputs = model(src_ids, src_pos, tgt_ids, src_lengths,
                            teacher_forcing_ratio=0.0)

            batch_size, tgt_len_minus1, vocab_size = outputs.shape
            outputs_flat = outputs.reshape(-1, vocab_size)
            targets_flat = tgt_ids[:, 1:].reshape(-1)

            loss = criterion(outputs_flat, targets_flat)
            n_tokens = (targets_flat != 0).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def train(variant: str = 'ptf_attention', batch_size: int = config.BATCH_SIZE):
    torch.manual_seed(config.SEED)
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"Training variant: {variant} | batch_size={batch_size} | device={device}")
    print(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # ── Load vocab + embedding matrix ──────────────────────────────
    vocab = Vocabulary.load(os.path.join(config.DATA_PROCESSED, 'vocab.json'))
    matrix_path = os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy')
    embedding_matrix = load_embedding_matrix(matrix_path)

    # ── Build model ────────────────────────────────────────────────
    model = build_model(len(vocab), embedding_matrix, variant=variant)

    # ── Loss: ignore PAD token (index 0) ──────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # ── Optimiser ──────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # ── Checkpoint dir ─────────────────────────────────────────────
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    best_ckpt = os.path.join(config.CHECKPOINTS_DIR, f'best_{variant}.pt')
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    # ── Training loop ──────────────────────────────────────────────
    for epoch in range(1, config.EPOCHS + 1):
        # Decay teacher forcing ratio over epochs (start 0.8 → end ~0.0)
        teacher_forcing_ratio = max(0.0, 0.8 - epoch * (0.8 / config.EPOCHS))

        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 teacher_forcing_ratio, device)
        val_loss   = eval_epoch(model, test_loader, criterion, device)
        elapsed    = time.time() - t0

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch:3d}/{config.EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"tf_ratio={teacher_forcing_ratio:.2f} | {elapsed:.1f}s")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':   val_loss,
                'variant':    variant,
                'vocab_size': len(vocab),
            }, best_ckpt)
            print(f"  ✓ Saved best checkpoint → {best_ckpt}")

    # Save training history
    history_path = os.path.join(config.CHECKPOINTS_DIR, f'history_{variant}.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='ptf_attention',
                        choices=['seq2seq', 'attention', 'ptf_attention'],
                        help='Which model variant to train')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    train(variant=args.variant, batch_size=args.batch_size)