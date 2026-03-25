"""
src/train.py  (v5)
Changes:
  - Encoder freezing for first ENCODER_FREEZE_EPOCHS epochs
  - Prints frozen/unfrozen status clearly
  - Syntax warning fix (raw strings in comments)
"""

import os, sys, argparse, pickle, time
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


def set_encoder_grad(model, requires_grad: bool):
    """Freeze or unfreeze encoder parameters."""
    for param in model.encoder.parameters():
        param.requires_grad = requires_grad
    for param in model.ptf_embed.parameters():
        param.requires_grad = requires_grad


def train_epoch(model, loader, optimizer, criterion, tf_ratio, device):
    model.train()
    total_loss, total_tokens = 0, 0
    for src_ids, src_pos, tgt_ids, src_lengths in tqdm(loader, desc="  train", leave=False):
        src_ids     = src_ids.to(device)
        src_pos     = src_pos.to(device)
        tgt_ids     = tgt_ids.to(device)
        src_lengths = src_lengths.to(device)

        optimizer.zero_grad()
        outputs = model(src_ids, src_pos, tgt_ids, src_lengths, tf_ratio)
        B, T1, V = outputs.shape
        loss = criterion(outputs.reshape(-1, V), tgt_ids[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
        optimizer.step()

        n = (tgt_ids[:, 1:] != 0).sum().item()
        total_loss   += loss.item() * n
        total_tokens += n
    return total_loss / max(total_tokens, 1)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for src_ids, src_pos, tgt_ids, src_lengths in tqdm(loader, desc="  eval ", leave=False):
            src_ids     = src_ids.to(device)
            src_pos     = src_pos.to(device)
            tgt_ids     = tgt_ids.to(device)
            src_lengths = src_lengths.to(device)

            outputs = model(src_ids, src_pos, tgt_ids, src_lengths,
                            teacher_forcing_ratio=0.0)
            B, T1, V = outputs.shape
            loss = criterion(outputs.reshape(-1, V), tgt_ids[:, 1:].reshape(-1))
            n = (tgt_ids[:, 1:] != 0).sum().item()
            total_loss   += loss.item() * n
            total_tokens += n
    return total_loss / max(total_tokens, 1)


def train(variant='ptf_attention', batch_size=config.BATCH_SIZE):
    torch.manual_seed(config.SEED)
    device = config.DEVICE

    print(f"\n{'='*60}")
    print(f"Training: {variant} | batch={batch_size} | device={device}")
    print(f"  HIDDEN={config.HIDDEN_DIM} | LR={config.LEARNING_RATE} "
          f"| TF {config.TF_START}->{config.TF_END} "
          f"| freeze_enc={config.ENCODER_FREEZE_EPOCHS}ep")
    print(f"  Target column: {config.SUMMARY_COL}")
    print(f"{'='*60}")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    vocab  = Vocabulary.load(os.path.join(config.DATA_PROCESSED, 'vocab.json'))
    matrix = load_embedding_matrix(
        os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy'))
    print(f"  Vocab: {len(vocab)} | Summary col: {config.SUMMARY_COL}")

    model = build_model(len(vocab), matrix, variant=variant)
    criterion = nn.CrossEntropyLoss(ignore_index=0,
                                    label_smoothing=config.LABEL_SMOOTHING)

    # Start with encoder frozen — only train decoder
    set_encoder_grad(model, requires_grad=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Phase 1 (ep 1-{config.ENCODER_FREEZE_EPOCHS}): "
          f"encoder FROZEN | trainable params={trainable:,}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5
    )

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    best_ckpt      = os.path.join(config.CHECKPOINTS_DIR, f'best_{variant}.pt')
    best_val_loss  = float('inf')
    patience_count = 0
    history        = {'train_loss': [], 'val_loss': [], 'lr': []}
    encoder_thawed = False

    for epoch in range(1, config.EPOCHS + 1):

        # Unfreeze encoder after ENCODER_FREEZE_EPOCHS
        if epoch == config.ENCODER_FREEZE_EPOCHS + 1 and not encoder_thawed:
            set_encoder_grad(model, requires_grad=True)
            encoder_thawed = True
            # Rebuild optimizer to include newly unfrozen params
            optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE * 0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5
            )
            total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n  Phase 2 (ep {epoch}+): encoder UNFROZEN | "
                  f"all params={total_p:,} | LR halved to {config.LEARNING_RATE*0.5:.2e}")

        progress = (epoch - 1) / max(config.EPOCHS - 1, 1)
        tf_ratio = config.TF_START - (config.TF_START - config.TF_END) * progress

        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer,
                                 criterion, tf_ratio, device)
        val_loss   = eval_epoch(model, test_loader, criterion, device)
        elapsed    = time.time() - t0

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(lr_now)

        delta  = best_val_loss - val_loss
        marker = ''
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            marker = ' ✓'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'variant': variant,
                'vocab_size': len(vocab),
            }, best_ckpt)
        else:
            patience_count += 1

        if epoch % config.SAVE_EVERY == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_loss': val_loss},
                       os.path.join(config.CHECKPOINTS_DIR,
                                    f'{variant}_ep{epoch}.pt'))

        phase = 'frozen' if epoch <= config.ENCODER_FREEZE_EPOCHS else 'full'
        print(f"Ep {epoch:3d}/{config.EPOCHS} [{phase}] | "
              f"train={train_loss:.4f} val={val_loss:.4f} "
              f"d={delta:+.4f} tf={tf_ratio:.2f} "
              f"lr={lr_now:.2e} | {elapsed:.0f}s{marker}")

        if patience_count >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    with open(os.path.join(config.CHECKPOINTS_DIR,
                           f'history_{variant}.pkl'), 'wb') as f:
        pickle.dump(history, f)

    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='ptf_attention',
                        choices=['seq2seq', 'attention', 'ptf_attention'])
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()
    train(variant=args.variant, batch_size=args.batch_size)