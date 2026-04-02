"""
src/train_muril.py  (v3 — correct MuRIL tokenization)

MuRIL is frozen (feature extractor). Raw source text is tokenized
with the MuRIL tokenizer — NOT by passing our vocab indices.
This is the correct way to use frozen BERT for feature extraction.

Usage:
    pip install transformers
    python src/train_muril.py --variant muril_bilstm_pos --batch_size 32
    python src/train_muril.py --variant word2vec_bilstm  --batch_size 64
    python src/train_muril.py --variant muril_bilstm     --batch_size 32
    python src/train_muril.py --variant muril_lstm       --batch_size 32
"""

import os, sys, argparse, pickle, time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.muril_model import build_muril_model, MURIL_VARIANTS
from src.preprocess import Vocabulary
from src.embedding import load_embedding_matrix
from src.dataset import get_dataloaders


def get_tokenizer(model_name='google/muril-base-cased'):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        print(f"MuRIL tokenizer loaded: {model_name}")
        return tok
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        return None


def train_epoch_standard(model, loader, optimizer, criterion, tf_ratio, device):
    """For word2vec_bilstm — no MuRIL tokenization needed."""
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
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / max(total_tokens, 1)


def eval_epoch_standard(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for src_ids, src_pos, tgt_ids, src_lengths in tqdm(loader, desc="  eval ", leave=False):
            src_ids     = src_ids.to(device)
            src_pos     = src_pos.to(device)
            tgt_ids     = tgt_ids.to(device)
            src_lengths = src_lengths.to(device)
            outputs = model(src_ids, src_pos, tgt_ids, src_lengths, 0.0)
            B, T1, V = outputs.shape
            loss = criterion(outputs.reshape(-1, V), tgt_ids[:, 1:].reshape(-1))
            n = (tgt_ids[:, 1:] != 0).sum().item()
            total_loss += loss.item() * n
            total_tokens += n
    return total_loss / max(total_tokens, 1)


def train_epoch_muril(model, loader, optimizer, criterion, tf_ratio, device):
    """For MuRIL variants — loader yields muril_ids and muril_mask too."""
    model.train()
    total_loss, total_tokens = 0, 0
    for src_ids, src_pos, tgt_ids, src_lengths, muril_ids, muril_mask in tqdm(
            loader, desc="  train", leave=False):
        src_ids     = src_ids.to(device)
        src_pos     = src_pos.to(device)
        tgt_ids     = tgt_ids.to(device)
        src_lengths = src_lengths.to(device)
        muril_ids   = muril_ids.to(device)
        muril_mask  = muril_mask.to(device)
        optimizer.zero_grad()
        outputs = model(src_ids, src_pos, tgt_ids, src_lengths, tf_ratio,
                        muril_input_ids=muril_ids, muril_attn_mask=muril_mask)
        B, T1, V = outputs.shape
        loss = criterion(outputs.reshape(-1, V), tgt_ids[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
        optimizer.step()
        n = (tgt_ids[:, 1:] != 0).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / max(total_tokens, 1)


def eval_epoch_muril(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for src_ids, src_pos, tgt_ids, src_lengths, muril_ids, muril_mask in tqdm(
                loader, desc="  eval ", leave=False):
            src_ids     = src_ids.to(device)
            src_pos     = src_pos.to(device)
            tgt_ids     = tgt_ids.to(device)
            src_lengths = src_lengths.to(device)
            muril_ids   = muril_ids.to(device)
            muril_mask  = muril_mask.to(device)
            outputs = model(src_ids, src_pos, tgt_ids, src_lengths, 0.0,
                            muril_input_ids=muril_ids, muril_attn_mask=muril_mask)
            B, T1, V = outputs.shape
            loss = criterion(outputs.reshape(-1, V), tgt_ids[:, 1:].reshape(-1))
            n = (tgt_ids[:, 1:] != 0).sum().item()
            total_loss += loss.item() * n
            total_tokens += n
    return total_loss / max(total_tokens, 1)


def train(variant='muril_bilstm_pos', batch_size=32):
    torch.manual_seed(config.SEED)
    device    = config.DEVICE
    use_muril = MURIL_VARIANTS[variant]['use_muril']

    print(f"\n{'='*60}")
    print(f"Training: {variant} | device={device}")
    print(f"  muril={use_muril} | bilstm={MURIL_VARIANTS[variant]['use_bilstm']}"
          f" | pos_gate={MURIL_VARIANTS[variant]['use_pos_gate']}")
    if use_muril:
        print(f"  MuRIL: FROZEN (feature extractor mode)")
    print(f"{'='*60}")

    vocab  = Vocabulary.load(os.path.join(config.DATA_PROCESSED, 'vocab.json'))
    matrix = None

    if use_muril:
        tokenizer = get_tokenizer()
        from src.muril_dataset import get_muril_dataloaders
        train_loader, test_loader = get_muril_dataloaders(
            tokenizer, batch_size=batch_size, muril_max_len=256
        )
    else:
        matrix = load_embedding_matrix(
            os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy'))
        train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = build_muril_model(
        variant=variant,
        vocab_size=len(vocab),
        embedding_matrix=matrix,
        freeze_muril=True,
    )

    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {n_train:,} | Frozen: {n_frozen:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0,
                                    label_smoothing=config.LABEL_SMOOTHING)
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
    history        = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, config.EPOCHS + 1):
        progress = (epoch - 1) / max(config.EPOCHS - 1, 1)
        tf_ratio = config.TF_START - (config.TF_START - config.TF_END) * progress

        t0 = time.time()
        if use_muril:
            train_loss = train_epoch_muril(model, train_loader, optimizer,
                                           criterion, tf_ratio, device)
            val_loss   = eval_epoch_muril(model, test_loader, criterion, device)
        else:
            train_loss = train_epoch_standard(model, train_loader, optimizer,
                                              criterion, tf_ratio, device)
            val_loss   = eval_epoch_standard(model, test_loader, criterion, device)

        elapsed = time.time() - t0
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        delta  = best_val_loss - val_loss
        marker = ''
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            marker = ' ✓'
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'variant': variant,
                        'vocab_size': len(vocab)}, best_ckpt)
        else:
            patience_count += 1

        if epoch % config.SAVE_EVERY == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_loss': val_loss},
                       os.path.join(config.CHECKPOINTS_DIR,
                                    f'{variant}_ep{epoch}.pt'))

        print(f"Ep {epoch:3d}/{config.EPOCHS} | "
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
    parser.add_argument('--variant', type=str, default='muril_bilstm_pos',
                        choices=list(MURIL_VARIANTS))
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    train(variant=args.variant, batch_size=args.batch_size)