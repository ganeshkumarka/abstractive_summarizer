"""
src/evaluate_muril.py
---------------------
Evaluate all trained models (paper variants + proposed MuRIL variants)
and print a unified comparison table for the project report.

Usage:
    python src/evaluate_muril.py
    python src/evaluate_muril.py --variant muril_bilstm_pos
"""

import os, sys, argparse, pickle, torch
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocess import Vocabulary
from src.embedding import load_embedding_matrix
from src.model import build_model
from src.muril_model import build_muril_model, MURIL_VARIANTS
from src.dataset import get_dataloaders
from src.evaluate import compute_rouge, ids_to_text


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_paper_model(variant, vocab_size, matrix):
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, f'best_{variant}.pt')
    if not os.path.exists(ckpt_path):
        return None, None
    try:
        model = build_model(vocab_size, matrix, variant=variant)
        model = model.cpu()
        ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
    except (OSError, RuntimeError) as e:
        print(f"  Checkpoint error: {e}")
        return None, None


def load_muril_model(variant, vocab_size, matrix):
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, f'best_{variant}.pt')
    if not os.path.exists(ckpt_path):
        return None, None
    try:
        # Load to CPU first to avoid OOM when VRAM is fragmented
        model = build_muril_model(variant, vocab_size=vocab_size,
                                   embedding_matrix=matrix, freeze_muril=True)
        model = model.cpu()
        ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
    except (OSError, RuntimeError) as e:
        print(f"  Checkpoint error: {e}")
        return None, None


# ── Evaluation loops ───────────────────────────────────────────────────────────

def eval_paper_variant(model, test_loader, vocab, device):
    start_idx = vocab.word2idx[config.START_TOKEN]
    end_idx   = vocab.word2idx[config.END_TOKEN]
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    examples = []

    model.eval()
    with torch.no_grad():
        for src_ids, src_pos, tgt_ids, src_lengths in tqdm(
                test_loader, desc="  scoring", leave=False):
            src_ids     = src_ids.to(device)
            src_pos     = src_pos.to(device)
            src_lengths = src_lengths.to(device)

            gen_ids = model.generate(
                src_ids, src_pos, src_lengths,
                max_len=config.MAX_SUMMARY_LEN,
                start_idx=start_idx, end_idx=end_idx,
            )
            for i in range(src_ids.size(0)):
                ref = ids_to_text(tgt_ids[i].tolist(), vocab)
                hyp = ids_to_text(gen_ids[i].tolist(), vocab)
                if len(examples) < 2:
                    examples.append({'ref': ref, 'hyp': hyp})
                if ref.strip() and hyp.strip():
                    s = compute_rouge(ref, hyp)
                    scores['rouge1'].append(s['rouge1'])
                    scores['rouge2'].append(s['rouge2'])
                    scores['rougeL'].append(s['rougeL'])

    return {k: round(sum(v)/max(len(v),1), 2) for k, v in scores.items()}, examples


def eval_muril_variant(model, test_loader, vocab, device, use_muril):
    """Evaluate MuRIL variant — handles both w2v+bilstm and MuRIL loaders."""
    start_idx = vocab.word2idx[config.START_TOKEN]
    end_idx   = vocab.word2idx[config.END_TOKEN]
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    examples = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  scoring", leave=False):
            if use_muril:
                src_ids, src_pos, tgt_ids, src_lengths, muril_ids, muril_mask = batch
                muril_ids  = muril_ids.to(device)
                muril_mask = muril_mask.to(device)
            else:
                src_ids, src_pos, tgt_ids, src_lengths = batch
                muril_ids = muril_mask = None

            src_ids     = src_ids.to(device)
            src_pos     = src_pos.to(device)
            src_lengths = src_lengths.to(device)

            gen_ids = model.generate(
                src_ids, src_pos, src_lengths,
                max_len=config.MAX_SUMMARY_LEN,
                start_idx=start_idx, end_idx=end_idx,
                muril_input_ids=muril_ids,
                muril_attn_mask=muril_mask,
            )
            for i in range(src_ids.size(0)):
                ref = ids_to_text(tgt_ids[i].tolist(), vocab)
                hyp = ids_to_text(gen_ids[i].tolist(), vocab)
                if len(examples) < 2:
                    examples.append({'ref': ref, 'hyp': hyp})
                if ref.strip() and hyp.strip():
                    s = compute_rouge(ref, hyp)
                    scores['rouge1'].append(s['rouge1'])
                    scores['rouge2'].append(s['rouge2'])
                    scores['rougeL'].append(s['rougeL'])

    return {k: round(sum(v)/max(len(v),1), 2) for k, v in scores.items()}, examples


# ── Main ───────────────────────────────────────────────────────────────────────

def run_all_evaluation():
    device = config.DEVICE
    vocab  = Vocabulary.load(os.path.join(config.DATA_PROCESSED, 'vocab.json'))
    matrix = load_embedding_matrix(
        os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy'))

    _, std_test_loader = get_dataloaders(batch_size=64)

    # MuRIL test loader (only created if needed)
    muril_test_loader = None

    all_results = {}

    # ── Paper variants ─────────────────────────────────────────────────────────
    paper_variants = ['seq2seq', 'attention', 'ptf_attention']
    print("\n=== Paper baseline variants ===")
    for v in paper_variants:
        model, ckpt = load_paper_model(v, len(vocab), matrix)
        if model is None:
            print(f"  {v}: no checkpoint found")
            continue
        print(f"\n  {v} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
        model = model.to(device)
        scores, examples = eval_paper_variant(model, std_test_loader, vocab, device)
        all_results[v] = scores
        print(f"  ROUGE-1={scores['rouge1']:.2f} | "
              f"ROUGE-2={scores['rouge2']:.2f} | "
              f"ROUGE-L={scores['rougeL']:.2f}")
        for ex in examples[:1]:
            print(f"    REF: {ex['ref'][:60]}")
            print(f"    HYP: {ex['hyp'][:60]}")
        del model
        if device == 'cuda':
            import gc; gc.collect()
            torch.cuda.empty_cache()

    # ── Proposed MuRIL variants ────────────────────────────────────────────────
    muril_order = ['word2vec_bilstm', 'muril_lstm',
                   'muril_bilstm', 'muril_bilstm_pos']

    # Pre-load MuRIL tokenizer once (avoid repeated HF downloads per variant)
    muril_tokenizer = None
    try:
        from transformers import AutoTokenizer
        muril_tokenizer = AutoTokenizer.from_pretrained('google/muril-base-cased')
        print("MuRIL tokenizer loaded")
    except Exception as e:
        print(f"Could not load MuRIL tokenizer: {e}")

    print("\n=== Proposed MuRIL variants ===")
    for v in muril_order:
        model, ckpt = load_muril_model(v, len(vocab), matrix)
        if model is None:
            print(f"  {v}: no checkpoint found")
            continue

        use_muril = MURIL_VARIANTS[v]['use_muril']
        print(f"\n  {v} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
        model = model.to(device)

        if use_muril:
            if muril_tokenizer is None:
                print(f"  Skipping {v}: MuRIL tokenizer not available")
                continue
            try:
                from src.muril_dataset import get_muril_dataloaders
                _, loader = get_muril_dataloaders(muril_tokenizer, batch_size=32, muril_max_len=128)
            except Exception as e:
                print(f"  Could not load MuRIL loader: {e}")
                continue
        else:
            loader = std_test_loader

        scores, examples = eval_muril_variant(model, loader, vocab, device, use_muril)
        all_results[v] = scores
        print(f"  ROUGE-1={scores['rouge1']:.2f} | "
              f"ROUGE-2={scores['rouge2']:.2f} | "
              f"ROUGE-L={scores['rougeL']:.2f}")
        for ex in examples[:1]:
            print(f"    REF: {ex['ref'][:60]}")
            print(f"    HYP: {ex['hyp'][:60]}")
        # Free GPU memory before loading next model
        del model
        if device == 'cuda':
            import gc; gc.collect()
            torch.cuda.empty_cache()

    # ── Full comparison table ──────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"{'Model':<28} {'Val Loss':>10} {'ROUGE-1':>9} {'ROUGE-2':>9} {'ROUGE-L':>9}")
    print("-"*72)

    rows = [
        ('seq2seq',          'Seq2Seq (baseline)'),
        ('attention',        'Attention'),
        ('ptf_attention',    'PTF+Attention (paper)'),
        ('word2vec_bilstm',  'Word2Vec+BiLSTM'),
        ('muril_lstm',       'MuRIL+LSTM'),
        ('muril_bilstm',     'MuRIL+BiLSTM'),
        ('muril_bilstm_pos', 'MuRIL+BiLSTM+POS-gate'),
    ]
    for key, label in rows:
        if key not in all_results:
            print(f"  {label:<26} {'—':>10} {'—':>9} {'—':>9} {'—':>9}")
            continue
        s = all_results[key]
        # Get val loss from checkpoint
        ckpt_path = os.path.join(config.CHECKPOINTS_DIR, f'best_{key}.pt')
        val_loss = '—'
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location='cpu')
            val_loss = f"{ck['val_loss']:.4f}"
        marker = ' ←' if key == 'muril_bilstm_pos' else ''
        print(f"  {label:<26} {val_loss:>10} "
              f"{s['rouge1']:>9.2f} {s['rouge2']:>9.2f} {s['rougeL']:>9.2f}{marker}")
    print("="*72)

    # Save
    results_path = os.path.join(config.CHECKPOINTS_DIR, 'full_evaluation.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nAll results saved → {results_path}")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default=None)
    args = parser.parse_args()
    run_all_evaluation()