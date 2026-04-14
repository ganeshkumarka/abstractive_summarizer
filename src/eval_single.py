"""
src/eval_single.py
------------------
Evaluate a single model variant in isolation.
Use this when evaluate_muril.py runs out of GPU memory.

Usage:
    python src/eval_single.py --variant muril_bilstm
    python src/eval_single.py --variant muril_bilstm_pos
    python src/eval_single.py --variant word2vec_bilstm
    python src/eval_single.py --variant attention
"""

import os, sys, argparse, gc
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocess import Vocabulary
from src.embedding import load_embedding_matrix
from src.evaluate import compute_rouge, ids_to_text
from tqdm import tqdm


def run(variant):
    device = config.DEVICE
    vocab  = Vocabulary.load(os.path.join(config.DATA_PROCESSED, 'vocab.json'))
    matrix = load_embedding_matrix(
        os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy'))

    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, f'best_{variant}.pt')
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint: {ckpt_path}")
        return

    # Determine if paper or MuRIL variant
    paper_variants = ['seq2seq', 'attention', 'ptf_attention']
    muril_variants = ['word2vec_bilstm', 'muril_lstm', 'muril_bilstm', 'muril_bilstm_pos']

    print(f"\nEvaluating: {variant} on {device}")

    if variant in paper_variants:
        from src.model import build_model
        from src.dataset import get_dataloaders
        model = build_model(len(vocab), matrix, variant=variant).cpu()
        ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device); model.eval()
        print(f"  Loaded epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

        _, test_loader = get_dataloaders(batch_size=64)
        start_idx = vocab.word2idx[config.START_TOKEN]
        end_idx   = vocab.word2idx[config.END_TOKEN]

        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        examples = []
        with torch.no_grad():
            for src_ids, src_pos, tgt_ids, src_lengths in tqdm(test_loader, desc="  eval"):
                src_ids = src_ids.to(device); src_pos = src_pos.to(device)
                src_lengths = src_lengths.to(device)
                gen = model.generate(src_ids, src_pos, src_lengths,
                                     start_idx=start_idx, end_idx=end_idx)
                for i in range(src_ids.size(0)):
                    ref = ids_to_text(tgt_ids[i].tolist(), vocab)
                    hyp = ids_to_text(gen[i].tolist(), vocab)
                    if len(examples) < 3:
                        examples.append({'ref': ref, 'hyp': hyp})
                    if ref.strip() and hyp.strip():
                        s = compute_rouge(ref, hyp)
                        scores['rouge1'].append(s['rouge1'])
                        scores['rouge2'].append(s['rouge2'])
                        scores['rougeL'].append(s['rougeL'])

    elif variant in muril_variants:
        from src.muril_model import build_muril_model, MURIL_VARIANTS
        use_muril = MURIL_VARIANTS[variant]['use_muril']

        m = None if use_muril else matrix
        model = build_muril_model(variant, vocab_size=len(vocab),
                                   embedding_matrix=m, freeze_muril=True).cpu()
        ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device); model.eval()
        print(f"  Loaded epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

        if use_muril:
            from transformers import AutoTokenizer
            from src.muril_dataset import get_muril_dataloaders
            tok = AutoTokenizer.from_pretrained('google/muril-base-cased')
            _, test_loader = get_muril_dataloaders(tok, batch_size=16, muril_max_len=128)
        else:
            from src.dataset import get_dataloaders
            _, test_loader = get_dataloaders(batch_size=64)

        start_idx = vocab.word2idx[config.START_TOKEN]
        end_idx   = vocab.word2idx[config.END_TOKEN]

        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        examples = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="  eval"):
                if use_muril:
                    src_ids, src_pos, tgt_ids, src_lengths, muril_ids, muril_mask = batch
                    muril_ids  = muril_ids.to(device)
                    muril_mask = muril_mask.to(device)
                else:
                    src_ids, src_pos, tgt_ids, src_lengths = batch
                    muril_ids = muril_mask = None

                src_ids = src_ids.to(device); src_pos = src_pos.to(device)
                src_lengths = src_lengths.to(device)

                gen = model.generate(src_ids, src_pos, src_lengths,
                                     start_idx=start_idx, end_idx=end_idx,
                                     muril_input_ids=muril_ids,
                                     muril_attn_mask=muril_mask)
                for i in range(src_ids.size(0)):
                    ref = ids_to_text(tgt_ids[i].tolist(), vocab)
                    hyp = ids_to_text(gen[i].tolist(), vocab)
                    if len(examples) < 3:
                        examples.append({'ref': ref, 'hyp': hyp})
                    if ref.strip() and hyp.strip():
                        s = compute_rouge(ref, hyp)
                        scores['rouge1'].append(s['rouge1'])
                        scores['rouge2'].append(s['rouge2'])
                        scores['rougeL'].append(s['rougeL'])
    else:
        print(f"Unknown variant: {variant}")
        return

    r1 = round(sum(scores['rouge1'])/max(len(scores['rouge1']),1), 2)
    r2 = round(sum(scores['rouge2'])/max(len(scores['rouge2']),1), 2)
    rl = round(sum(scores['rougeL'])/max(len(scores['rougeL']),1), 2)

    print(f"\n  ROUGE-1={r1} | ROUGE-2={r2} | ROUGE-L={rl}")
    print("\n  Sample outputs:")
    for ex in examples[:2]:
        s = compute_rouge(ex['ref'], ex['hyp'])
        print(f"    REF: {ex['ref'][:60]}")
        print(f"    HYP: {ex['hyp'][:60]}")
        print(f"    R1={s['rouge1']:.1f} R2={s['rouge2']:.1f} RL={s['rougeL']:.1f}")

    # Clear GPU
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return r1, r2, rl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    args = parser.parse_args()
    run(args.variant)