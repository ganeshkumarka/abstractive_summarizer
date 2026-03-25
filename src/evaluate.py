r"""
src/evaluate.py  (v3 — fixed ROUGE for Malayalam Unicode)

Root cause of ROUGE=0: rouge_score library uses re.sub(r'[^\w\s]','',text)
internally. Python's \w without re.UNICODE flag strips Malayalam characters,
leaving empty strings → ROUGE=0 for all samples.

Fix: implement ROUGE-1, ROUGE-2, ROUGE-L directly using whitespace tokenization
which preserves Malayalam Unicode. This is correct for Malayalam since words
are already space-separated after our preprocessing.

Usage:
    python src/evaluate.py --variant ptf_attention
    python src/evaluate.py --all
r"""

import os, sys, argparse, pickle
import torch
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import get_dataloaders
from src.model import build_model, Seq2SeqModel
from src.preprocess import Vocabulary
from src.embedding import load_embedding_matrix


# ── Unicode-safe ROUGE implementation ─────────────────────────────────────────

def tokenize_malayalam(text: str) -> list:
    """
    Simple whitespace tokenizer that preserves Malayalam Unicode.
    Do NOT use regex char classes — they strip Malayalam Unicode by default.
    """
    return [t for t in text.strip().split() if t]


def get_ngrams(tokens: list, n: int) -> Counter:
    """Return Counter of n-grams from a token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def rouge_n(ref_tokens: list, hyp_tokens: list, n: int) -> dict:
    """Compute ROUGE-N precision, recall, F1."""
    ref_ngrams = get_ngrams(ref_tokens, n)
    hyp_ngrams = get_ngrams(hyp_tokens, n)

    ref_count = sum(ref_ngrams.values())
    hyp_count = sum(hyp_ngrams.values())

    if ref_count == 0 or hyp_count == 0:
        return {'p': 0.0, 'r': 0.0, 'f': 0.0}

    overlap = sum((ref_ngrams & hyp_ngrams).values())
    precision = overlap / hyp_count
    recall    = overlap / ref_count
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {'p': precision, 'r': recall, 'f': f1}


def lcs_length(x: list, y: list) -> int:
    """Compute length of longest common subsequence (dynamic programming)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimised: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(ref_tokens: list, hyp_tokens: list) -> dict:
    """Compute ROUGE-L precision, recall, F1 using LCS."""
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)

    if ref_len == 0 or hyp_len == 0:
        return {'p': 0.0, 'r': 0.0, 'f': 0.0}

    lcs = lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / hyp_len
    recall    = lcs / ref_len
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {'p': precision, 'r': recall, 'f': f1}


def compute_rouge(ref_text: str, hyp_text: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L for a single ref/hyp pair.
    Returns F1 scores × 100 (to match paper reporting style).
    """
    ref_tok = tokenize_malayalam(ref_text)
    hyp_tok = tokenize_malayalam(hyp_text)

    r1 = rouge_n(ref_tok, hyp_tok, 1)
    r2 = rouge_n(ref_tok, hyp_tok, 2)
    rl = rouge_l(ref_tok, hyp_tok)

    return {
        'rouge1': r1['f'] * 100,
        'rouge2': r2['f'] * 100,
        'rougeL': rl['f'] * 100,
    }


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_from_checkpoint(variant: str, vocab_size: int,
                                embedding_matrix) -> Seq2SeqModel:
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, f'best_{variant}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")
    model = build_model(vocab_size, embedding_matrix, variant=variant)
    ckpt  = torch.load(ckpt_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path} "
          f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model


def ids_to_text(ids: list, vocab: Vocabulary) -> str:
    """Convert token ids → Malayalam string, skipping special tokens."""
    tokens = vocab.decode(ids, skip_special=True)
    tokens = [t for t in tokens if t != config.UNK_TOKEN]
    return ' '.join(tokens)


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate_model(model: Seq2SeqModel, test_loader, vocab: Vocabulary,
                   device: str = config.DEVICE) -> dict:
    """
    Greedy decode test set and compute ROUGE scores.
    Uses our Unicode-safe scorer (not rouge_score library).
    """
    start_idx = vocab.word2idx[config.START_TOKEN]
    end_idx   = vocab.word2idx[config.END_TOKEN]

    all_scores  = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    gen_lengths = []
    examples    = []

    model.eval()
    with torch.no_grad():
        for src_ids, src_pos, tgt_ids, src_lengths in tqdm(
                test_loader, desc="  evaluating"):
            src_ids     = src_ids.to(device)
            src_pos     = src_pos.to(device)
            src_lengths = src_lengths.to(device)

            gen_ids = model.generate(
                src_ids, src_pos, src_lengths,
                max_len=config.MAX_SUMMARY_LEN,
                start_idx=start_idx,
                end_idx=end_idx,
            )

            for i in range(src_ids.size(0)):
                ref_text = ids_to_text(tgt_ids[i].tolist(), vocab)
                hyp_text = ids_to_text(gen_ids[i].tolist(), vocab)

                gen_lengths.append(len(hyp_text.split()))

                if len(examples) < 3:
                    examples.append({'ref': ref_text, 'hyp': hyp_text})

                if not ref_text.strip() or not hyp_text.strip():
                    continue

                scores = compute_rouge(ref_text, hyp_text)
                all_scores['rouge1'].append(scores['rouge1'])
                all_scores['rouge2'].append(scores['rouge2'])
                all_scores['rougeL'].append(scores['rougeL'])

    avg_len     = sum(gen_lengths) / max(len(gen_lengths), 1)
    unique_out  = len(set(e['hyp'] for e in examples))
    print(f"  Debug | avg gen length: {avg_len:.1f} | "
          f"unique in sample: {unique_out}/{len(examples)}")
    print("  Sample outputs:")
    for ex in examples:
        ref_tok_count = len(tokenize_malayalam(ex['ref']))
        hyp_tok_count = len(tokenize_malayalam(ex['hyp']))
        pair_score = compute_rouge(ex['ref'], ex['hyp'])
        print(f"    REF ({ref_tok_count} tok): {ex['ref'][:70]}")
        print(f"    HYP ({hyp_tok_count} tok): {ex['hyp'][:70]}")
        print(f"    R1={pair_score['rouge1']:.1f} "
              f"R2={pair_score['rouge2']:.1f} "
              f"RL={pair_score['rougeL']:.1f}")
        print()

    results = {k: round(sum(v) / max(len(v), 1), 2)
               for k, v in all_scores.items()}
    return results


def print_results_table(results_by_variant: dict):
    print("\n" + "="*55)
    print(f"{'Metric':<12} {'Seq2Seq':>12} {'Attention':>12} {'PTF+Attn':>12}")
    print("-"*55)
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        row = f"{metric.upper():<12}"
        for variant in ['seq2seq', 'attention', 'ptf_attention']:
            val = results_by_variant.get(variant, {}).get(metric, '-')
            row += f"{str(val):>12}"
        print(row)
    print("="*55)


def run_evaluation(variant=None, batch_size=config.BATCH_SIZE):
    vocab  = Vocabulary.load(os.path.join(config.DATA_PROCESSED, 'vocab.json'))
    matrix = load_embedding_matrix(
        os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy')
    )
    _, test_loader = get_dataloaders(batch_size=batch_size)

    variants_to_eval = (
        [variant] if variant
        else ['seq2seq', 'attention', 'ptf_attention']
    )

    results_by_variant = {}
    for v in variants_to_eval:
        print(f"\nEvaluating: {v}")
        try:
            model  = load_model_from_checkpoint(v, len(vocab), matrix)
            scores = evaluate_model(model, test_loader, vocab)
            results_by_variant[v] = scores
            print(f"  ROUGE-1={scores['rouge1']:.2f} | "
                  f"ROUGE-2={scores['rouge2']:.2f} | "
                  f"ROUGE-L={scores['rougeL']:.2f}")
        except FileNotFoundError as e:
            print(f"  Skipping {v}: {e}")

    if len(results_by_variant) > 1:
        print_results_table(results_by_variant)

    results_path = os.path.join(config.CHECKPOINTS_DIR,
                                'evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_by_variant, f)
    print(f"\nResults saved → {results_path}")
    return results_by_variant


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default=None,
                        choices=['seq2seq', 'attention', 'ptf_attention'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()
    run_evaluation(
        variant=None if args.all else args.variant,
        batch_size=args.batch_size,
    )