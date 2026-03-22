"""
src/evaluate.py
---------------
Evaluation using ROUGE-1, ROUGE-2, ROUGE-L — exactly as reported in
the paper's Tables 3, 4, 5.

Reproduces the comparative table for all three variants at the given batch size.

Usage:
    # Evaluate a specific variant:
    python src/evaluate.py --variant ptf_attention

    # Reproduce full comparison table (all 3 variants):
    python src/evaluate.py --all
"""

import os
import sys
import argparse
import pickle
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import get_dataloaders
from src.model import build_model, Seq2SeqModel
from src.preprocess import Vocabulary
from src.embedding import load_embedding_matrix


def load_model_from_checkpoint(variant: str, vocab_size: int,
                                embedding_matrix) -> Seq2SeqModel:
    """Load best saved checkpoint for a variant."""
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, f'best_{variant}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Train the model first."
        )
    model = build_model(vocab_size, embedding_matrix, variant=variant)
    ckpt  = torch.load(ckpt_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f})")
    return model


def ids_to_text(ids: list, vocab: Vocabulary) -> str:
    """Convert predicted token ids → Malayalam string."""
    tokens = vocab.decode(ids, skip_special=True)
    return ' '.join(tokens)


def evaluate_model(model: Seq2SeqModel, test_loader, vocab: Vocabulary,
                   device: str = config.DEVICE) -> dict:
    """
    Run greedy decoding on test set and compute ROUGE scores.

    Returns:
        dict with keys 'rouge1', 'rouge2', 'rougeL' (F-measure ×100)
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    start_idx = vocab.word2idx[config.START_TOKEN]
    end_idx   = vocab.word2idx[config.END_TOKEN]

    all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    model.eval()
    with torch.no_grad():
        for src_ids, src_pos, tgt_ids, src_lengths in tqdm(test_loader, desc="  evaluating"):
            src_ids     = src_ids.to(device)
            src_pos     = src_pos.to(device)
            src_lengths = src_lengths.to(device)

            # Generate summaries
            gen_ids = model.generate(
                src_ids, src_pos, src_lengths,
                max_len=config.MAX_SUMMARY_LEN,
                start_idx=start_idx,
                end_idx=end_idx,
            )

            # Score each sample in the batch
            for i in range(src_ids.size(0)):
                ref_text  = ids_to_text(tgt_ids[i].tolist(), vocab)
                hyp_text  = ids_to_text(gen_ids[i].tolist(), vocab)

                if not ref_text.strip() or not hyp_text.strip():
                    continue

                scores = scorer.score(ref_text, hyp_text)
                all_scores['rouge1'].append(scores['rouge1'].fmeasure * 100)
                all_scores['rouge2'].append(scores['rouge2'].fmeasure * 100)
                all_scores['rougeL'].append(scores['rougeL'].fmeasure * 100)

    results = {k: round(sum(v) / max(len(v), 1), 2) for k, v in all_scores.items()}
    return results


def print_results_table(results_by_variant: dict):
    """Print a comparison table matching paper Tables 3-5."""
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


def run_evaluation(variant: str = None, batch_size: int = config.BATCH_SIZE):
    vocab = Vocabulary.load(os.path.join(config.DATA_PROCESSED, 'vocab.json'))
    matrix = load_embedding_matrix(
        os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy')
    )
    _, test_loader = get_dataloaders(batch_size=batch_size)

    if variant:
        variants_to_eval = [variant]
    else:
        variants_to_eval = ['seq2seq', 'attention', 'ptf_attention']

    results_by_variant = {}
    for v in variants_to_eval:
        print(f"\nEvaluating: {v}")
        try:
            model = load_model_from_checkpoint(v, len(vocab), matrix)
            scores = evaluate_model(model, test_loader, vocab)
            results_by_variant[v] = scores
            print(f"  ROUGE-1={scores['rouge1']:.2f} | "
                  f"ROUGE-2={scores['rouge2']:.2f} | "
                  f"ROUGE-L={scores['rougeL']:.2f}")
        except FileNotFoundError as e:
            print(f"  Skipping {v}: {e}")

    if len(results_by_variant) > 1:
        print_results_table(results_by_variant)

    # Save results
    results_path = os.path.join(config.CHECKPOINTS_DIR, 'evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_by_variant, f)
    print(f"\nResults saved → {results_path}")

    return results_by_variant


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default=None,
                        choices=['seq2seq', 'attention', 'ptf_attention'],
                        help='Evaluate a specific variant (default: all)')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all three variants and print comparison table')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    run_evaluation(
        variant=None if args.all else args.variant,
        batch_size=args.batch_size,
    )