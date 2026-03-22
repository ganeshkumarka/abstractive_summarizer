"""
src/inference.py
----------------
Generate abstractive summaries from new Malayalam text using a trained model.

Usage:
    python src/inference.py --text "നിങ്ങളുടെ മലയാളം ഖണ്ഡിക ഇവിടെ"
    python src/inference.py --variant ptf_attention
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocess import Vocabulary, MalayalamStemmer, MalayalamPOSTagger, process_sample
from src.model import build_model
from src.embedding import load_embedding_matrix
from src.evaluate import load_model_from_checkpoint


def preprocess_input(text: str, stemmer, tagger, vocab: Vocabulary):
    """
    Preprocess a single Malayalam text string → (src_ids, src_pos, src_len).
    Returns tensors ready for model.generate().
    """
    tokens, tags = process_sample(text, stemmer, tagger)

    src_ids = vocab.encode(tokens, max_len=config.MAX_INPUT_LEN)
    pos_vecs = [tagger.tag2onehot(tag) for tag in tags]
    while len(pos_vecs) < config.MAX_INPUT_LEN:
        pos_vecs.append([0] * config.POS_DIM)
    src_pos = pos_vecs[:config.MAX_INPUT_LEN]

    src_ids_t = torch.tensor([src_ids], dtype=torch.long, device=config.DEVICE)
    src_pos_t = torch.tensor([src_pos], dtype=torch.float, device=config.DEVICE)
    src_len_t = torch.tensor([max(int(sum(1 for x in src_ids if x != 0)), 1)],
                              dtype=torch.long, device=config.DEVICE)

    return src_ids_t, src_pos_t, src_len_t


def summarize(text: str, model, vocab: Vocabulary,
              stemmer, tagger) -> str:
    """
    Generate an abstractive summary for a Malayalam text string.
    """
    src_ids, src_pos, src_len = preprocess_input(text, stemmer, tagger, vocab)

    start_idx = vocab.word2idx[config.START_TOKEN]
    end_idx   = vocab.word2idx[config.END_TOKEN]

    gen_ids = model.generate(
        src_ids, src_pos, src_len,
        max_len=config.MAX_SUMMARY_LEN,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    tokens = vocab.decode(gen_ids[0].tolist(), skip_special=True)
    return ' '.join(tokens)


def demo(variant: str = 'ptf_attention'):
    """Run a quick demo on a few examples from the test set."""
    from datasets import load_dataset

    print(f"Loading model: {variant}")
    vocab = Vocabulary.load(os.path.join(config.DATA_PROCESSED, 'vocab.json'))
    matrix = load_embedding_matrix(
        os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy')
    )
    model = load_model_from_checkpoint(variant, len(vocab), matrix)

    stemmer = MalayalamStemmer()
    tagger  = MalayalamPOSTagger(mode='rule')

    # Load a few examples from HuggingFace
    print("Loading dataset for demo examples...")
    ds = load_dataset(config.HF_DATASET_NAME)
    df = ds['train'].to_pandas()
    samples = df.sample(5, random_state=42)

    print("\n" + "="*70)
    for _, row in samples.iterrows():
        src_text = row[config.INPUT_COL]
        ref_text = row[config.SUMMARY_COL]
        gen_text = summarize(src_text, model, vocab, stemmer, tagger)

        print(f"\nINPUT   : {src_text[:120]}...")
        print(f"REFERENCE: {ref_text}")
        print(f"GENERATED: {gen_text}")
        print("-"*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default=None,
                        help='Malayalam text to summarize')
    parser.add_argument('--variant', type=str, default='ptf_attention',
                        choices=['seq2seq', 'attention', 'ptf_attention'])
    args = parser.parse_args()

    vocab = Vocabulary.load(os.path.join(config.DATA_PROCESSED, 'vocab.json'))
    matrix = load_embedding_matrix(
        os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy')
    )
    model = load_model_from_checkpoint(args.variant, len(vocab), matrix)
    stemmer = MalayalamStemmer()
    tagger  = MalayalamPOSTagger(mode='rule')

    if args.text:
        summary = summarize(args.text, model, vocab, stemmer, tagger)
        print(f"\nInput  : {args.text}")
        print(f"Summary: {summary}")
    else:
        demo(args.variant)