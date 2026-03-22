"""
src/embedding.py
----------------
Implements the paper's embedding layer (§3.2):

  1. Train Word2Vec on the Malayalam corpus (semantic vectors)
  2. Build an embedding matrix for the full vocabulary
  3. PTF: concatenate [word_vector | pos_onehot] → new representation
     fed into the LSTM encoder

The combined embedding dim = WORD2VEC_DIM + POS_DIM (= 100 + 33 = 133 by default)

Usage:
    python src/embedding.py
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocess import Vocabulary


# ── Train Word2Vec ─────────────────────────────────────────────────────────────

def train_word2vec(sentences: list, save_path: str) -> Word2Vec:
    """
    Train a Word2Vec model on Malayalam token lists.
    
    Args:
        sentences: list of token lists (stemmed words)
        save_path: where to save the trained model
    Returns:
        trained gensim Word2Vec model
    """
    print(f"Training Word2Vec | dim={config.WORD2VEC_DIM}, "
          f"window={config.WORD2VEC_WIN}, epochs={config.WORD2VEC_ITER}")
    model = Word2Vec(
        sentences=sentences,
        vector_size=config.WORD2VEC_DIM,
        window=config.WORD2VEC_WIN,
        min_count=config.WORD2VEC_MIN,
        workers=4,
        epochs=config.WORD2VEC_ITER,
        seed=config.SEED,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"  Word2Vec trained on {len(model.wv)} unique tokens → saved to {save_path}")
    return model


def load_word2vec(path: str) -> Word2Vec:
    return Word2Vec.load(path)


# ── Build Embedding Matrix ─────────────────────────────────────────────────────

def build_embedding_matrix(vocab: Vocabulary, w2v_model: Word2Vec) -> np.ndarray:
    """
    Create a (vocab_size × WORD2VEC_DIM) matrix where each row is the
    Word2Vec vector for that token. Unknown tokens get random initialisation.

    Returns:
        numpy array of shape (vocab_size, WORD2VEC_DIM)
    """
    vocab_size = len(vocab)
    matrix = np.random.uniform(-0.1, 0.1, (vocab_size, config.WORD2VEC_DIM)).astype(np.float32)

    # PAD token → all zeros
    matrix[vocab.word2idx[config.PAD_TOKEN]] = np.zeros(config.WORD2VEC_DIM)

    hits = 0
    for word, idx in vocab.word2idx.items():
        if word in w2v_model.wv:
            matrix[idx] = w2v_model.wv[word]
            hits += 1

    print(f"Embedding matrix: {vocab_size} tokens, "
          f"{hits} covered by Word2Vec ({100*hits/vocab_size:.1f}%)")
    return matrix


def save_embedding_matrix(matrix: np.ndarray, path: str):
    np.save(path, matrix)
    print(f"Embedding matrix saved → {path}")


def load_embedding_matrix(path: str) -> np.ndarray:
    return np.load(path)


# ── PTF Embedding Layer (PyTorch Module) ──────────────────────────────────────

class PTFEmbedding(nn.Module):
    """
    POS Tagging Feature (PTF) Embedding Layer — the core contribution of the paper.

    For each input token at position t:
        word_vec  = Word2Vec_lookup(token_id)        # shape: (WORD2VEC_DIM,)
        pos_vec   = one_hot(BIS_tag)                 # shape: (POS_DIM,)
        ptf_vec   = concat([word_vec, pos_vec])      # shape: (EMBED_DIM,)

    This combined vector captures both:
      - Semantic information  (from Word2Vec)
      - Syntactic/morphological information (from BIS POS tag)

    The PTF vector is what gets fed into the LSTM encoder.

    When use_pos=False (baseline attention model), only word_vec is used
    and a linear projection maps it to EMBED_DIM for consistency.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_matrix: np.ndarray,
        use_pos: bool = True,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.use_pos = use_pos

        # Word embedding layer — initialised from pre-trained Word2Vec
        self.word_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.WORD2VEC_DIM,
            padding_idx=0,
        )
        self.word_embed.weight.data.copy_(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        if freeze_embeddings:
            self.word_embed.weight.requires_grad = False

        # Dropout on embeddings
        self.dropout = nn.Dropout(config.DROPOUT)

        if use_pos:
            # PTF: word_embed dim + POS one-hot dim
            self.output_dim = config.WORD2VEC_DIM + config.POS_DIM
        else:
            # Baseline: project word_embed to same output dim for fair comparison
            self.proj = nn.Linear(config.WORD2VEC_DIM, config.EMBED_DIM, bias=False)
            self.output_dim = config.EMBED_DIM

    def forward(self, token_ids: torch.Tensor, pos_onehot: torch.Tensor = None):
        """
        Args:
            token_ids : (batch, seq_len) — integer token indices
            pos_onehot: (batch, seq_len, POS_DIM) — one-hot POS vectors
                        Required when use_pos=True, ignored otherwise.
        Returns:
            embedding : (batch, seq_len, output_dim)
        """
        word_vecs = self.word_embed(token_ids)          # (B, S, WORD2VEC_DIM)
        word_vecs = self.dropout(word_vecs)

        if self.use_pos:
            if pos_onehot is None:
                raise ValueError("pos_onehot must be provided when use_pos=True")
            pos_vecs = pos_onehot.float()               # (B, S, POS_DIM)
            combined = torch.cat([word_vecs, pos_vecs], dim=-1)  # (B, S, EMBED_DIM)
            return combined
        else:
            return self.proj(word_vecs)                 # (B, S, EMBED_DIM)


# ── Convenience: build from saved files ───────────────────────────────────────

def build_ptf_embedding(vocab: Vocabulary, use_pos: bool = True) -> PTFEmbedding:
    """
    Load saved Word2Vec + embedding matrix and return a ready PTFEmbedding module.
    Call this from model.py.
    """
    w2v_path    = os.path.join(config.DATA_EMBEDDINGS, 'word2vec.model')
    matrix_path = os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy')

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(
            f"Embedding matrix not found at {matrix_path}. "
            "Run `python src/embedding.py` first."
        )

    matrix = load_embedding_matrix(matrix_path)
    return PTFEmbedding(
        vocab_size=len(vocab),
        embedding_matrix=matrix,
        use_pos=use_pos,
    )


# ── Main: train Word2Vec and build matrix ─────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(config.DATA_EMBEDDINGS, exist_ok=True)

    # Load preprocessed data
    processed_path = os.path.join(config.DATA_PROCESSED, 'train.pkl')
    vocab_path     = os.path.join(config.DATA_PROCESSED, 'vocab.json')

    if not os.path.exists(processed_path):
        raise FileNotFoundError("Run src/preprocess.py first.")

    with open(processed_path, 'rb') as f:
        train_data = pickle.load(f)

    vocab = Vocabulary.load(vocab_path)

    # Collect all sentences (source + target)
    sentences = [s['src_tokens'] for s in train_data] + \
                [s['tgt_tokens'] for s in train_data]

    # Train Word2Vec
    w2v = train_word2vec(
        sentences=sentences,
        save_path=os.path.join(config.DATA_EMBEDDINGS, 'word2vec.model')
    )

    # Build and save embedding matrix
    matrix = build_embedding_matrix(vocab, w2v)
    save_embedding_matrix(
        matrix,
        os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy')
    )

    print("Embedding stage complete.")