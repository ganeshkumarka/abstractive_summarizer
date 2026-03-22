"""
config.py
---------
Central config for all hyperparameters and paths.
Edit this file to run experiments — never hardcode values in other files.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
DATA_EMBEDDINGS = os.path.join(BASE_DIR, "data", "embeddings")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# HuggingFace dataset (Social-Sum-Mal)
HF_DATASET_NAME = "rahulraj2k16/Social-Sum-Mal"

# ── Data columns ───────────────────────────────────────────────────────────────
# Which columns to use from the Social-Sum-Mal dataset
INPUT_COL   = "input"           # source paragraph
SUMMARY_COL = "long_summary"    # change to "extreme_summary" for headline task

# ── Preprocessing ──────────────────────────────────────────────────────────────
MAX_INPUT_LEN   = 100   # max tokens in source (pad/truncate)
MAX_SUMMARY_LEN = 30    # max tokens in summary
MIN_FREQ        = 2     # minimum word frequency to keep in vocab

# Special tokens
PAD_TOKEN   = "<PAD>"
UNK_TOKEN   = "<UNK>"
START_TOKEN = "<START>"
END_TOKEN   = "<END>"

# ── Word2Vec Embedding ─────────────────────────────────────────────────────────
WORD2VEC_DIM  = 100     # word embedding size
WORD2VEC_WIN  = 5       # context window
WORD2VEC_MIN  = 2       # min count
WORD2VEC_ITER = 10      # training epochs for Word2Vec

# ── POS Tagging Feature (PTF) ──────────────────────────────────────────────────
# BIS tagset has 11 categories (paper Table 2)
# We one-hot encode them → 11-dim vector
# Combined embedding = WORD2VEC_DIM + POS_DIM
BIS_TAGS = [
    "NN", "NNP", "NST",          # Noun
    "JJ",                         # Adjective
    "PRP", "PRF", "PRL", "PRC", "PRQ",  # Pronoun
    "RB",                         # Adverb
    "DMD", "DMR", "DMQ",         # Demonstrative
    "PP",                         # Preposition
    "VF", "VNF", "VINF", "VAUX", # Verb
    "CCD", "CCS", "UT",          # Conjunction
    "RPD", "CL", "INJ", "NEG",   # Particles
    "QTF", "QTC", "QTO",         # Quantifiers
    "RDF", "SYM", "PUNC", "UNK", "ECH",  # Residuals
]
POS_DIM       = len(BIS_TAGS)   # ~33 dims
EMBED_DIM     = WORD2VEC_DIM + POS_DIM  # final embedding size fed to encoder

# ── Model Architecture ─────────────────────────────────────────────────────────
HIDDEN_DIM    = 256     # LSTM hidden state size (encoder & decoder)
NUM_LAYERS    = 1       # LSTM layers (paper uses 1)
DROPOUT       = 0.3

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 128     # paper tests 32, 64, 128 — best at 128
EPOCHS        = 90      # paper trains for 90 epochs
LEARNING_RATE = 1e-3
CLIP_GRAD     = 5.0     # gradient clipping threshold
TRAIN_SPLIT   = 0.8     # 80% train, 20% test (paper uses ~3500/2046 split)
SEED          = 42

# ── Evaluation ─────────────────────────────────────────────────────────────────
BEAM_SIZE     = 4       # beam search width at inference

# ── Device ─────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model variants to compare (matches paper Table 3-5) ───────────────────────
MODEL_VARIANTS = {
    "seq2seq":       {"use_attention": False, "use_pos": False},
    "attention":     {"use_attention": True,  "use_pos": False},
    "ptf_attention": {"use_attention": True,  "use_pos": True},   # ← proposed model
}