"""
config.py  (v6 — data augmentation + MIN_FREQ=1)
"""
import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
DATA_EMBEDDINGS = os.path.join(BASE_DIR, "data", "embeddings")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

HF_DATASET_NAME = "rahulraj2k16/Social-Sum-Mal"

INPUT_COL   = "input"
SUMMARY_COL = "extreme_summary"   # used only for inference demo; training uses all 3

MAX_INPUT_LEN   = 150
MAX_SUMMARY_LEN = 30    # covers long_summary p95=27 and extreme_summary p95=10
MIN_FREQ        = 1     # keep ALL words — drops UNK rate from 36% to ~5%

PAD_TOKEN   = "<PAD>"
UNK_TOKEN   = "<UNK>"
START_TOKEN = "<START>"
END_TOKEN   = "<END>"

WORD2VEC_DIM  = 100
WORD2VEC_WIN  = 5
WORD2VEC_MIN  = 1       # match MIN_FREQ
WORD2VEC_ITER = 15

BIS_TAGS = [
    "NN", "NNP", "NST", "JJ",
    "PRP", "PRF", "PRL", "PRC", "PRQ",
    "RB", "DMD", "DMR", "DMQ", "PP",
    "VF", "VNF", "VINF", "VAUX",
    "CCD", "CCS", "UT",
    "RPD", "CL", "INJ", "NEG",
    "QTF", "QTC", "QTO",
    "RDF", "SYM", "PUNC", "UNK", "ECH",
]
POS_DIM   = len(BIS_TAGS)
EMBED_DIM = WORD2VEC_DIM + POS_DIM

HIDDEN_DIM    = 128
NUM_LAYERS    = 1
DROPOUT       = 0.4

BATCH_SIZE    = 64
EPOCHS        = 150
LEARNING_RATE = 5e-4
CLIP_GRAD     = 1.0
TRAIN_SPLIT   = 0.8
SEED          = 42

LABEL_SMOOTHING       = 0.1
TF_START              = 0.9
TF_END                = 0.3
ENCODER_FREEZE_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 20
SAVE_EVERY              = 5
BEAM_SIZE               = 4

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_VARIANTS = {
    "seq2seq":       {"use_attention": False, "use_pos": False},
    "attention":     {"use_attention": True,  "use_pos": False},
    "ptf_attention": {"use_attention": True,  "use_pos": True},
}