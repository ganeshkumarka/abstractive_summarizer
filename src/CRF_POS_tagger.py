"""
src/crf_pos_tagger.py
---------------------
CRF-based Malayalam POS tagger using sklearn-crfsuite.

Replicates the feature extraction from:
  https://github.com/Pruthwik/CRF-Based-Malayalam-POS-Tagger

Features per token (from create_features_for_pos_tagging.py):
  - The token itself
  - Prefixes of length 1-4
  - Suffixes of length 1-7
  - Token length category (LESS if ≤4 chars, MORE otherwise)
  - Context: previous and next token features

This matches the paper's CRF tagger approach (Ajees & Idicula 2018, ref [1]).
Uses sklearn-crfsuite (pure Python, no CRF++ needed).

Two modes:
  1. 'train'  — train a new CRF model on BIS-tagged data
  2. 'load'   — load a pre-trained model
  3. 'rule'   — fallback to heuristic tagger (no model needed)

Install:
    pip install sklearn-crfsuite

Usage:
    tagger = CRFMalayalamPOSTagger(mode='rule')  # for now
    tokens = ['ജ്ഞാനോദയ', 'ചിന്തകർ', 'ആശയങ്ങൾ']
    tagged = tagger.tag(tokens)
    # → [('ജ്ഞാനോദയ', 'NN'), ('ചിന്തകർ', 'NN'), ('ആശയങ്ങൾ', 'NN')]
"""

import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Feature extraction (mirrors create_features_for_pos_tagging.py) ──────────

PREFIX_LEN = 4
SUFFIX_LEN = 7

def get_token_features(tokens: list, i: int) -> dict:
    """
    Extract CRF features for token at position i in the sentence.
    Mirrors the feature template from the GitHub repo.
    """
    token = tokens[i]
    tlen  = len(token)

    feats = {
        'bias':   1.0,
        'token':  token,
        'len_cat': 'LESS' if tlen <= 4 else 'MORE',
    }

    # Prefixes of length 1-4
    for k in range(1, PREFIX_LEN + 1):
        feats[f'pre{k}'] = token[:k] if tlen >= k else 'NULL'

    # Suffixes of length 1-7
    for k in range(1, SUFFIX_LEN + 1):
        feats[f'suf{k}'] = token[tlen - k:] if tlen >= k else 'NULL'

    # Previous token context
    if i > 0:
        prev = tokens[i - 1]
        feats['prev_token'] = prev
        for k in range(1, 3):
            feats[f'prev_suf{k}'] = prev[len(prev)-k:] if len(prev) >= k else 'NULL'
    else:
        feats['BOS'] = True

    # Next token context
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        feats['next_token'] = nxt
        for k in range(1, 3):
            feats[f'next_pre{k}'] = nxt[:k] if len(nxt) >= k else 'NULL'
    else:
        feats['EOS'] = True

    return feats


def sent_to_features(tokens: list) -> list:
    return [get_token_features(tokens, i) for i in range(len(tokens))]


# ── Heuristic fallback (same as before) ───────────────────────────────────────

SUFFIX_TAG_MAP = {
    'ുന്നു': 'VF', 'ിക്കുന്നു': 'VF', 'ിച്ചു': 'VF',
    'ിക്കും': 'VF', 'ുക': 'VINF', 'ാൻ': 'VINF',
    'ായ':  'JJ',  'ിയ': 'JJ',
    'ായി': 'RB',  'ായും': 'RB',
    'ുടെ': 'PP',  'ിന്': 'PP',
    'ഉം':  'CCD', 'ോ':   'CCD',
    'ില്ല': 'NEG', 'ല്ല': 'NEG',
    'ം':   'NN',  'നം':  'NN',
    'ൻ':   'NNP', 'ൾ':  'NNP',
}

def heuristic_tag(word: str) -> str:
    for suffix, tag in sorted(SUFFIX_TAG_MAP.items(), key=lambda x: -len(x[0])):
        if word.endswith(suffix) or word == suffix:
            return tag
    return 'NN'


# ── Main tagger class ──────────────────────────────────────────────────────────

class CRFMalayalamPOSTagger:
    """
    Drop-in replacement for MalayalamPOSTagger in preprocess.py.
    Supports three modes: 'rule', 'crf_sklearn', 'crf_pretrained'
    """

    def __init__(self, mode: str = 'rule', model_path: str = None):
        """
        Args:
            mode: 'rule'           — heuristic suffix tagger (current, fast)
                  'crf_sklearn'    — sklearn-crfsuite trained on annotated data
                  'crf_pretrained' — load a saved sklearn-crfsuite model
            model_path: path to saved .pkl model (for crf_pretrained mode)
        """
        self.mode = mode
        self.crf  = None
        self.tag2idx = {tag: i for i, tag in enumerate(config.BIS_TAGS)}

        if mode == 'crf_pretrained' and model_path:
            self.load(model_path)
        elif mode == 'crf_sklearn':
            try:
                import sklearn_crfsuite
                self.crf = sklearn_crfsuite.CRF(
                    algorithm='lbfgs',
                    c1=0.1, c2=0.1,
                    max_iterations=100,
                    all_possible_transitions=True,
                )
                print("CRF tagger ready (not yet trained — call .train(X, y))")
            except ImportError:
                print("sklearn-crfsuite not installed. pip install sklearn-crfsuite")
                print("Falling back to rule-based tagger")
                self.mode = 'rule'

    def train(self, X_train: list, y_train: list):
        """
        Train the CRF model.
        Args:
            X_train: list of sentences, each sentence = list of feature dicts
            y_train: list of sentences, each sentence = list of BIS tag strings
        """
        if self.crf is None:
            raise RuntimeError("CRF not initialised. Use mode='crf_sklearn'")
        print(f"Training CRF on {len(X_train)} sentences...")
        self.crf.fit(X_train, y_train)
        self.mode = 'crf_sklearn'
        print("CRF training complete")

    def tag(self, tokens: list) -> list:
        """
        Tag a list of tokens.
        Returns: list of (token, BIS_tag) tuples
        """
        if self.mode in ('crf_sklearn', 'crf_pretrained') and self.crf is not None:
            features = sent_to_features(tokens)
            tags = self.crf.predict_single(features)
        else:
            tags = [heuristic_tag(t) for t in tokens]

        return list(zip(tokens, tags))

    def tag2onehot(self, tag: str) -> list:
        """Convert BIS tag string → 33-dim one-hot vector."""
        vec = [0] * config.POS_DIM
        idx = self.tag2idx.get(tag, self.tag2idx.get('UNK', 0))
        vec[idx] = 1
        return vec

    def save(self, path: str):
        """Save trained CRF model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.crf, f)
        print(f"CRF model saved → {path}")

    def load(self, path: str):
        """Load a pre-trained CRF model."""
        with open(path, 'rb') as f:
            self.crf = pickle.load(f)
        self.mode = 'crf_pretrained'
        print(f"CRF model loaded ← {path}")


# ── Training helper: load BIS-tagged corpus ───────────────────────────────────

def load_conll_data(conll_path: str):
    """
    Load CoNLL-format tagged data.
    Expected format: one token per line, token TAB tag, blank line = sentence end.

    Returns:
        X: list of feature-dict lists (one per sentence)
        y: list of tag lists (one per sentence)
    """
    X, y = [], []
    tokens, tags = [], []

    with open(conll_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if tokens:
                    X.append(sent_to_features(tokens))
                    y.append(tags)
                    tokens, tags = [], []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[-1])   # last column = tag

    if tokens:
        X.append(sent_to_features(tokens))
        y.append(tags)

    print(f"Loaded {len(X)} sentences from {conll_path}")
    return X, y


def train_crf_from_conll(conll_path: str, save_path: str = None):
    """
    Train a CRF POS tagger from a CoNLL file and optionally save it.
    Use the Social-Sum-Mal data or any BIS-tagged Malayalam corpus.

    Returns the trained tagger.
    """
    tagger = CRFMalayalamPOSTagger(mode='crf_sklearn')
    X, y = load_conll_data(conll_path)
    tagger.train(X, y)
    if save_path:
        tagger.save(save_path)
    return tagger


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing CRF POS Tagger (rule-based mode):")
    tagger = CRFMalayalamPOSTagger(mode='rule')

    test_sentences = [
        ['എല്ലായിടത്തും', 'എല്ലാവർക്കും', 'സ്വാതന്ത്ര്യം', 'ലഭിച്ചിരുന്നോ'],
        ['ജ്ഞാനോദയ', 'ചിന്തകർ', 'ആശയങ്ങൾ', 'പ്രചരിപ്പിച്ചു'],
    ]

    for sent in test_sentences:
        tagged = tagger.tag(sent)
        print(f"  Input : {sent}")
        print(f"  Tagged: {tagged}")
        # Show one-hot for first token
        _, tag0 = tagged[0]
        oh = tagger.tag2onehot(tag0)
        print(f"  One-hot of '{tag0}': {oh[:10]}... (dim={len(oh)})")
        print()

    print("Feature extraction demo:")
    feats = get_token_features(['ജ്ഞാനോദയ', 'ചിന്തകർ'], 0)
    for k, v in list(feats.items())[:8]:
        print(f"  {k}: {v}")