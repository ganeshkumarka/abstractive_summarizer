# """
# src/crf_pos_tagger.py
# ---------------------
# CRF-based Malayalam POS tagger.

# Supports two backends:
#   1. python-crfsuite — loads the pre-trained malayalam-pos-model.m directly
#      from https://github.com/Pruthwik/CRF-Based-Malayalam-POS-Tagger
#      No CRF++ installation needed. Works on Windows.

#   2. sklearn-crfsuite — pure Python CRF, train on annotated data

#   3. rule-based fallback — suffix heuristics, no dependencies

# Install:
#     pip install python-crfsuite   ← for pre-trained model
#     pip install sklearn-crfsuite  ← for training new model

# Usage (recommended — use pre-trained model):
#     # Download malayalam-pos-model.m from GitHub repo
#     from src.crf_pos_tagger import CRFMalayalamPOSTagger
#     tagger = CRFMalayalamPOSTagger(mode='pycrfsuite',
#                                     model_path='malayalam-pos-model.m')
#     tagged = tagger.tag(['ജ്ഞാനോദയ', 'ചിന്തകർ', 'ആശയങ്ങൾ'])
# """

# import os, sys, pickle
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config


# # ── Feature extraction (mirrors create_features_for_pos_tagging.py exactly) ──

# PREFIX_LEN = 4
# SUFFIX_LEN = 7

# def get_token_features_list(tokens: list, i: int) -> list:
#     """
#     Extract features as a LIST of strings — format required by pycrfsuite.
#     Matches the feature template in create_features_for_pos_tagging.py exactly:
#       token, pre1..pre4, suf1..suf7, len_cat
#     Plus context features for the CRF sequence model.
#     """
#     token = tokens[i]
#     tlen  = len(token)
#     feats = []

#     # Current token features (mirrors the repo's feature extractor)
#     feats.append(f'token={token}')
#     feats.append(f'len_cat={"LESS" if tlen <= 4 else "MORE"}')

#     for k in range(1, PREFIX_LEN + 1):
#         val = token[:k] if tlen >= k else 'NULL'
#         feats.append(f'pre{k}={val}')

#     for k in range(1, SUFFIX_LEN + 1):
#         val = token[tlen - k:] if tlen >= k else 'NULL'
#         feats.append(f'suf{k}={val}')

#     # Previous token context
#     if i > 0:
#         prev = tokens[i - 1]
#         feats.append(f'prev_token={prev}')
#         for k in range(1, 3):
#             feats.append(f'prev_suf{k}={prev[len(prev)-k:] if len(prev) >= k else "NULL"}')
#     else:
#         feats.append('BOS')

#     # Next token context
#     if i < len(tokens) - 1:
#         nxt = tokens[i + 1]
#         feats.append(f'next_token={nxt}')
#         for k in range(1, 3):
#             feats.append(f'next_pre{k}={nxt[:k] if len(nxt) >= k else "NULL"}')
#     else:
#         feats.append('EOS')

#     return feats


# def get_token_features(tokens: list, i: int) -> dict:
#     """Dict format for sklearn-crfsuite compatibility."""
#     token = tokens[i]
#     tlen  = len(token)
#     feats = {
#         'bias': 1.0,
#         'token': token,
#         'len_cat': 'LESS' if tlen <= 4 else 'MORE',
#     }
#     for k in range(1, PREFIX_LEN + 1):
#         feats[f'pre{k}'] = token[:k] if tlen >= k else 'NULL'
#     for k in range(1, SUFFIX_LEN + 1):
#         feats[f'suf{k}'] = token[tlen - k:] if tlen >= k else 'NULL'
#     if i > 0:
#         prev = tokens[i - 1]
#         feats['prev_token'] = prev
#         for k in range(1, 3):
#             feats[f'prev_suf{k}'] = prev[len(prev)-k:] if len(prev) >= k else 'NULL'
#     else:
#         feats['BOS'] = True
#     if i < len(tokens) - 1:
#         nxt = tokens[i + 1]
#         feats['next_token'] = nxt
#         for k in range(1, 3):
#             feats[f'next_pre{k}'] = nxt[:k] if len(nxt) >= k else 'NULL'
#     else:
#         feats['EOS'] = True
#     return feats


# def sent_to_features(tokens: list) -> list:
#     """For sklearn-crfsuite (list of dicts)."""
#     return [get_token_features(tokens, i) for i in range(len(tokens))]


# def sent_to_features_list(tokens: list) -> list:
#     """For pycrfsuite (list of list-of-strings)."""
#     return [get_token_features_list(tokens, i) for i in range(len(tokens))]


# # ── Tag normalisation (CoNLL hierarchical → flat BIS) ─────────────────────────

# CONLL_TAG_MAP = {'PSP': 'PP', 'VM': 'VF'}

# def normalise_tag(raw_tag: str) -> str:
#     """N_NN → NN, V_VM_VF → VF, PSP → PP, JJ → JJ"""
#     flat = raw_tag.strip().split('_')[-1]
#     return CONLL_TAG_MAP.get(flat, flat)


# # ── Heuristic fallback ────────────────────────────────────────────────────────

# SUFFIX_TAG_MAP = {
#     'ുന്നു': 'VF', 'ിക്കുന്നു': 'VF', 'ിച്ചു': 'VF',
#     'ിക്കും': 'VF', 'ുക': 'VINF', 'ാൻ': 'VINF',
#     'ായ':  'JJ',  'ിയ': 'JJ',
#     'ായി': 'RB',  'ായും': 'RB',
#     'ുടെ': 'PP',  'ിന്': 'PP',
#     'ഉം':  'CCD', 'ോ':   'CCD',
#     'ില്ല': 'NEG', 'ല്ല': 'NEG',
#     'ം':   'NN',  'നം':  'NN',
#     'ൻ':   'NNP', 'ൾ':  'NNP',
# }

# def heuristic_tag(word: str) -> str:
#     for suffix, tag in sorted(SUFFIX_TAG_MAP.items(), key=lambda x: -len(x[0])):
#         if word.endswith(suffix) or word == suffix:
#             return tag
#     return 'NN'


# # ── Main tagger class ─────────────────────────────────────────────────────────

# class CRFMalayalamPOSTagger:
#     """
#     Malayalam POS tagger. Four modes:

#     'pycrfsuite'   — loads malayalam-pos-model.m (the pre-trained CRF++ model)
#                      using python-crfsuite. BEST option — same model as the paper.
#                      pip install python-crfsuite

#     'crf_sklearn'  — sklearn-crfsuite, train on annotated CoNLL data
#                      pip install sklearn-crfsuite

#     'crf_pretrained' — load a saved sklearn-crfsuite .pkl model

#     'rule'         — heuristic suffix tagger, no dependencies (fallback)
#     """

#     def __init__(self, mode: str = 'rule', model_path: str = None):
#         self.mode    = mode
#         self.crf     = None
#         self.tagger  = None   # pycrfsuite tagger
#         self.tag2idx = {tag: i for i, tag in enumerate(config.BIS_TAGS)}

#         if mode == 'pycrfsuite':
#             self._load_pycrfsuite(model_path)
#         elif mode == 'crf_pretrained' and model_path:
#             self._load_sklearn(model_path)
#         elif mode == 'crf_sklearn':
#             self._init_sklearn()

#     def _load_pycrfsuite(self, model_path):
#         """Load the pre-trained CRF++ model using python-crfsuite."""
#         try:
#             import pycrfsuite
#             if not model_path or not os.path.exists(model_path):
#                 print(f"Model not found: {model_path}")
#                 print("Download malayalam-pos-model.m from:")
#                 print("  https://github.com/Pruthwik/CRF-Based-Malayalam-POS-Tagger")
#                 print("Falling back to rule-based tagger")
#                 self.mode = 'rule'
#                 return
#             self.tagger = pycrfsuite.Tagger()
#             self.tagger.open(model_path)
#             self.mode = 'pycrfsuite'
#             print(f"CRF++ model loaded (python-crfsuite) ← {model_path}")
#         except ImportError:
#             print("python-crfsuite not installed. Run: pip install python-crfsuite")
#             print("Falling back to rule-based tagger")
#             self.mode = 'rule'

#     def _load_sklearn(self, model_path):
#         with open(model_path, 'rb') as f:
#             self.crf = pickle.load(f)
#         self.mode = 'crf_pretrained'
#         print(f"sklearn-CRF model loaded ← {model_path}")

#     def _init_sklearn(self):
#         try:
#             import sklearn_crfsuite
#             self.crf = sklearn_crfsuite.CRF(
#                 algorithm='lbfgs', c1=0.1, c2=0.1,
#                 max_iterations=100, all_possible_transitions=True,
#             )
#             self.mode = 'crf_sklearn'
#         except ImportError:
#             print("sklearn-crfsuite not installed. Run: pip install sklearn-crfsuite")
#             self.mode = 'rule'

#     def train(self, X_train: list, y_train: list):
#         """Train sklearn-crfsuite model."""
#         if self.crf is None:
#             raise RuntimeError("CRF not initialised. Use mode='crf_sklearn'")
#         print(f"Training CRF on {len(X_train)} sentences...")
#         self.crf.fit(X_train, y_train)
#         print("CRF training complete")

#     def tag(self, tokens: list) -> list:
#         """
#         Tag a list of tokens. Returns list of (token, BIS_tag) tuples.
#         Automatically uses the best available backend.
#         """
#         if not tokens:
#             return []

#         if self.mode == 'pycrfsuite' and self.tagger is not None:
#             feats = sent_to_features_list(tokens)
#             tags  = self.tagger.tag(feats)
#             # Normalise CRF++ output tags (may be hierarchical)
#             tags = [normalise_tag(t) for t in tags]

#         elif self.mode in ('crf_sklearn', 'crf_pretrained') and self.crf is not None:
#             feats = sent_to_features(tokens)
#             tags  = self.crf.predict_single(feats)

#         else:
#             tags = [heuristic_tag(t) for t in tokens]

#         return list(zip(tokens, tags))

#     def tag2onehot(self, tag: str) -> list:
#         """Convert BIS tag string → 33-dim one-hot vector."""
#         vec = [0] * config.POS_DIM
#         idx = self.tag2idx.get(tag, self.tag2idx.get('UNK', 0))
#         vec[idx] = 1
#         return vec

#     def save(self, path: str):
#         """Save sklearn-crfsuite model to .pkl"""
#         with open(path, 'wb') as f:
#             pickle.dump(self.crf, f)
#         print(f"Model saved → {path}")

#     def load(self, path: str):
#         """Load sklearn-crfsuite .pkl model"""
#         self._load_sklearn(path)


# # ── CoNLL data loader ─────────────────────────────────────────────────────────

# def load_conll_data(conll_path: str):
#     """Load CoNLL tagged data. Handles flat and hierarchical BIS tags."""
#     X, y = [], []
#     tokens, tags = [], []

#     with open(conll_path, encoding='utf-8') as f:
#         for line in f:
#             line = line.rstrip()
#             if not line:
#                 if tokens:
#                     X.append(sent_to_features(tokens))
#                     y.append(tags)
#                     tokens, tags = [], []
#             else:
#                 parts = line.split('\t')
#                 if len(parts) >= 2:
#                     token = parts[0].strip()
#                     flat_tag = normalise_tag(parts[-1].strip())
#                     if token:
#                         tokens.append(token)
#                         tags.append(flat_tag)

#     if tokens:
#         X.append(sent_to_features(tokens))
#         y.append(tags)

#     from collections import Counter
#     all_tags = [t for s in y for t in s]
#     print(f"Loaded {len(X)} sentences, {len(all_tags)} tokens")
#     print(f"  Top tags: {Counter(all_tags).most_common(8)}")
#     return X, y


# # ── Demo ──────────────────────────────────────────────────────────────────────

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', default=None,
#                         help='Path to malayalam-pos-model.m (CRF++ model)')
#     args = parser.parse_args()

#     if args.model:
#         print(f"Loading pre-trained CRF++ model: {args.model}")
#         tagger = CRFMalayalamPOSTagger(mode='pycrfsuite', model_path=args.model)
#     else:
#         print("No model specified — using rule-based tagger")
#         print("To use pre-trained model: python src/crf_pos_tagger.py --model malayalam-pos-model.m")
#         tagger = CRFMalayalamPOSTagger(mode='rule')

#     test_sents = [
#         ['ജ്ഞാനോദയ', 'ചിന്തകർ', 'ആശയങ്ങൾ', 'പ്രചരിപ്പിച്ചു'],
#         ['ഈ', 'ആംബുലൻസുകൾ', 'ലഭ്യമല്ലെങ്കിൽ', 'മാത്രമേ', 'സേവനം', 'തേടാവൂ'],
#     ]
#     for sent in test_sents:
#         tagged = tagger.tag(sent)
#         print(f"Input : {sent}")
#         print(f"Tagged: {tagged}")
#         print()

#after crf++ issue
"""
src/crf_pos_tagger.py
---------------------
CRF-based Malayalam POS tagger.

Supports two backends:
  1. python-crfsuite — loads the pre-trained malayalam-pos-model.m directly
     from https://github.com/Pruthwik/CRF-Based-Malayalam-POS-Tagger
     No CRF++ installation needed. Works on Windows.

  2. sklearn-crfsuite — pure Python CRF, train on annotated data

  3. rule-based fallback — suffix heuristics, no dependencies

Install:
    pip install python-crfsuite   ← for pre-trained model
    pip install sklearn-crfsuite  ← for training new model

Usage (recommended — use pre-trained model):
    # Download malayalam-pos-model.m from GitHub repo
    from src.crf_pos_tagger import CRFMalayalamPOSTagger
    tagger = CRFMalayalamPOSTagger(mode='pycrfsuite',
                                    model_path='malayalam-pos-model.m')
    tagged = tagger.tag(['ജ്ഞാനോദയ', 'ചിന്തകർ', 'ആശയങ്ങൾ'])
"""

import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Feature extraction (mirrors create_features_for_pos_tagging.py exactly) ──

PREFIX_LEN = 4
SUFFIX_LEN = 7

def get_token_features_list(tokens: list, i: int) -> list:
    """
    Extract features as a LIST of strings — format required by pycrfsuite.
    Matches the feature template in create_features_for_pos_tagging.py exactly:
      token, pre1..pre4, suf1..suf7, len_cat
    Plus context features for the CRF sequence model.
    """
    token = tokens[i]
    tlen  = len(token)
    feats = []

    # Current token features (mirrors the repo's feature extractor)
    feats.append(f'token={token}')
    feats.append(f'len_cat={"LESS" if tlen <= 4 else "MORE"}')

    for k in range(1, PREFIX_LEN + 1):
        val = token[:k] if tlen >= k else 'NULL'
        feats.append(f'pre{k}={val}')

    for k in range(1, SUFFIX_LEN + 1):
        val = token[tlen - k:] if tlen >= k else 'NULL'
        feats.append(f'suf{k}={val}')

    # Previous token context
    if i > 0:
        prev = tokens[i - 1]
        feats.append(f'prev_token={prev}')
        for k in range(1, 3):
            feats.append(f'prev_suf{k}={prev[len(prev)-k:] if len(prev) >= k else "NULL"}')
    else:
        feats.append('BOS')

    # Next token context
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        feats.append(f'next_token={nxt}')
        for k in range(1, 3):
            feats.append(f'next_pre{k}={nxt[:k] if len(nxt) >= k else "NULL"}')
    else:
        feats.append('EOS')

    return feats


def get_token_features(tokens: list, i: int) -> dict:
    """Dict format for sklearn-crfsuite compatibility."""
    token = tokens[i]
    tlen  = len(token)
    feats = {
        'bias': 1.0,
        'token': token,
        'len_cat': 'LESS' if tlen <= 4 else 'MORE',
    }
    for k in range(1, PREFIX_LEN + 1):
        feats[f'pre{k}'] = token[:k] if tlen >= k else 'NULL'
    for k in range(1, SUFFIX_LEN + 1):
        feats[f'suf{k}'] = token[tlen - k:] if tlen >= k else 'NULL'
    if i > 0:
        prev = tokens[i - 1]
        feats['prev_token'] = prev
        for k in range(1, 3):
            feats[f'prev_suf{k}'] = prev[len(prev)-k:] if len(prev) >= k else 'NULL'
    else:
        feats['BOS'] = True
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        feats['next_token'] = nxt
        for k in range(1, 3):
            feats[f'next_pre{k}'] = nxt[:k] if len(nxt) >= k else 'NULL'
    else:
        feats['EOS'] = True
    return feats


def sent_to_features(tokens: list) -> list:
    """For sklearn-crfsuite (list of dicts)."""
    return [get_token_features(tokens, i) for i in range(len(tokens))]


def sent_to_features_list(tokens: list) -> list:
    """For pycrfsuite (list of list-of-strings)."""
    return [get_token_features_list(tokens, i) for i in range(len(tokens))]


# ── Tag normalisation (CoNLL hierarchical → flat BIS) ─────────────────────────

CONLL_TAG_MAP = {'PSP': 'PP', 'VM': 'VF'}

def normalise_tag(raw_tag: str) -> str:
    """N_NN → NN, V_VM_VF → VF, PSP → PP, JJ → JJ"""
    flat = raw_tag.strip().split('_')[-1]
    return CONLL_TAG_MAP.get(flat, flat)


# ── Heuristic fallback ────────────────────────────────────────────────────────

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


# ── Main tagger class ─────────────────────────────────────────────────────────

class CRFMalayalamPOSTagger:
    """
    Malayalam POS tagger. Four modes:

    'pycrfsuite'   — loads malayalam-pos-model.m (the pre-trained CRF++ model)
                     using python-crfsuite. BEST option — same model as the paper.
                     pip install python-crfsuite

    'crf_sklearn'  — sklearn-crfsuite, train on annotated CoNLL data
                     pip install sklearn-crfsuite

    'crf_pretrained' — load a saved sklearn-crfsuite .pkl model

    'rule'         — heuristic suffix tagger, no dependencies (fallback)
    """

    def __init__(self, mode: str = 'rule', model_path: str = None):
        self.mode    = mode
        self.crf     = None
        self.tagger  = None   # pycrfsuite tagger
        self.tag2idx = {tag: i for i, tag in enumerate(config.BIS_TAGS)}

        if mode == 'pycrfsuite':
            self._load_pycrfsuite(model_path)
        elif mode == 'crf_pretrained' and model_path:
            self._load_sklearn(model_path)
        elif mode == 'crf_sklearn':
            self._init_sklearn()

    def _load_pycrfsuite(self, model_path):
        """Load the pre-trained CRF++ model using python-crfsuite."""
        try:
            import pycrfsuite
            if not model_path or not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                print("Download malayalam-pos-model.m from:")
                print("  https://github.com/Pruthwik/CRF-Based-Malayalam-POS-Tagger")
                print("Falling back to rule-based tagger")
                self.mode = 'rule'
                return
            self.tagger = pycrfsuite.Tagger()
            self.tagger.open(model_path)
            self.mode = 'pycrfsuite'
            print(f"CRF++ model loaded (python-crfsuite) ← {model_path}")
        except ImportError:
            print("python-crfsuite not installed. Run: pip install python-crfsuite")
            print("Falling back to rule-based tagger")
            self.mode = 'rule'

    def _load_sklearn(self, model_path):
        with open(model_path, 'rb') as f:
            self.crf = pickle.load(f)
        self.mode = 'crf_pretrained'
        print(f"sklearn-CRF model loaded ← {model_path}")

    def _init_sklearn(self):
        try:
            import sklearn_crfsuite
            self.crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs', c1=0.1, c2=0.1,
                max_iterations=100, all_possible_transitions=True,
            )
            self.mode = 'crf_sklearn'
        except ImportError:
            print("sklearn-crfsuite not installed. Run: pip install sklearn-crfsuite")
            self.mode = 'rule'

    def train(self, X_train: list, y_train: list):
        """Train sklearn-crfsuite model."""
        if self.crf is None:
            raise RuntimeError("CRF not initialised. Use mode='crf_sklearn'")
        print(f"Training CRF on {len(X_train)} sentences...")
        self.crf.fit(X_train, y_train)
        print("CRF training complete")

    def tag(self, tokens: list) -> list:
        """
        Tag a list of tokens. Returns list of (token, BIS_tag) tuples.
        Automatically uses the best available backend.
        """
        if not tokens:
            return []

        if self.mode == 'pycrfsuite' and self.tagger is not None:
            feats = sent_to_features_list(tokens)
            tags  = self.tagger.tag(feats)
            # Normalise CRF++ output tags (may be hierarchical)
            tags = [normalise_tag(t) for t in tags]

        elif self.mode in ('crf_sklearn', 'crf_pretrained') and self.crf is not None:
            feats = sent_to_features(tokens)
            tags  = self.crf.predict_single(feats)

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
        """Save sklearn-crfsuite model to .pkl"""
        with open(path, 'wb') as f:
            pickle.dump(self.crf, f)
        print(f"Model saved → {path}")

    def load(self, path: str):
        """Load sklearn-crfsuite .pkl model"""
        self._load_sklearn(path)


# ── CoNLL data loader ─────────────────────────────────────────────────────────

def load_conll_data(conll_path: str):
    """Load CoNLL tagged data. Handles flat and hierarchical BIS tags."""
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
                    token = parts[0].strip()
                    flat_tag = normalise_tag(parts[-1].strip())
                    if token:
                        tokens.append(token)
                        tags.append(flat_tag)

    if tokens:
        X.append(sent_to_features(tokens))
        y.append(tags)

    from collections import Counter
    all_tags = [t for s in y for t in s]
    print(f"Loaded {len(X)} sentences, {len(all_tags)} tokens")
    print(f"  Top tags: {Counter(all_tags).most_common(8)}")
    return X, y


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None,
                        help='Path to malayalam-pos-model.m (CRF++ model)')
    args = parser.parse_args()

    if args.model:
        print(f"Loading pre-trained CRF++ model: {args.model}")
        tagger = CRFMalayalamPOSTagger(mode='pycrfsuite', model_path=args.model)
    else:
        print("No model specified — using rule-based tagger")
        print("To use pre-trained model: python src/crf_pos_tagger.py --model malayalam-pos-model.m")
        tagger = CRFMalayalamPOSTagger(mode='rule')

    test_sents = [
        ['ജ്ഞാനോദയ', 'ചിന്തകർ', 'ആശയങ്ങൾ', 'പ്രചരിപ്പിച്ചു'],
        ['ഈ', 'ആംബുലൻസുകൾ', 'ലഭ്യമല്ലെങ്കിൽ', 'മാത്രമേ', 'സേവനം', 'തേടാവൂ'],
    ]
    for sent in test_sents:
        tagged = tagger.tag(sent)
        print(f"Input : {sent}")
        print(f"Tagged: {tagged}")
        print()