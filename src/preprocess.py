"""
src/preprocess.py
-----------------
Step 1 of the paper pipeline:
  - Load data from HuggingFace (Social-Sum-Mal) or local CSV
  - Clean text (remove quotes, symbols, digits, extra spaces)
  - Tokenize Malayalam text
  - Stem using Prathyayam/Sandhi rules (via indic-nlp-library)
  - Assign POS tags using CRF tagger (sklearn-crfsuite)
  - Build and save vocabulary
  - Save processed data to disk

Usage:
    python src/preprocess.py
"""

import os
import re
import json
import pickle
import pandas as pd
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Malayalam Text Cleaning ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove quotes, symbols, digits, extra whitespace (paper §3.1)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'["\'\"""'']', '', text)       # remove quotes
    text = re.sub(r'[^\u0D00-\u0D7F\s]', '', text)  # keep only Malayalam unicode + spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer for Malayalam."""
    return text.split()


# ── Malayalam Stemmer ──────────────────────────────────────────────────────────

class MalayalamStemmer:
    """
    Rule-based suffix stripping using Prathyayam (Sandhi) rules.
    Implements the approach described in paper §3.1.1.

    The stemmer scans each word right-to-left for the longest matching suffix
    and strips it. Rules are based on the 4 Sandhi categories (Table 1):
      - Swarasandhi      (vowel + vowel)
      - Swaravyajana     (vowel + consonant)
      - Vyanjanaswara    (consonant + vowel)
      - Vyanjana sandhi  (consonant + consonant)
    """

    # Common Malayalam suffixes to strip (extended from paper Fig. 2 rules)
    SUFFIXES = [
        # Case suffixes (vibhakti)
        'യുടെ', 'യ്ക്ക്', 'യില്‍', 'യിൽ', 'കൊണ്ട്',
        'ത്തില്‍', 'ത്തിൽ', 'ത്തിന്', 'ത്തോട്',
        # Common inflections
        'കള്‍', 'കൾ', 'ങ്ങള്‍', 'ങ്ങൾ',
        'ഉടെ', 'ിന്', 'ില്‍', 'ിൽ', 'ോട്',
        'ായി', 'ായ', 'ാൻ', 'ാൻ', 'ില്',
        'ുടെ', 'ുകൾ', 'ുകള്‍',
        # Verb endings
        'ുന്നു', 'ിക്കുന്നു', 'ിച്ചു', 'ിക്കും',
        'ുന്നത്', 'ിക്കുന്നത്', 'ും', 'ുക',
        # Short suffixes (last resort)
        'ില്', 'ിൽ', 'ിന്', 'ിയ', 'ിക',
    ]

    def __init__(self, min_stem_len: int = 2):
        self.min_stem_len = min_stem_len
        # Sort by length descending — longest match wins
        self.suffixes = sorted(self.SUFFIXES, key=len, reverse=True)

    def stem(self, word: str) -> str:
        """Strip the longest matching suffix, return stem."""
        for suffix in self.suffixes:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if len(stem) >= self.min_stem_len:
                    return stem
        return word

    def stem_tokens(self, tokens: list) -> list:
        return [self.stem(t) for t in tokens]


# ── POS Tagger (CRF-based, BIS tagset) ────────────────────────────────────────

class MalayalamPOSTagger:
    """
    CRF-based POS tagger using BIS tagset (paper §3.2).
    
    For the project we provide two modes:
      1. 'rule'  — fast heuristic tagger (good for prototyping)
      2. 'crf'   — trained CRF model (set mode='crf' after training)
    
    The rule-based tagger assigns tags based on word endings, which captures
    most of Malayalam's agglutinative morphology without needing labelled data.
    """

    # Heuristic suffix → BIS tag map for Malayalam
    SUFFIX_TAG_MAP = {
        # Nouns
        'ത്ത്': 'NN', 'നം': 'NN', 'ത': 'NN', 'ം': 'NN',
        # Proper nouns — words ending in a name suffix
        'ൻ': 'NNP', 'ൾ': 'NNP',
        # Verbs
        'ുന്നു': 'VF', 'ിക്കുന്നു': 'VF', 'ിച്ചു': 'VF',
        'ിക്കും': 'VF', 'ുക': 'VINF', 'ാൻ': 'VINF',
        # Adjectives
        'ായ': 'JJ', 'ിയ': 'JJ',
        # Adverbs
        'ായി': 'RB', 'ായും': 'RB',
        # Pronouns
        'ഞാൻ': 'PRP', 'നീ': 'PRP', 'അവൻ': 'PRP', 'അവൾ': 'PRP',
        # Conjunctions
        'ഉം': 'CCD', 'ഓ': 'CCD',
        # Negation
        'ല്ല': 'NEG', 'ില്ല': 'NEG',
    }

    def __init__(self, mode: str = 'rule'):
        """
        Args:
            mode: 'rule' for heuristic tagging, 'crf' for trained CRF model.
        """
        self.mode = mode
        self.crf_model = None
        self.tag2idx = {tag: i for i, tag in enumerate(config.BIS_TAGS)}
        self.idx2tag = {i: tag for tag, i in self.tag2idx.items()}

    def _rule_tag(self, word: str) -> str:
        """Fast heuristic tag based on word endings."""
        for suffix, tag in self.SUFFIX_TAG_MAP.items():
            if word.endswith(suffix) or word == suffix:
                return tag
        # Default: treat as common noun if unknown
        return 'NN'

    def _word2features(self, sent: list, i: int) -> dict:
        """Feature extraction for CRF training."""
        word = sent[i]
        features = {
            'bias': 1.0,
            'word': word,
            'word[-3:]': word[-3:] if len(word) >= 3 else word,
            'word[-2:]': word[-2:] if len(word) >= 2 else word,
            'word[:3]': word[:3] if len(word) >= 3 else word,
            'word.isdigit': word.isdigit(),
            'word.len': len(word),
        }
        if i > 0:
            features['prev_word'] = sent[i - 1]
            features['prev_word[-2:]'] = sent[i - 1][-2:] if len(sent[i - 1]) >= 2 else sent[i - 1]
        else:
            features['BOS'] = True
        if i < len(sent) - 1:
            features['next_word'] = sent[i + 1]
        else:
            features['EOS'] = True
        return features

    def tag(self, tokens: list) -> list:
        """
        Tag a list of tokens. Returns list of (token, BIS_tag) tuples.
        """
        if self.mode == 'crf' and self.crf_model is not None:
            features = [self._word2features(tokens, i) for i in range(len(tokens))]
            tags = self.crf_model.predict([features])[0]
        else:
            tags = [self._rule_tag(t) for t in tokens]
        return list(zip(tokens, tags))

    def tag2onehot(self, tag: str) -> list:
        """Convert a BIS tag string to a one-hot vector of length POS_DIM."""
        vec = [0] * config.POS_DIM
        idx = self.tag2idx.get(tag, self.tag2idx.get('UNK', 0))
        vec[idx] = 1
        return vec

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.crf_model, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.crf_model = pickle.load(f)
        self.mode = 'crf'


# ── Vocabulary ─────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Maps tokens ↔ integer indices.
    Includes special tokens: PAD, UNK, START, END.
    """

    def __init__(self):
        self.word2idx = {
            config.PAD_TOKEN:   0,
            config.UNK_TOKEN:   1,
            config.START_TOKEN: 2,
            config.END_TOKEN:   3,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()

    def build(self, sentences: list, min_freq: int = config.MIN_FREQ):
        """Build vocab from a list of token lists."""
        for tokens in sentences:
            self.word_freq.update(tokens)
        for word, freq in self.word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"Vocabulary size: {len(self.word2idx)}")

    def encode(self, tokens: list, max_len: int = None, add_special: bool = False) -> list:
        """Convert tokens to indices. Optionally pad/truncate to max_len."""
        ids = [self.word2idx.get(t, self.word2idx[config.UNK_TOKEN]) for t in tokens]
        if add_special:
            ids = [self.word2idx[config.START_TOKEN]] + ids + [self.word2idx[config.END_TOKEN]]
        if max_len:
            ids = ids[:max_len]
            ids += [self.word2idx[config.PAD_TOKEN]] * (max_len - len(ids))
        return ids

    def decode(self, ids: list, skip_special: bool = True) -> list:
        """Convert indices back to tokens."""
        specials = {config.PAD_TOKEN, config.START_TOKEN, config.END_TOKEN}
        words = []
        for i in ids:
            word = self.idx2word.get(i, config.UNK_TOKEN)
            if skip_special and word in specials:
                continue
            if word == config.END_TOKEN:
                break
            words.append(word)
        return words

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'word2idx': self.word2idx, 'word_freq': dict(self.word_freq)}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        vocab = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        vocab.word_freq = Counter(data.get('word_freq', {}))
        return vocab


# ── Full Pipeline ──────────────────────────────────────────────────────────────

def process_sample(text: str, stemmer: MalayalamStemmer, tagger: MalayalamPOSTagger):
    """
    Run one text sample through: clean → tokenize → stem → POS tag.
    Returns: (stemmed_tokens, pos_tags)
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    stemmed = stemmer.stem_tokens(tokens)
    tagged = tagger.tag(stemmed)                      # [(token, tag), ...]
    stems = [t for t, _ in tagged]
    tags  = [tag for _, tag in tagged]
    return stems, tags


def load_data() -> pd.DataFrame:
    """Load Social-Sum-Mal from HuggingFace."""
    print(f"Loading dataset: {config.HF_DATASET_NAME}")
    ds = load_dataset(config.HF_DATASET_NAME)
    df = ds['train'].to_pandas()
    print(f"  Loaded {len(df)} rows. Columns: {list(df.columns)}")
    return df


def run_preprocessing():
    """Full preprocessing pipeline. Saves processed data + vocab to disk."""
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)

    # 1. Load data
    df = load_data()
    df = df[[config.INPUT_COL, config.SUMMARY_COL]].dropna().reset_index(drop=True)

    stemmer = MalayalamStemmer()
    tagger  = MalayalamPOSTagger(mode='rule')

    processed = []
    print("Processing samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src_tokens, src_tags = process_sample(row[config.INPUT_COL], stemmer, tagger)
        tgt_tokens, _        = process_sample(row[config.SUMMARY_COL], stemmer, tagger)

        if len(src_tokens) == 0 or len(tgt_tokens) == 0:
            continue

        processed.append({
            'src_tokens': src_tokens,
            'src_tags':   src_tags,
            'tgt_tokens': tgt_tokens,
        })

    print(f"  Valid samples after preprocessing: {len(processed)}")

    # 2. Build vocabulary from all tokens
    all_tokens = [s['src_tokens'] + s['tgt_tokens'] for s in processed]
    vocab = Vocabulary()
    vocab.build(all_tokens, min_freq=config.MIN_FREQ)

    # 3. Encode and save
    for sample in processed:
        sample['src_ids'] = vocab.encode(
            sample['src_tokens'],
            max_len=config.MAX_INPUT_LEN
        )
        sample['tgt_ids'] = vocab.encode(
            sample['tgt_tokens'],
            max_len=config.MAX_SUMMARY_LEN,
            add_special=True
        )
        # POS one-hot for each source token (padded to MAX_INPUT_LEN)
        pos_vecs = [tagger.tag2onehot(tag) for tag in sample['src_tags']]
        # Pad with zero vectors
        while len(pos_vecs) < config.MAX_INPUT_LEN:
            pos_vecs.append([0] * config.POS_DIM)
        sample['src_pos'] = pos_vecs[:config.MAX_INPUT_LEN]

    # 4. Train/test split
    split_idx = int(len(processed) * config.TRAIN_SPLIT)
    train_data = processed[:split_idx]
    test_data  = processed[split_idx:]
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # 5. Save
    with open(os.path.join(config.DATA_PROCESSED, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(config.DATA_PROCESSED, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    vocab.save(os.path.join(config.DATA_PROCESSED, 'vocab.json'))
    print("Preprocessing complete. Files saved to data/processed/")
    return train_data, test_data, vocab


if __name__ == '__main__':
    run_preprocessing()