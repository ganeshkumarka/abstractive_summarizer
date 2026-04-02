"""
src/preprocess.py  (v4 — data augmentation + MIN_FREQ=1)

Key changes:
  - MIN_FREQ=1: every token kept in vocab → UNK rate drops from 36% to ~5%
  - Data augmentation: each source paragraph is paired with ALL three summary
    types (long_summary, extreme_summary, answer_summary) → triples training data
    from 2000 → ~5400 pairs
  - Vocab capped at 8000 (raised from 6000 to handle larger token set)

Usage:
    python src/preprocess.py
"""

import os, re, json, pickle
import pandas as pd
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def clean_text(text: str) -> str:
    """Remove ASCII punctuation/digits only. Keep all Malayalam Unicode intact."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~0-9]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text: str) -> list:
    return [t for t in text.split() if t]


class MalayalamPOSTagger:
    SUFFIX_TAG_MAP = {
        'ുന്നു': 'VF', 'ിക്കുന്നു': 'VF', 'ിച്ചു': 'VF',
        'ിക്കും': 'VF', 'ുക': 'VINF', 'ാൻ': 'VINF',
        'ായ':  'JJ',  'ിയ': 'JJ',
        'ായി': 'RB',  'ായും': 'RB',
        'ുടെ': 'PP',  'ിന്': 'PP',
        'ഉം':  'CCD', 'ോ':   'CCD',
        'ില്ല': 'NEG', 'ല്ല': 'NEG',
        'ം':   'NN',  'നം': 'NN',
        'ൻ':   'NNP', 'ൾ':  'NNP',
    }

    def __init__(self):
        self.tag2idx = {tag: i for i, tag in enumerate(config.BIS_TAGS)}

    def _rule_tag(self, word):
        for suffix, tag in self.SUFFIX_TAG_MAP.items():
            if word.endswith(suffix) or word == suffix:
                return tag
        return 'NN'

    def tag(self, tokens):
        return [(t, self._rule_tag(t)) for t in tokens]

    def tag2onehot(self, tag):
        vec = [0] * config.POS_DIM
        idx = self.tag2idx.get(tag, self.tag2idx.get('UNK', 0))
        vec[idx] = 1
        return vec


class Vocabulary:
    MAX_VOCAB = 8000   # raised from 6000

    def __init__(self):
        self.word2idx = {
            config.PAD_TOKEN:   0,
            config.UNK_TOKEN:   1,
            config.START_TOKEN: 2,
            config.END_TOKEN:   3,
        }
        self.idx2word  = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()

    def build(self, sentences, min_freq=1):
        for tokens in sentences:
            self.word_freq.update(tokens)
        candidates = [(w, f) for w, f in self.word_freq.items()
                      if f >= min_freq and w not in self.word2idx]
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:self.MAX_VOCAB - len(self.word2idx)]
        for word, _ in candidates:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word
        print(f"Vocabulary: {len(self.word2idx)} tokens "
              f"(cap={self.MAX_VOCAB}, min_freq={min_freq})")

    def encode(self, tokens, max_len=None, add_special=False):
        ids = [self.word2idx.get(t, 1) for t in tokens]
        if add_special:
            ids = [2] + ids + [3]
        if max_len:
            ids = ids[:max_len]
            ids += [0] * (max_len - len(ids))
        return ids

    def decode(self, ids, skip_special=True):
        specials = {config.PAD_TOKEN, config.START_TOKEN,
                    config.END_TOKEN, config.UNK_TOKEN}
        words = []
        for i in ids:
            word = self.idx2word.get(i, config.UNK_TOKEN)
            if word == config.END_TOKEN:
                break
            if skip_special and word in specials:
                continue
            words.append(word)
        return words

    def unk_rate(self, token_lists):
        total = unk = 0
        for tokens in token_lists:
            for t in tokens:
                total += 1
                if t not in self.word2idx:
                    unk += 1
        return unk / max(total, 1)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'word2idx': self.word2idx,
                       'word_freq': dict(self.word_freq)},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        v.word2idx  = data['word2idx']
        v.idx2word  = {int(i): w for w, i in v.word2idx.items()}
        v.word_freq = Counter(data.get('word_freq', {}))
        return v


def process_text(text, tagger):
    cleaned = clean_text(text)
    tokens  = tokenize(cleaned)
    tagged  = tagger.tag(tokens)
    return [t for t, _ in tagged], [tag for _, tag in tagged]


def load_data():
    print(f"Loading: {config.HF_DATASET_NAME}")
    ds = load_dataset(config.HF_DATASET_NAME)
    df = ds['train'].to_pandas()
    print(f"  Loaded {len(df)} rows")
    return df


def run_preprocessing():
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)
    df = load_data()

    # All three summary columns available in Social-Sum-Mal
    SUMMARY_COLS = ['long_summary', 'extreme_summary', 'answer_summary']
    available = [c for c in SUMMARY_COLS if c in df.columns]
    print(f"  Using summary columns: {available}")

    tagger = MalayalamPOSTagger()
    processed = []

    print("Processing samples (data augmentation: 1 source → multiple summaries)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src_text = row.get(config.INPUT_COL, '')
        if not isinstance(src_text, str) or not src_text.strip():
            continue

        src_tokens, src_tags = process_text(src_text, tagger)
        if len(src_tokens) == 0:
            continue

        # Pair this source with EACH available summary type
        for col in available:
            tgt_text = row.get(col, '')
            if not isinstance(tgt_text, str) or not tgt_text.strip():
                continue
            tgt_tokens, _ = process_text(tgt_text, tagger)
            if len(tgt_tokens) == 0:
                continue
            processed.append({
                'src_tokens': src_tokens,
                'src_tags':   src_tags,
                'tgt_tokens': tgt_tokens,
                'summary_type': col,
            })

    print(f"  Total pairs after augmentation: {len(processed)}")
    by_type = Counter(s['summary_type'] for s in processed)
    for t, c in by_type.items():
        print(f"    {t}: {c}")

    # Build vocab with MIN_FREQ=1
    all_tokens = [s['src_tokens'] + s['tgt_tokens'] for s in processed]
    vocab = Vocabulary()
    vocab.build(all_tokens, min_freq=1)

    # Check UNK rate
    unk_rate = vocab.unk_rate(all_tokens)
    print(f"  UNK rate: {100*unk_rate:.1f}%", end="")
    if unk_rate > 0.10:
        print(" ← still high, consider raising MAX_VOCAB")
    else:
        print(" ✓")

    # Encode
    for s in processed:
        s['src_ids'] = vocab.encode(s['src_tokens'],
                                    max_len=config.MAX_INPUT_LEN)
        s['tgt_ids'] = vocab.encode(s['tgt_tokens'],
                                    max_len=config.MAX_SUMMARY_LEN,
                                    add_special=True)
        pos_vecs = [tagger.tag2onehot(tag) for tag in s['src_tags']]
        while len(pos_vecs) < config.MAX_INPUT_LEN:
            pos_vecs.append([0] * config.POS_DIM)
        s['src_pos'] = pos_vecs[:config.MAX_INPUT_LEN]

    # Sanity check
    s0 = processed[0]
    unk_id = 1
    print(f"\n  Sanity check:")
    print(f"    src[:6]: {s0['src_tokens'][:6]}")
    print(f"    tgt[:6]: {s0['tgt_tokens'][:6]}")
    print(f"    src UNKs: {s0['src_ids'][:config.MAX_INPUT_LEN].count(unk_id)}"
          f"/{config.MAX_INPUT_LEN}")
    print(f"    tgt UNKs: {s0['tgt_ids'].count(unk_id)}/{config.MAX_SUMMARY_LEN}")

    # SOURCE-LEVEL SPLIT — split the 2000 raw rows FIRST, then augment each half.
    # Splitting after augmentation caused 96.2% data leakage (same source paragraph
    # in both train and test with different summary types).
    import random
    random.seed(config.SEED)

    # Group processed samples by source fingerprint (first 8 tokens)
    from collections import defaultdict
    src_groups = defaultdict(list)
    for s in processed:
        fp = tuple(s['src_tokens'][:8])
        src_groups[fp].append(s)

    unique_sources = list(src_groups.keys())
    random.shuffle(unique_sources)
    split_idx = int(len(unique_sources) * config.TRAIN_SPLIT)

    train_sources = set(unique_sources[:split_idx])
    test_sources  = set(unique_sources[split_idx:])

    train_data = [s for s in processed if tuple(s['src_tokens'][:8]) in train_sources]
    test_data  = [s for s in processed if tuple(s['src_tokens'][:8]) in test_sources]

    print(f"\n  Unique source paragraphs: {len(unique_sources)}")
    print(f"  Train sources: {len(train_sources)} → {len(train_data)} pairs")
    print(f"  Test sources : {len(test_sources)} → {len(test_data)} pairs")
    print(f"  Leakage: 0% (source-level split guaranteed)")

    with open(os.path.join(config.DATA_PROCESSED, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(config.DATA_PROCESSED, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    vocab.save(os.path.join(config.DATA_PROCESSED, 'vocab.json'))
    print("Done. Files saved to data/processed/")
    return train_data, test_data, vocab


if __name__ == '__main__':
    run_preprocessing()