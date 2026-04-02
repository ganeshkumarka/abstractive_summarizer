"""
src/check_leakage.py
--------------------
Checks for train/test source overlap after data augmentation.

When we augment by pairing each source with 3 summary types and then
shuffle randomly, the same source paragraph can appear in both train
and test with different summary types. This inflates ROUGE scores
because the model has seen the source text during training.
"""

import os, sys, pickle
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

train_path = os.path.join(config.DATA_PROCESSED, 'train.pkl')
test_path  = os.path.join(config.DATA_PROCESSED, 'test.pkl')

with open(train_path, 'rb') as f: train = pickle.load(f)
with open(test_path,  'rb') as f: test  = pickle.load(f)

# Use first 10 source tokens as a fingerprint
def fingerprint(sample):
    return tuple(sample['src_tokens'][:10])

train_fps = set(fingerprint(s) for s in train)
test_fps  = [fingerprint(s) for s in test]

overlap = sum(1 for fp in test_fps if fp in train_fps)
print(f"Train samples : {len(train)}")
print(f"Test samples  : {len(test)}")
print(f"Test samples whose SOURCE also appears in train: {overlap}/{len(test)} "
      f"({100*overlap/len(test):.1f}%)")

if overlap > 0:
    print("\nDATA LEAKAGE DETECTED.")
    print("The same source paragraph appears in both train and test")
    print("(with different summary types due to augmentation + random shuffle).")
    print("\nFix: split by SOURCE paragraph before augmenting, not after.")
else:
    print("\nNo leakage — Word2Vec+BiLSTM score of 50.79 is genuine.")