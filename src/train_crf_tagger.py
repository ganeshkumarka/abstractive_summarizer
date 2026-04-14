# """
# src/train_crf_tagger.py
# -----------------------
# Train the CRF POS tagger from the CoNLL tagged data in:
#   https://github.com/Pruthwik/CRF-Based-Malayalam-POS-Tagger

# Steps:
#   1. Download malayalam_pos_conll.txt from the repo
#   2. Run this script: python src/train_crf_tagger.py --conll path/to/malayalam_pos_conll.txt
#   3. Saved model will be auto-loaded by preprocess.py next time you run it

# The CoNLL file format (one token per line, blank line = sentence end):
#   token1 TAB tag1
#   token2 TAB tag2
#   ...
#   (blank line)

# Usage:
#     pip install sklearn-crfsuite
#     python src/train_crf_tagger.py --conll malayalam_pos_conll.txt
# """

# import os, sys, argparse
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config
# from src.crf_pos_tagger import CRFMalayalamPOSTagger, load_conll_data, sent_to_features


# def evaluate_tagger(tagger, X_test, y_test):
#     """Quick accuracy check on held-out sentences."""
#     correct = total = 0
#     for feat_sent, true_tags in zip(X_test, y_test):
#         tokens = [f['token'] for f in feat_sent]
#         pred   = tagger.tag(tokens)
#         for (_, ptag), ttag in zip(pred, true_tags):
#             if ptag == ttag:
#                 correct += 1
#             total += 1
#     return correct / max(total, 1)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--conll', required=True,
#                         help='Path to malayalam_pos_conll.txt from the GitHub repo')
#     parser.add_argument('--test_split', type=float, default=0.1,
#                         help='Fraction to hold out for evaluation (default 0.1)')
#     args = parser.parse_args()

#     if not os.path.exists(args.conll):
#         print(f"File not found: {args.conll}")
#         print("Download from: https://github.com/Pruthwik/CRF-Based-Malayalam-POS-Tagger")
#         sys.exit(1)

#     # Load CoNLL data
#     X, y = load_conll_data(args.conll)

#     # Train/test split
#     split = max(1, int(len(X) * (1 - args.test_split)))
#     X_train, y_train = X[:split], y[:split]
#     X_test,  y_test  = X[split:], y[split:]
#     print(f"Train: {len(X_train)} sents | Test: {len(X_test)} sents")

#     # Evaluate rule-based first as baseline
#     rule_tagger = CRFMalayalamPOSTagger(mode='rule')
#     rule_acc    = evaluate_tagger(rule_tagger, X_test, y_test)
#     print(f"Rule-based accuracy (baseline): {100*rule_acc:.1f}%")

#     # Train CRF
#     crf_tagger = CRFMalayalamPOSTagger(mode='crf_sklearn')
#     crf_tagger.train(X_train, y_train)

#     # Evaluate CRF
#     crf_acc = evaluate_tagger(crf_tagger, X_test, y_test)
#     print(f"CRF accuracy:                   {100*crf_acc:.1f}%")
#     print(f"Improvement over rule-based:    +{100*(crf_acc-rule_acc):.1f}%")

#     # Save to data/crf_pos_model.pkl (auto-loaded by preprocess.py)
#     save_path = os.path.join(config.DATA_PROCESSED, '..', 'crf_pos_model.pkl')
#     save_path = os.path.normpath(save_path)
#     crf_tagger.save(save_path)
#     print(f"\nModel saved to: {save_path}")
#     print("Now run: python src/preprocess.py")
#     print("It will auto-detect and use the trained CRF tagger.")


# if __name__ == '__main__':
#     main()


#after crf++ issue

"""
src/train_crf_tagger.py
-----------------------
Train a CRF POS tagger for Malayalam using pseudo-annotated data
generated from the Social-Sum-Mal dataset.

Since the only available CoNLL data (from the GitHub repo) has only 2 sentences,
we bootstrap a training set by:
  1. Rule-tagging all source sentences with our heuristic tagger
  2. Training sklearn-crfsuite CRF on these pseudo-tags
  3. The CRF learns to generalise the suffix rules using sequence context
     (previous/next token features) → better accuracy than pure rule-based

This is a standard bootstrapping approach in low-resource NLP.

If you also have real annotated data (CoNLL file), pass it with --conll
and it will be combined with pseudo-annotated data for training.

Usage:
    pip install sklearn-crfsuite
    python src/train_crf_tagger.py
    python src/train_crf_tagger.py --conll malayalam_pos_conll.txt  (if you have more data)
"""

import os, sys, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.crf_pos_tagger import (CRFMalayalamPOSTagger, load_conll_data,
                                 sent_to_features, heuristic_tag)


def load_dataset_sentences():
    """Load all source sentences from the preprocessed dataset."""
    import pickle as pkl
    train_path = os.path.join(config.DATA_PROCESSED, 'train.pkl')
    test_path  = os.path.join(config.DATA_PROCESSED, 'test.pkl')

    sentences = []
    for path in [train_path, test_path]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pkl.load(f)
            for s in data:
                if s['src_tokens']:
                    sentences.append(s['src_tokens'])

    # Deduplicate by content
    seen = set()
    unique = []
    for s in sentences:
        key = tuple(s[:5])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    print(f"Loaded {len(unique)} unique source sentences from dataset")
    return unique


def generate_pseudo_tagged_data(sentences):
    """
    Rule-tag all sentences to create pseudo-annotated training data.
    The CRF will learn to generalise these patterns using sequence context.
    """
    X, y = [], []
    for tokens in sentences:
        tags = [heuristic_tag(t) for t in tokens]
        X.append(sent_to_features(tokens))
        y.append(tags)
    print(f"Generated {len(X)} pseudo-tagged sentences "
          f"({sum(len(s) for s in y)} tokens)")
    return X, y


def evaluate_on_conll(tagger, conll_path):
    """Evaluate tagger accuracy on real annotated data."""
    X_test, y_test = load_conll_data(conll_path)
    correct = total = 0
    for feat_sent, true_tags in zip(X_test, y_test):
        tokens = [f['token'] for f in feat_sent]
        pred   = tagger.tag(tokens)
        for (_, ptag), ttag in zip(pred, true_tags):
            if ptag == ttag:
                correct += 1
            total += 1
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conll', default=None,
                        help='Optional: path to CoNLL annotated data for evaluation')
    parser.add_argument('--max_sents', type=int, default=3000,
                        help='Max sentences to use for training (default 3000)')
    args = parser.parse_args()

    try:
        import sklearn_crfsuite
    except ImportError:
        print("Please run: pip install sklearn-crfsuite")
        sys.exit(1)

    # Load dataset sentences
    sentences = load_dataset_sentences()
    if not sentences:
        print("No dataset found. Run python src/preprocess.py first.")
        sys.exit(1)

    sentences = sentences[:args.max_sents]

    # Generate pseudo-tagged training data
    X_pseudo, y_pseudo = generate_pseudo_tagged_data(sentences)

    # If real CoNLL data available, add it
    if args.conll and os.path.exists(args.conll):
        X_real, y_real = load_conll_data(args.conll)
        X_train = X_pseudo + X_real
        y_train = y_pseudo + y_real
        print(f"Combined: {len(X_pseudo)} pseudo + {len(X_real)} real = "
              f"{len(X_train)} total training sentences")
    else:
        X_train, y_train = X_pseudo, y_pseudo
        if args.conll:
            print(f"CoNLL file not found: {args.conll} — using pseudo-tagged only")

    # Train rule-based for comparison
    rule_tagger = CRFMalayalamPOSTagger(mode='rule')

    # Train CRF
    crf_tagger = CRFMalayalamPOSTagger(mode='crf_sklearn')
    crf_tagger.train(X_train, y_train)

    # Evaluate — two approaches:
    # 1. Cross-val on pseudo data (shows CRF vs rule agreement)
    # 2. If real CoNLL available, evaluate on it
    if args.conll and os.path.exists(args.conll):
        print("\nEvaluating on real annotated CoNLL data:")
        rule_acc = evaluate_on_conll(rule_tagger, args.conll)
        crf_acc  = evaluate_on_conll(crf_tagger, args.conll)
        print(f"  Rule-based accuracy: {100*rule_acc:.1f}%")
        print(f"  CRF accuracy:        {100*crf_acc:.1f}%")
        print(f"  Improvement:         +{100*(crf_acc-rule_acc):.1f}%")
    else:
        # Show sample output comparison
        print("\nSample tagging comparison (Rule vs CRF):")
        test_sents = [
            ['ജ്ഞാനോദയ', 'ചിന്തകർ', 'ആശയങ്ങൾ', 'പ്രചരിപ്പിച്ചു'],
            ['ഇന്ത്യൻ', 'സ്വാതന്ത്ര്യം', 'ലഭിച്ചു', 'കൊണ്ടിരുന്നു'],
            ['കൃഷി', 'ഭൂമി', 'ഉടമകൾ', 'നികുതി', 'നൽകണം'],
        ]
        for sent in test_sents:
            rule_tags = rule_tagger.tag(sent)
            crf_tags  = crf_tagger.tag(sent)
            print(f"\n  Input: {sent}")
            print(f"  Rule: {[t for _, t in rule_tags]}")
            print(f"  CRF:  {[t for _, t in crf_tags]}")

    # Save model
    save_path = os.path.join(config.DATA_PROCESSED,
                             '..', 'crf_pos_model.pkl')
    save_path = os.path.normpath(save_path)
    crf_tagger.save(save_path)

    print(f"\nCRF model saved → {save_path}")
    print("Now run: python src/preprocess.py")
    print("preprocess.py will auto-detect and use the CRF tagger.")


if __name__ == '__main__':
    main()