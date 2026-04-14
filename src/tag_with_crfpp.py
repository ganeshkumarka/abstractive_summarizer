"""
src/tag_with_crfpp.py
---------------------
Use the pre-trained CRF++ Malayalam POS model (malayalam-pos-model.m)
via WSL to tag all Social-Sum-Mal source sentences.

Pipeline (mirrors run_malayalam_pos_model_and_save_to_file.sh):
  1. Write source sentences to a temp file
  2. Run create_features_for_pos_tagging.py to extract features
  3. Run crf_test via WSL: crf_test -m malayalam-pos-model.m feature_file
  4. Parse the output (cut column 1 and last column = predicted tag)
  5. Save tagged sentences to data/crfpp_tags.pkl

Then preprocess.py loads these pre-computed tags instead of recomputing.
This is a ONE-TIME step — tags are cached to disk.

Requirements:
  - WSL installed with CRF++ compiled (crf_test available)
  - malayalam-pos-model.m downloaded to project root
  - create_features_for_pos_tagging.py downloaded to project root

Usage:
    python src/tag_with_crfpp.py
    python src/tag_with_crfpp.py --model_path path/to/malayalam-pos-model.m
"""

import os, sys, subprocess, pickle, tempfile, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from datasets import load_dataset
import pandas as pd
import re


def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~0-9]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text):
    return [t for t in text.split() if t]


def windows_to_wsl_path(win_path: str) -> str:
    """Convert D:\\foo\\bar → /mnt/d/foo/bar for WSL."""
    win_path = win_path.replace('\\', '/')
    if len(win_path) >= 2 and win_path[1] == ':':
        drive = win_path[0].lower()
        rest  = win_path[2:]
        return f'/mnt/{drive}{rest}'
    return win_path


def run_crfpp_on_sentences(sentences: list,
                            model_path: str,
                            feature_script: str) -> dict:
    """
    Tag sentences using CRF++ via WSL.

    Args:
        sentences : list of token lists
        model_path: path to malayalam-pos-model.m
        feature_script: path to create_features_for_pos_tagging.py

    Returns:
        dict mapping tuple(tokens) → list of BIS tags
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_dir  = os.path.join(base_dir, 'data', 'tmp_crfpp')
    os.makedirs(tmp_dir, exist_ok=True)

    input_file   = os.path.join(tmp_dir, 'sentences.txt')
    feature_file = os.path.join(tmp_dir, 'features.txt')
    output_file  = os.path.join(tmp_dir, 'tagged.txt')

    # Step 1: Write sentences to file (one sentence per line, tokens space-separated)
    with open(input_file, 'w', encoding='utf-8') as f:
        for tokens in sentences:
            f.write(' '.join(tokens) + '\n')
    print(f"Written {len(sentences)} sentences to {input_file}")

    # Convert paths to WSL format
    wsl_input    = windows_to_wsl_path(input_file)
    wsl_feature  = windows_to_wsl_path(feature_file)
    wsl_output   = windows_to_wsl_path(output_file)
    wsl_model    = windows_to_wsl_path(os.path.abspath(model_path))
    wsl_script   = windows_to_wsl_path(os.path.abspath(feature_script))

    # Step 2: Extract features using the repo's script via WSL
    print("Step 2: Extracting CRF features via WSL...")
    feat_cmd = f'wsl python3 {wsl_script} --input {wsl_input} --output {wsl_feature}'
    result = subprocess.run(feat_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Feature extraction failed:\n{result.stderr}")
        return {}
    print(f"  Features saved to {feature_file}")

    # Step 3: Run crf_test via WSL
    print("Step 3: Running crf_test via WSL...")
    crf_cmd = f'wsl crf_test -m {wsl_model} {wsl_feature}'
    result  = subprocess.run(crf_cmd, shell=True,
                              capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"crf_test failed:\n{result.stderr}")
        # Try with full path to crf_test
        print("Trying with full path ~/crfpp/crf_test ...")
        crf_cmd2 = f'wsl ~/crfpp/crf_test -m {wsl_model} {wsl_feature}'
        result   = subprocess.run(crf_cmd2, shell=True,
                                   capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print(f"Still failed:\n{result.stderr}")
            return {}

    # Step 4: Parse CRF++ output
    # Format: token  pre1  pre2  pre3  pre4  suf1..suf7  len_cat  PREDICTED_TAG
    # Column 1 = token, last column = predicted tag
    print("Step 4: Parsing CRF++ output...")
    tagged_sentences = {}
    current_tokens, current_tags = [], []

    for line in result.stdout.split('\n'):
        line = line.strip()
        if not line:
            if current_tokens:
                key = tuple(current_tokens)
                tagged_sentences[key] = current_tags[:]
                current_tokens, current_tags = [], []
        else:
            parts = line.split('\t')
            if len(parts) >= 2:
                token = parts[0]
                # Normalise hierarchical tag
                raw_tag = parts[-1].strip()
                flat = raw_tag.split('_')[-1]
                tag_map = {'PSP': 'PP', 'VM': 'VF'}
                tag = tag_map.get(flat, flat)
                current_tokens.append(token)
                current_tags.append(tag)

    if current_tokens:
        tagged_sentences[tuple(current_tokens)] = current_tags

    print(f"Tagged {len(tagged_sentences)} unique sentences")

    # Step 5: Save to cache
    cache_path = os.path.join(config.DATA_PROCESSED, '..', 'crfpp_tags.pkl')
    cache_path = os.path.normpath(cache_path)
    with open(cache_path, 'wb') as f:
        pickle.dump(tagged_sentences, f)
    print(f"Tags cached → {cache_path}")

    # Cleanup
    for fp in [input_file, feature_file]:
        if os.path.exists(fp): os.remove(fp)

    return tagged_sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',   default='malayalam-pos-model.m')
    parser.add_argument('--feature_script', default='create_features_for_pos_tagging.py')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        print("Download from: https://github.com/Pruthwik/CRF-Based-Malayalam-POS-Tagger")
        sys.exit(1)

    if not os.path.exists(args.feature_script):
        print(f"Feature script not found: {args.feature_script}")
        print("Download from: https://github.com/Pruthwik/CRF-Based-Malayalam-POS-Tagger")
        sys.exit(1)

    # Load all source sentences
    print(f"Loading {config.HF_DATASET_NAME}...")
    ds = load_dataset(config.HF_DATASET_NAME)
    df = ds['train'].to_pandas()

    sentences = []
    for _, row in df.iterrows():
        text = row.get(config.INPUT_COL, '')
        if isinstance(text, str) and text.strip():
            tokens = tokenize(clean_text(text))
            if tokens:
                sentences.append(tokens)

    print(f"Loaded {len(sentences)} source sentences")
    tagged = run_crfpp_on_sentences(sentences, args.model_path,
                                     args.feature_script)

    if tagged:
        print("\nSample tags:")
        for key, tags in list(tagged.items())[:3]:
            print(f"  {list(key)[:5]} → {tags[:5]}")
        print("\nNow update preprocess.py to use crfpp_tags.pkl")
        print("Or just run: python src/preprocess.py")
        print("(preprocess.py will auto-use CRF++ tags if crfpp_tags.pkl exists)")
    else:
        print("\nTagging failed. Check WSL and CRF++ installation.")


if __name__ == '__main__':
    main()

    