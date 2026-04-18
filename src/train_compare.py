"""
src/train_compare.py

Compare:
1. Word2Vec + BiLSTM
2. MuRIL + BiLSTM
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

import config
from src.dataset import MalayalamSumDataset
from src.model import Seq2SeqModel
from src.embedding import PTFEmbedding
from src.evaluate import compute_rouge
from src.muril_seq2seq import MuRILSeq2Seq

# ===============================
# MuRIL Dataset (NEW)
# ===============================
# class MuRILDataset(Dataset):
#     def __init__(self, samples, tokenizer, vocab, max_len=128, max_tgt=40):
#         self.tokenizer = tokenizer
#         self.vocab = vocab
#         self.max_tgt = max_tgt
#         self.data = []

#         for s in samples:
#             if 'src_tokens' in s and 'tgt_tokens' in s:
#                 src = " ".join(s['src_tokens'])
#                 tgt = s['tgt_tokens']

#                 tgt_ids = [vocab.get(w, 1) for w in tgt]  # 1 = UNK
#                 tgt_ids = tgt_ids[:max_tgt]

#                 self.data.append((src, tgt_ids))

#     def __getitem__(self, idx):
#         src, tgt_ids = self.data[idx]

#         enc = self.tokenizer(
#             src,
#             padding="max_length",
#             truncation=True,
#             max_length=128,
#             return_tensors="pt"
#         )

#         tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)

#         return {
#             "input_ids": enc["input_ids"].squeeze(0),
#             "attention_mask": enc["attention_mask"].squeeze(0),
#             "tgt_ids": tgt_ids
#         }


# ===============================
# MuRIL + BiLSTM Model
# ===============================
class MuRILBiLSTM(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/muril-base-cased")
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 768)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        x, _ = self.lstm(x)
        return self.fc(x)


# ===============================
# Load Data
# ===============================
def load_data():
    with open(os.path.join(config.DATA_PROCESSED, 'train.pkl'), 'rb') as f:
        train_data = pickle.load(f)

    with open(os.path.join(config.DATA_PROCESSED, 'test.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    return train_data, test_data


# ===============================
# Word2Vec + BiLSTM
# ===============================
def train_w2v():
    print("\n=== Word2Vec + BiLSTM ===")

    train_data, test_data = load_data()

    train_ds = MalayalamSumDataset(train_data)
    test_ds  = MalayalamSumDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=4)

    with open(os.path.join(config.DATA_PROCESSED, 'vocab.json'),encoding='utf-8') as f:
        vocab = json.load(f)

    embedding = PTFEmbedding(
        embedding_path=os.path.join(config.DATA_EMBEDDINGS, 'embedding_matrix.npy')
    )

    model = Seq2SeqModel(
        vocab_size=len(vocab),
        embedding_dim=embedding.embedding_dim,
        hidden_dim=256,
        embedding_layer=embedding.embedding
    )

    print("Model ready (Word2Vec + BiLSTM)")
    print(f"Train batches: {len(train_loader)}")


# ===============================
# MuRIL + BiLSTM
# ===============================
def train_muril():
    print("\n=== MuRIL Seq2Seq Training ===")

    train_data, test_data = load_data()

    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

    # FIX encoding issue
    with open(os.path.join(config.DATA_PROCESSED, 'vocab.json'), encoding='utf-8') as f:
        vocab = json.load(f)

    from src.dataset_transformer import TransformerDataset

    train_ds = TransformerDataset(train_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    model = MuRILSeq2Seq(vocab_size=len(vocab)).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"Train batches: {len(train_loader)}")

    model.train()

    for epoch in range(3):
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            tgt_ids = batch["labels"].to(config.DEVICE)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, tgt_ids[:, :-1])

            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                tgt_ids[:, 1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# ===============================
# Evaluation
# ===============================
def evaluate_sample():
    ref = "ജ്ഞാനോദയ ചിന്തകർ പ്രചരിപ്പിച്ച നവോത്ഥാന ആശയങ്ങൾ ജനകീയ വിപ്ലവങ്ങൾക്ക് കാരണമായി"
    hyp = "നവോത്ഥാന ആശയങ്ങൾ സമൂഹത്തിൽ വലിയ മാറ്റങ്ങൾ സൃഷ്ടിച്ചു"

    score = compute_rouge(ref, hyp)
    print("\nSample ROUGE:", score)


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    print("\nChoose model:")
    print("1 → Word2Vec + BiLSTM")
    print("2 → MuRIL + BiLSTM")

    choice = input("Enter choice: ")

    if choice == "1":
        train_w2v()

    elif choice == "2":
        train_muril()

    else:
        print("Invalid choice")

    evaluate_sample()


def generate_summary(model, tokenizer, text, vocab, max_len=40):
    model.eval()

    inv_vocab = {v: k for k, v in vocab.items()}

    enc = tokenizer(text, return_tensors="pt").to(config.DEVICE)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    tgt_ids = torch.zeros((1, max_len), dtype=torch.long).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, tgt_ids)

    preds = outputs.argmax(-1).squeeze().cpu().numpy()

    words = [inv_vocab.get(i, "<UNK>") for i in preds]

    return " ".join(words)