import torch
import torch.nn as nn
from transformers import AutoModel


class MuRILSeq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()

        # Encoder (MuRIL)
        self.encoder = AutoModel.from_pretrained("google/muril-base-cased")

        # Reduce 768 → hidden_dim
        self.enc_proj = nn.Linear(768, hidden_dim)

        # Decoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask, tgt_ids):
        # Encoder
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        enc_hidden = enc_out.last_hidden_state[:, 0, :]  # CLS token

        enc_hidden = self.enc_proj(enc_hidden).unsqueeze(0)  # (1, B, H)

        # Decoder input
        emb = self.embedding(tgt_ids)

        out, _ = self.decoder(emb, (enc_hidden, enc_hidden))
        logits = self.fc(out)

        return logits