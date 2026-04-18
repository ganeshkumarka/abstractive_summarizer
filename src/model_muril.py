import torch
import torch.nn as nn
from transformers import AutoModel


class MuRILSeq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()

        self.encoder = AutoModel.from_pretrained("google/muril-base-cased")

        # freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.enc_hidden = 768
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.init_fc = nn.Linear(self.enc_hidden, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        with torch.no_grad():
            enc_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        enc_hidden = enc_outputs.last_hidden_state[:, 0]

        h0 = self.init_fc(enc_hidden).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        embedded = self.embedding(decoder_input_ids)

        outputs, _ = self.decoder(embedded, (h0, c0))

        logits = self.fc_out(outputs)

        return logits