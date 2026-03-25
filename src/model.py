"""
src/model.py  (v4 — added repetition penalty in generate())

Fix: repetition_penalty blocks tokens that appeared in the last N positions
from being generated again. This prevents the ഒരു/ൽ repetition loop.
"""

import os, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.attention import BahdanauAttention
from src.embedding import PTFEmbedding


class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded, src_lengths):
        embedded = self.dropout(embedded)
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden, cell) = self.lstm(packed)
        encoder_outputs, _ = pad_packed_sequence(packed_out, batch_first=True)
        return encoder_outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers=1, dropout=0.3, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        if use_attention:
            self.attention = BahdanauAttention(hidden_dim)
            lstm_input_dim = embed_dim + hidden_dim
        else:
            lstm_input_dim = embed_dim

        self.lstm   = nn.LSTM(
            input_size=lstm_input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, input_token, hidden, cell,
                     encoder_outputs, src_mask=None):
        embedded = self.dropout(self.embed(input_token.unsqueeze(1)))

        attn_weights = None
        if self.use_attention:
            query = hidden[-1]
            context, attn_weights = self.attention(query, encoder_outputs, src_mask)
            lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        else:
            lstm_input = embedded

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell, attn_weights


class Seq2SeqModel(nn.Module):

    def __init__(self, vocab_size, embedding_matrix,
                 use_attention=True, use_pos=True):
        super().__init__()
        self.use_pos       = use_pos
        self.use_attention = use_attention

        self.ptf_embed = PTFEmbedding(
            vocab_size=vocab_size,
            embedding_matrix=embedding_matrix,
            use_pos=use_pos,
        )
        embed_out_dim = self.ptf_embed.output_dim

        self.encoder = Encoder(
            embed_dim=embed_out_dim,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=config.WORD2VEC_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            use_attention=use_attention,
        )
        self.enc2dec_h = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        self.enc2dec_c = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)

    def _init_decoder_state(self, enc_hidden, enc_cell):
        return (torch.tanh(self.enc2dec_h(enc_hidden)),
                torch.tanh(self.enc2dec_c(enc_cell)))

    def _make_src_mask(self, src_ids, enc_len):
        """Trim mask to actual encoder output length (not MAX_INPUT_LEN)."""
        return (src_ids[:, :enc_len] == 0)

    def forward(self, src_ids, src_pos, tgt_ids, src_lengths,
                teacher_forcing_ratio=0.5):
        tgt_len = tgt_ids.size(1)

        src_embedded = self.ptf_embed(src_ids, src_pos)
        enc_out, enc_h, enc_c = self.encoder(src_embedded, src_lengths)
        dec_h, dec_c = self._init_decoder_state(enc_h, enc_c)
        src_mask     = self._make_src_mask(src_ids, enc_out.size(1))

        outputs   = []
        dec_input = tgt_ids[:, 0]

        for t in range(1, tgt_len):
            pred, dec_h, dec_c, _ = self.decoder.forward_step(
                dec_input, dec_h, dec_c, enc_out, src_mask
            )
            outputs.append(pred)
            use_tf    = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = tgt_ids[:, t] if use_tf else pred.argmax(dim=-1)

        return torch.stack(outputs, dim=1)

    def generate(self, src_ids, src_pos, src_lengths,
                 max_len=config.MAX_SUMMARY_LEN,
                 start_idx=2, end_idx=3,
                 block_unk=True,
                 no_repeat_ngram=3):
        """
        Greedy decoding with:
          block_unk=True      — never output PAD(0) or UNK(1)
          no_repeat_ngram=3   — block any token seen in the last 3 positions
                                to prevent "ഒരു ഒരു ഒരു..." repetition loops
        """
        self.eval()
        with torch.no_grad():
            src_embedded = self.ptf_embed(src_ids, src_pos)
            enc_out, enc_h, enc_c = self.encoder(src_embedded, src_lengths)
            dec_h, dec_c = self._init_decoder_state(enc_h, enc_c)
            src_mask     = self._make_src_mask(src_ids, enc_out.size(1))

            B         = src_ids.size(0)
            dec_input = torch.full((B,), start_idx,
                                   dtype=torch.long, device=src_ids.device)
            generated = []
            done      = torch.zeros(B, dtype=torch.bool, device=src_ids.device)

            for step in range(max_len):
                pred, dec_h, dec_c, _ = self.decoder.forward_step(
                    dec_input, dec_h, dec_c, enc_out, src_mask
                )

                # Block PAD and UNK from being output
                if block_unk:
                    pred[:, 0] = float('-inf')
                    pred[:, 1] = float('-inf')

                # Repetition penalty: block tokens seen in last N positions
                if no_repeat_ngram > 0 and step >= 1:
                    recent = generated[-no_repeat_ngram:]   # list of (B,) tensors
                    for prev_tok in recent:
                        for b in range(B):
                            pred[b, prev_tok[b].item()] = float('-inf')

                next_tok = pred.argmax(dim=-1)
                generated.append(next_tok)
                done |= (next_tok == end_idx)
                if done.all():
                    break
                dec_input = next_tok

        return torch.stack(generated, dim=1)


def build_model(vocab_size, embedding_matrix, variant='ptf_attention'):
    if variant not in config.MODEL_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'")
    flags = config.MODEL_VARIANTS[variant]
    model = Seq2SeqModel(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        use_attention=flags['use_attention'],
        use_pos=flags['use_pos'],
    ).to(config.DEVICE)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{variant}' | params={n:,} | device={config.DEVICE}")
    return model