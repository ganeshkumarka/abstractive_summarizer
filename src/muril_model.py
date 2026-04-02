"""
src/muril_model.py
------------------
Proposed novel architecture: MuRIL-BiLSTM with POS-Gated Attention

Components:
  1. MuRIL encoder  — google/muril-base-cased pretrained on 17 Indian languages
                      replaces Word2Vec; understands Malayalam morphology contextually
  2. BiLSTM layer   — stacked on MuRIL hidden states; bidirectional captures
                      both left and right context (upgrade from paper's LSTM)
  3. POS-Gated Attention — POS tag scores directly gate attention weights
                           instead of concatenating POS to embeddings
  4. LSTM decoder   — same as paper

Ablation variants supported (set flags in build_muril_model):
  A. Word2Vec + LSTM     (paper baseline — already trained)
  B. Word2Vec + BiLSTM   (BiLSTM contribution isolated)
  C. MuRIL + BiLSTM      (MuRIL contribution)
  D. MuRIL + BiLSTM + POS-gate (full proposed model)

Install:
    pip install transformers

Usage:
    python src/train_muril.py --variant muril_bilstm_pos
"""

import os, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── MuRIL Embedding ────────────────────────────────────────────────────────────

class MuRILEmbedding(nn.Module):
    """
    Contextual embedding using google/muril-base-cased.

    MuRIL is pretrained on:
      - Wikipedia in 17 Indian languages (including Malayalam)
      - Common Crawl data for Indian languages
      - Transliterated content

    This replaces Word2Vec + POS-one-hot concatenation from the paper.
    MuRIL's hidden states implicitly encode POS information through
    contextual training — a verb in different sentence positions gets
    different representations.

    We freeze MuRIL during early training and unfreeze later for fine-tuning.
    Output is projected from 768 → MURIL_OUTPUT_DIM to control model size.
    """

    MURIL_OUTPUT_DIM = 128    # match HIDDEN_DIM — reduces params and mismatch

    def __init__(self, model_name='google/muril-base-cased', freeze=True):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert      = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise ImportError(
                f"Could not load {model_name}. Run: pip install transformers\n"
                f"Then: from transformers import AutoModel\n"
                f"Original error: {e}"
            )

        self.proj    = nn.Linear(768, self.MURIL_OUTPUT_DIM)
        self.dropout = nn.Dropout(0.1)
        self.output_dim = self.MURIL_OUTPUT_DIM

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
            print(f"MuRIL loaded (frozen) → projected to {self.MURIL_OUTPUT_DIM}d")
        else:
            print(f"MuRIL loaded (trainable) → projected to {self.MURIL_OUTPUT_DIM}d")

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids     : (B, seq_len) — MuRIL tokenizer output
            attention_mask: (B, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            (B, seq_len, MURIL_OUTPUT_DIM)
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h   = out.last_hidden_state           # (B, S, 768)
        h   = self.dropout(h)
        return self.proj(h)                   # (B, S, 256)

    def unfreeze(self):
        """Call after initial training to fine-tune MuRIL."""
        for p in self.bert.parameters():
            p.requires_grad = True
        print("MuRIL unfrozen for fine-tuning")


# ── BiLSTM Encoder ─────────────────────────────────────────────────────────────

class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder stacked on top of MuRIL representations.

    Why BiLSTM over LSTM:
      - LSTM reads left-to-right only. For Malayalam, which is SOV (Subject-Object-Verb),
        the verb appears at the end — a left-to-right LSTM processes it last.
      - BiLSTM reads both directions; the forward pass sees the subject first,
        the backward pass sees the verb first. The combined representation is
        richer for morphologically complex, agglutinative languages like Malayalam.

    Output: projects concatenated [forward, backward] → hidden_dim
    """

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Project [forward(H) || backward(H)] → H
        self.proj_out    = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proj_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proj_cell   = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, embedded, src_lengths):
        """
        Args:
            embedded   : (B, S, input_dim)
            src_lengths: (B,)
        Returns:
            encoder_outputs: (B, S, hidden_dim)   — projected bidirectional states
            hidden         : (1, B, hidden_dim)   — merged final hidden
            cell           : (1, B, hidden_dim)   — merged final cell
        """
        embedded = self.dropout(embedded)
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden, cell) = self.lstm(packed)
        enc_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # enc_out: (B, S, 2*H) — project to (B, S, H)
        enc_out = self.proj_out(enc_out)

        # Merge bidirectional final states
        # hidden: (2, B, H) → cat → (B, 2H) → proj → (B, H) → (1, B, H)
        B = hidden.size(1)
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=-1)
        cell_cat   = torch.cat([cell[0],   cell[1]],   dim=-1)
        merged_h = torch.tanh(self.proj_hidden(hidden_cat)).unsqueeze(0)
        merged_c = torch.tanh(self.proj_cell(cell_cat)).unsqueeze(0)

        return enc_out, merged_h, merged_c


# ── POS-Gated Attention ────────────────────────────────────────────────────────

class POSGatedAttention(nn.Module):
    """
    POS-Gated Attention — the architectural novelty.

    Standard Bahdanau attention:
        e_ij = V · tanh(W_s · s_{i-1} + W_h · h_j)
        a_ij = softmax(e_ij)

    POS-Gated Attention (proposed):
        e_ij = V · tanh(W_s · s_{i-1} + W_h · h_j)
        g_j  = sigmoid(W_p · pos_j)          ← POS gate per source position
        e_ij = e_ij * g_j                     ← gate scales alignment score
        a_ij = softmax(e_ij)

    The gate g_j is a learned scalar per source position j, derived from
    its BIS POS tag. It learns to up-weight nouns/verbs (salient content)
    and down-weight conjunctions/particles (less informative for summary).

    This is more principled than the paper's approach (which concatenates
    POS to word embeddings at the input, far from where it matters).
    By injecting POS at the attention layer, we directly control WHICH
    source words the decoder focuses on at each generation step.
    """

    def __init__(self, hidden_dim, pos_dim):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V   = nn.Linear(hidden_dim, 1, bias=False)
        # POS gate: maps POS one-hot → scalar gate per position
        self.W_p = nn.Linear(pos_dim, 1, bias=True)

    def forward(self, decoder_hidden, encoder_outputs, pos_onehot, src_mask=None):
        """
        Args:
            decoder_hidden : (B, H)
            encoder_outputs: (B, S, H)
            pos_onehot     : (B, S, POS_DIM) — BIS POS one-hot for each source token
            src_mask       : (B, S) bool — True = PAD position

        Returns:
            context_vector : (B, H)
            attn_weights   : (B, S)
        """
        B, S, H = encoder_outputs.size()

        dec_exp = decoder_hidden.unsqueeze(1).expand(-1, S, -1)  # (B, S, H)

        # Standard alignment score
        energy = torch.tanh(
            self.W_s(dec_exp) + self.W_h(encoder_outputs)
        )                                                          # (B, S, H)
        scores = self.V(energy).squeeze(-1)                        # (B, S)

        # POS gate — learns which POS categories deserve attention
        pos_float = pos_onehot.float()                             # (B, S, POS_DIM)
        gate = torch.sigmoid(self.W_p(pos_float)).squeeze(-1)      # (B, S)

        # Gate scales the alignment scores
        gated_scores = scores * gate                               # (B, S)

        if src_mask is not None:
            gated_scores = gated_scores.masked_fill(src_mask, float('-inf'))

        attn_weights = F.softmax(gated_scores, dim=-1)             # (B, S)
        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)                                               # (B, H)

        return context, attn_weights


# ── Decoder (LSTM, same as paper) ─────────────────────────────────────────────

class LSTMDecoder(nn.Module):
    """LSTM decoder compatible with both standard and POS-gated attention."""

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 pos_dim, num_layers=1, dropout=0.3, use_pos_gate=True):
        super().__init__()
        self.use_pos_gate = use_pos_gate
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        if use_pos_gate:
            self.attention = POSGatedAttention(hidden_dim, pos_dim)
        else:
            from src.attention import BahdanauAttention
            self.attention = BahdanauAttention(hidden_dim)

        self.lstm   = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, input_token, hidden, cell,
                     encoder_outputs, pos_onehot, src_mask=None):
        embedded = self.dropout(
            self.embed(input_token.unsqueeze(1))
        )                                                     # (B, 1, E)
        query = hidden[-1]                                    # (B, H)

        if self.use_pos_gate:
            context, attn = self.attention(
                query, encoder_outputs, pos_onehot, src_mask
            )
        else:
            context, attn = self.attention(query, encoder_outputs, src_mask)

        lstm_in = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        out, (hidden, cell) = self.lstm(lstm_in, (hidden, cell))
        pred = self.fc_out(out.squeeze(1))                    # (B, V)
        return pred, hidden, cell, attn


# ── Full MuRIL-BiLSTM Model ────────────────────────────────────────────────────

class MuRILBiLSTMModel(nn.Module):
    """
    Full proposed model: MuRIL + BiLSTM + POS-Gated Attention.

    Ablation flags:
      use_muril=False  → use pretrained Word2Vec embedding instead
      use_bilstm=False → use unidirectional LSTM encoder
      use_pos_gate=False → use standard Bahdanau attention (no POS gate)
    """

    def __init__(self, vocab_size, embedding_matrix=None,
                 use_muril=True, use_bilstm=True, use_pos_gate=True,
                 muril_model_name='google/muril-base-cased',
                 freeze_muril=True):
        super().__init__()
        self.use_muril    = use_muril
        self.use_bilstm   = use_bilstm
        self.use_pos_gate = use_pos_gate

        # ── Embedding ──────────────────────────────────────────────
        if use_muril:
            self.embedding = MuRILEmbedding(muril_model_name, freeze=freeze_muril)
            enc_input_dim  = self.embedding.output_dim
        else:
            from src.embedding import PTFEmbedding
            self.embedding = PTFEmbedding(
                vocab_size=vocab_size,
                embedding_matrix=embedding_matrix,
                use_pos=True,
            )
            enc_input_dim = self.embedding.output_dim

        # ── Encoder ────────────────────────────────────────────────
        if use_bilstm:
            self.encoder = BiLSTMEncoder(
                input_dim=enc_input_dim,
                hidden_dim=config.HIDDEN_DIM,
                num_layers=config.NUM_LAYERS,
                dropout=config.DROPOUT,
            )
        else:
            from src.model import Encoder
            self.encoder = Encoder(
                embed_dim=enc_input_dim,
                hidden_dim=config.HIDDEN_DIM,
                num_layers=config.NUM_LAYERS,
                dropout=config.DROPOUT,
            )

        # ── Decoder ────────────────────────────────────────────────
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=config.WORD2VEC_DIM,
            hidden_dim=config.HIDDEN_DIM,
            pos_dim=config.POS_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            use_pos_gate=use_pos_gate,
        )

        self.enc2dec_h = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        self.enc2dec_c = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)

    def _make_src_mask(self, src_ids, enc_len):
        return (src_ids[:, :enc_len] == 0)

    def _encode(self, src_ids, src_pos, src_lengths,
                muril_input_ids=None, muril_attn_mask=None):
        if self.use_muril:
            embedded = self.embedding(muril_input_ids, muril_attn_mask)
            # CRITICAL: MuRIL tokenizes to muril_max_len (128), not MAX_INPUT_LEN (150).
            # src_lengths counts tokens in our vocab tokenizer — wrong length for MuRIL.
            # Use actual non-padding lengths from muril_attn_mask instead.
            muril_lengths = muril_attn_mask.sum(dim=1).clamp(min=1).cpu()
            enc_out, h, c = self.encoder(embedded, muril_lengths)
        else:
            embedded = self.embedding(src_ids, src_pos)
            enc_out, h, c = self.encoder(embedded, src_lengths)
        return enc_out, h, c

    def forward(self, src_ids, src_pos, tgt_ids, src_lengths,
                teacher_forcing_ratio=0.5,
                muril_input_ids=None, muril_attn_mask=None):

        tgt_len = tgt_ids.size(1)
        enc_out, enc_h, enc_c = self._encode(
            src_ids, src_pos, src_lengths, muril_input_ids, muril_attn_mask
        )
        dec_h = torch.tanh(self.enc2dec_h(enc_h))
        dec_c = torch.tanh(self.enc2dec_c(enc_c))

        enc_len = enc_out.size(1)

        if self.use_muril and muril_attn_mask is not None:
            # For MuRIL: src_mask comes from MuRIL attention mask (0=pad, 1=real)
            # Trim to enc_len in case muril_max_len > enc_len after packing
            muril_mask_trimmed = muril_attn_mask[:, :enc_len]
            src_mask = (muril_mask_trimmed == 0)   # True = PAD
        else:
            src_mask = self._make_src_mask(src_ids, enc_len)

        # POS for attention gate — trim to enc_len
        pos_trimmed = src_pos[:, :enc_len, :]

        outputs   = []
        dec_input = tgt_ids[:, 0]

        for t in range(1, tgt_len):
            pred, dec_h, dec_c, _ = self.decoder.forward_step(
                dec_input, dec_h, dec_c, enc_out, pos_trimmed, src_mask
            )
            outputs.append(pred)
            use_tf    = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = tgt_ids[:, t] if use_tf else pred.argmax(dim=-1)

        return torch.stack(outputs, dim=1)

    def generate(self, src_ids, src_pos, src_lengths,
                 max_len=config.MAX_SUMMARY_LEN, start_idx=2, end_idx=3,
                 muril_input_ids=None, muril_attn_mask=None,
                 no_repeat_ngram=3):
        self.eval()
        with torch.no_grad():
            enc_out, enc_h, enc_c = self._encode(
                src_ids, src_pos, src_lengths, muril_input_ids, muril_attn_mask
            )
            dec_h = torch.tanh(self.enc2dec_h(enc_h))
            dec_c = torch.tanh(self.enc2dec_c(enc_c))
            enc_len = enc_out.size(1)
            if self.use_muril and muril_attn_mask is not None:
                muril_mask_trimmed = muril_attn_mask[:, :enc_len]
                src_mask = (muril_mask_trimmed == 0)
            else:
                src_mask = self._make_src_mask(src_ids, enc_len)
            pos_trimmed = src_pos[:, :enc_len, :]

            B         = src_ids.size(0)
            dec_input = torch.full((B,), start_idx,
                                   dtype=torch.long, device=src_ids.device)
            generated = []
            done      = torch.zeros(B, dtype=torch.bool, device=src_ids.device)

            for step in range(max_len):
                pred, dec_h, dec_c, _ = self.decoder.forward_step(
                    dec_input, dec_h, dec_c, enc_out, pos_trimmed, src_mask
                )
                pred[:, 0] = float('-inf')   # block PAD
                pred[:, 1] = float('-inf')   # block UNK

                if no_repeat_ngram > 0 and step >= 1:
                    for prev in generated[-no_repeat_ngram:]:
                        for b in range(B):
                            pred[b, prev[b].item()] = float('-inf')

                next_tok = pred.argmax(dim=-1)
                generated.append(next_tok)
                done |= (next_tok == end_idx)
                if done.all():
                    break
                dec_input = next_tok

        return torch.stack(generated, dim=1)


# ── Factory ────────────────────────────────────────────────────────────────────

MURIL_VARIANTS = {
    # Ablation study — isolates each contribution
    'word2vec_bilstm':     {'use_muril': False, 'use_bilstm': True,  'use_pos_gate': False},
    'muril_lstm':          {'use_muril': True,  'use_bilstm': False, 'use_pos_gate': False},
    'muril_bilstm':        {'use_muril': True,  'use_bilstm': True,  'use_pos_gate': False},
    'muril_bilstm_pos':    {'use_muril': True,  'use_bilstm': True,  'use_pos_gate': True},  # proposed
}


def build_muril_model(variant='muril_bilstm_pos', vocab_size=None,
                      embedding_matrix=None, freeze_muril=True):
    if variant not in MURIL_VARIANTS:
        raise ValueError(f"Choose from: {list(MURIL_VARIANTS)}")
    flags = MURIL_VARIANTS[variant]
    model = MuRILBiLSTMModel(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        freeze_muril=freeze_muril,
        **flags,
    ).to(config.DEVICE)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{variant}' | trainable params={n:,} | device={config.DEVICE}")
    return model