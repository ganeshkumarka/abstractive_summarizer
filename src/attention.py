"""
src/attention.py
----------------
Implements Bahdanau (additive) attention — the exact mechanism described
in paper §3.4, equations (1), (2), (3).

Three components (paper §3.4):
  1. Alignment score  eij = a(s_{i-1}, h_j)         — Eq. (1)
  2. Attention weights at  = softmax(eij)            — Eq. (2)
  3. Context vector   h*_t = Σ at_i * h_i            — Eq. (3)

The alignment function a() is implemented as a small feed-forward network
(as described in the original Bahdanau et al. 2014 paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BahdanauAttention(nn.Module):
    """
    Bahdanau additive attention (paper §3.4).

    For each decoder step i, given:
      - decoder hidden state  s_{i-1}   : (batch, hidden_dim)
      - all encoder outputs   H         : (batch, src_len, hidden_dim)

    Computes:
      e_ij   = V_a · tanh(W_a · s_{i-1} + U_a · h_j)   ← alignment score
      a_t    = softmax(e_ij)                              ← attention weights
      c_t    = Σ_j a_tj * h_j                            ← context vector
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # W_a: project decoder hidden state
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # U_a: project encoder hidden states
        self.U_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # V_a: final scoring vector (reduces to scalar per position)
        self.V_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                src_mask: torch.Tensor = None):
        """
        Args:
            decoder_hidden  : (batch, hidden_dim)    — s_{i-1}
            encoder_outputs : (batch, src_len, hidden_dim)  — all h_j
            src_mask        : (batch, src_len) bool — True for PAD positions

        Returns:
            context_vector  : (batch, hidden_dim)   — weighted sum of encoder outputs
            attention_weights: (batch, src_len)     — softmax weights (useful for viz)
        """
        batch_size, src_len, _ = encoder_outputs.size()

        # Expand decoder hidden state to match src_len
        # (batch, hidden_dim) → (batch, 1, hidden_dim) → (batch, src_len, hidden_dim)
        dec_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)

        # Compute alignment scores  eij = V_a · tanh(W_a·s + U_a·h)
        # Each of shape (batch, src_len, hidden_dim)
        energy = torch.tanh(
            self.W_a(dec_hidden_expanded) + self.U_a(encoder_outputs)
        )
        # (batch, src_len, 1) → (batch, src_len)
        alignment_scores = self.V_a(energy).squeeze(-1)

        # Mask padding positions so they get ~0 weight
        if src_mask is not None:
            alignment_scores = alignment_scores.masked_fill(src_mask, float('-inf'))

        # Attention weights: softmax over alignment scores  (Eq. 2)
        attention_weights = F.softmax(alignment_scores, dim=-1)  # (batch, src_len)

        # Context vector: weighted sum of encoder outputs  (Eq. 3)
        # (batch, 1, src_len) × (batch, src_len, hidden_dim) → (batch, 1, hidden_dim)
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)  # (batch, hidden_dim)

        return context_vector, attention_weights