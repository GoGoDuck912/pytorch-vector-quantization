import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch import einsum


class GumbelQuantize(nn.Module):
    """
    Reference:
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(self, hidden_channel, n_e, e_dim, kl_weight=1.0,
                 temp_init=1.0, straight_through=True,):
        super().__init__()

        self.e_dim = e_dim
        self.n_e = n_e

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(hidden_channel, n_e, kernel_size=1)
        self.embedding = nn.Embedding(n_e, e_dim)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            # shape specifying (batch, height, width, channel)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def forward(self, z, temp=None):
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        min_encoding_indices = soft_one_hot.argmax(dim=1)

        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embedding.weight)

        # kl divergence loss w.r.t uniform distributions
        code_prob = F.softmax(logits, dim=1)
        loss = self.kl_weight * torch.sum(code_prob * torch.log(code_prob * self.n_e + 1e-10), dim=1).mean()

        return z_q, loss, min_encoding_indices