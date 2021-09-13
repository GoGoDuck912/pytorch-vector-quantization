import torch
import torch.nn as nn

from vector_quantize import VectorQuantizer


class ProductVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, num_quantizers):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.num_quantizers = num_quantizers
        assert self.e_dim % self.num_quantizers == 0
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim//num_quantizers) for _ in range(num_quantizers)])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, z):
        all_z_q = []
        all_losses = []
        all_min_encoding_indices = []

        z_chunk = torch.chunk(z, self.num_quantizers, dim=1)
        for idx, quantizer in enumerate(self.vq_layers):
            z_q, loss, min_encoding_indices = quantizer(z_chunk[idx])

            all_z_q.append(z_q)
            all_losses.append(loss)
            all_min_encoding_indices.append(min_encoding_indices)

        all_z_q = torch.cat(all_z_q, dim=1)
        mean_losses = torch.stack(all_losses).mean()
        all_min_encoding_indices = torch.stack(all_min_encoding_indices, dim=1)

        return all_z_q, mean_losses, all_min_encoding_indices
