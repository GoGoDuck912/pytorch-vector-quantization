import torch
import torch.nn as nn

from vector_quantize import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e, e_dim, num_quantizers):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.num_quantizers = num_quantizers
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim) for _ in range(num_quantizers)])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, z):
        all_losses = []
        all_min_encoding_indices = []

        z_q = 0
        residual = z
        for quantizer in self.vq_layers:
            z_res, loss, min_encoding_indices = quantizer(residual)
            residual = residual - z_res
            z_q = z_q + z_res

            all_losses.append(loss)
            all_min_encoding_indices.append(min_encoding_indices)

        mean_losses = torch.stack(all_losses).mean()
        all_min_encoding_indices = torch.stack(all_min_encoding_indices, dim=1)

        return z_q, mean_losses, all_min_encoding_indices
