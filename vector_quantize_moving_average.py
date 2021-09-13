import torch
import torch.nn as nn
import torch.nn.functional as F


def moving_average(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


class MovingAverageVectorQuantizer(nn.Module):
    """
    Reference:
        https://github.com/deepmind/sonnet
    """

    def __init__(self, n_e, e_dim, decay=0.99, beta=1.0, eps=1e-5):
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.decay = decay
        self.beta = beta
        self.eps = eps

        embedding = torch.randn(n_e, e_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('embedding_avg', embedding.clone())
        self.register_buffer('cluster_size', torch.zeros(n_e))

    def get_codebook(self):
        return self.embedding

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = F.embedding(indices, self.embedding)
        if shape is not None:
            z_q = z_q.view(shape)
            # shape specifying (batch, height, width, channel)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)


        # distances from z to embeddings e (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(z_flattened, self.embedding.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, self.embedding).view(z.shape)

        # update codebook embedding by moving average
        if self.training:
            embedding_onehot = F.one_hot(min_encoding_indices, self.n_e).type(z_flattened.dtype)
            embedding_sum = embedding_onehot.t() @ z_flattened
            # TODO: all-reduce embedding_onehot and embedding_sum across gpus
            moving_average(self.cluster_size, embedding_onehot.sum(0), self.decay)
            moving_average(self.embedding_avg, embedding_sum, self.decay)
            n = self.cluster_size.sum()
            cluster_size = laplace_smoothing(self.cluster_size, self.n_e, self.eps) * n
            embedding_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embedding_normalized)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
               torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        min_encoding_indices = min_encoding_indices.view(z.shape[:-1])

        return z_q, loss, min_encoding_indices


