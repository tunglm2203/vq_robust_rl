from typing import Tuple, Union, cast, Optional, Sequence, Any

import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizerEMA_unshared_codebook(nn.Module):
    def __init__(self, number_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99,
                 n_codebooks=17, update_codebook=True, epsilon=1e-5):
        super(VectorQuantizerEMA_unshared_codebook, self).__init__()

        self.number_embeddings = number_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.decay = decay
        self.epsilon = epsilon

        self.codebooks = nn.Parameter(torch.Tensor(n_codebooks, number_embeddings))
        self.codebooks.data.normal_()
        self.ema_codebooks = nn.Parameter(torch.Tensor(n_codebooks, number_embeddings))
        self.ema_codebooks.data.normal_()
        self.codebooks.requires_grad = False
        self.ema_codebooks.requires_grad = False

        self.register_buffer('ema_cluster_size', torch.zeros(n_codebooks, number_embeddings))

        self._update_codebook = update_codebook

    def enable_update_codebook(self):
        self._update_codebook = True

    def disable_update_codebook(self):
        self._update_codebook = False

    def forward(self, z):
        B, D = z.shape              # BxD

        z = z.unsqueeze(2)          # BxDx1

        Z_mat = z.repeat(1, 1, self.number_embeddings)          # BxDx1 -> BxDxK
        E_mat = self.codebooks.unsqueeze(0).repeat(B, 1, 1)     # DxK   -> BxDxK

        # distances from z to embeddings e_j
        distances = (Z_mat - E_mat) ** 2            # BxDxK

        # find closest encodings
        encoding_indices = torch.argmin(distances, dim=2, keepdim=True)             # BxDx1
        encodings = torch.zeros(B, D, self.number_embeddings, device=z.device)      # BxDxK
        encodings.scatter_(2, encoding_indices, 1)                         # one-hot: BxDxK

        # get quantized latent vectors
        quantized = torch.sum(encodings * self.codebooks, dim=2, keepdim=True)   # BxDx1

        # use EMA to update the embedding vectors
        if self.training and self._update_codebook:
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * torch.sum(encodings, dim=0)
            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size, dim=1, keepdim=True)
            self.ema_cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.number_embeddings * self.epsilon) * n

            dw = torch.sum(encodings * z, dim=0)
            self.ema_codebooks = nn.Parameter(self.decay * self.ema_codebooks + (1 - self.decay) * dw)

            self.codebooks = nn.Parameter(self.ema_codebooks / self.ema_cluster_size)

        q_latent_loss = F.mse_loss(quantized.detach(), z.detach())
        # compute loss for embedding
        # e_latent_loss = F.mse_loss(quantized.detach(), z)
        # loss = self.commitment_cost * e_latent_loss
        loss = q_latent_loss

        # Straight Through Estimator
        quantized = z + (quantized - z).detach()

        # perplexity
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.squeeze(dim=2)


class VectorQuantizer_unshared_codebook(nn.Module):
    def __init__(self, number_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99,
                 n_codebooks=17, update_codebook=True, epsilon=1e-5):
        super(VectorQuantizer_unshared_codebook, self).__init__()

        self.number_embeddings = number_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.decay = decay
        self.epsilon = epsilon

        self.codebooks = nn.Parameter(torch.Tensor(n_codebooks, number_embeddings))
        self.codebooks.data.normal_()
        self.ema_codebooks = nn.Parameter(torch.Tensor(n_codebooks, number_embeddings))
        self.ema_codebooks.data.normal_()

        self.register_buffer('ema_cluster_size', torch.zeros(n_codebooks, number_embeddings))

        self._update_codebook = update_codebook

    def enable_update_codebook(self):
        self._update_codebook = True

    def disable_update_codebook(self):
        self._update_codebook = False

    def forward(self, z):
        B, D = z.shape              # BxD

        z = z.unsqueeze(2)          # BxDx1

        Z_mat = z.repeat(1, 1, self.number_embeddings)          # BxDx1 -> BxDxK
        E_mat = self.codebooks.unsqueeze(0).repeat(B, 1, 1)     # DxK   -> BxDxK

        # distances from z to embeddings e_j
        distances = (Z_mat - E_mat) ** 2            # BxDxK

        # find closest encodings
        encoding_indices = torch.argmin(distances, dim=2, keepdim=True)             # BxDx1
        encodings = torch.zeros(B, D, self.number_embeddings, device=z.device)      # BxDxK
        encodings.scatter_(2, encoding_indices, 1)                         # one-hot: BxDxK

        # get quantized latent vectors
        quantized = torch.sum(encodings * self.codebooks, dim=2, keepdim=True)   # BxDx1

        if self.training and self._update_codebook:
            q_latent_loss = F.mse_loss(quantized, z.detach())
        else:
            q_latent_loss = F.mse_loss(quantized.detach(), z.detach())

        # compute loss for embedding
        # e_latent_loss = F.mse_loss(quantized.detach(), z)
        # loss = self.commitment_cost * e_latent_loss
        loss = q_latent_loss

        # Straight Through Estimator
        quantized = z + (quantized - z).detach()

        # perplexity
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.squeeze(dim=2)

