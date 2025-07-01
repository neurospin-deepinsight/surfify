##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
The Surface Vision Transformer.
"""

# Imports
import torch
from torch import nn
from ..utils import get_logger, debug_msg


# Global parameters
logger = get_logger()


class SiT(nn.Module):
    """ The SiT model: implements surface vision transformers.

    Notes
    -----
    Debuging messages can be displayed by changing the log level using
    ``setup_logging(level='debug')``.

    See Also
    --------
    mSiT

    References
    ----------
    Dahan, Simon et al., Surface Vision Transformers: Attention-Based
    Modelling applied to Cortical Analysis, MIDL, 2022.
    """
    def __init__(self, dim, depth, heads, mlp_dim, n_patches,
                 n_channels, n_vertices, n_classes=1, pool="cls", dim_head=64,
                 dropout=0., emb_dropout=0.):
        """ Init SiT.

        Parameters
        ----------
        dim: int
            the sequence of N flattened patches of size
            n_channels x n_verticesis is first projected onto a sequence of
            dimension dim with a trainable linear embedding layer.
        depth: int
            the number of transformer blocks, each composed of
            a multi-head self-attention layer (MSA), implementing the
            self-attention mechanism across the sequence, and a feed-forward
            network (FFN), which expands then reduces the sequence
            dimension.
        heads: int
            number of attention heads.
        mlp_dim: int
            MLP hidden dim to embedding dim.
        n_patches: int
            the number of patches.
        n_channels: int
            the number of channels.
        n_vertices: int
            the number of vertices.
        n_classes: int, default 1
            the number of classes to predict: if <=0 return the latent space.
        pool: str, default 'cls'
            the polling strategy: 'cls' (cls token) or 'mean' (mean pooling).
        dim_head: int, default 64
            the output dimension of the layer.
        dropout: float, default 0.
            transformer dropout rate.
        emb_dropout: float, default 0.
            embeding dropout rate.
        """
        super().__init__()
        assert pool in {"cls", "mean"}, (
            "Pool type must be either cls (cls token) or mean (mean pooling).")
        self.patch_embedding = nn.Linear(n_channels * n_vertices, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout)
        self.pool = pool
        self.mlp_head = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, n_classes)) if n_classes > 0
            else nn.Identity()
        )

    def forward(self, x):
        """ Forward method.

        Parameters
        ----------
        x: Tensor (n_samples, n_channels, n_patches, n_vertices)
            the input data.

        Returns
        -------
        x: Tensor (n_samples, n_channels, n_patches, n_vertices)
            the output data.        
        """
        logger.debug("Rearange...")
        logger.debug(debug_msg("input", x))
        x = torch.swapdims(x, 1, 2)
        x = torch.flatten(x, start_dim=2)
        logger.debug(debug_msg("output", x))

        logger.debug("Linear embeding...")
        logger.debug(debug_msg("input", x))
        x = self.patch_embedding(x)
        logger.debug(debug_msg("output", x))
        n_samples, n_patches, _ = x.shape

        logger.debug("Positional embeding...")
        logger.debug(debug_msg("input", x))
        cls_tokens = self.cls_token.repeat(n_samples, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n_patches + 1)]
        x = self.dropout(x)
        logger.debug(debug_msg("output", x))

        logger.debug("L transformer blocks...")
        logger.debug(debug_msg("input", x))
        x = self.transformer(x)
        logger.debug(debug_msg("output", x))

        logger.debug("Pooling...")
        logger.debug(debug_msg("input", x))
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        logger.debug(debug_msg("output", x))

        logger.debug("Phenotype prediction...")
        logger.debug(debug_msg("input", x))
        x = self.mlp_head(x)
        logger.debug(debug_msg("output", x))

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (torch.swapdims(
            torch.reshape(t, (t.size(dim=0), t.size(dim=1), self.heads, -1)),
            1, 2) for t in qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = torch.swapdims(out, 1, 2)
        out = torch.flatten(out, start_dim=2)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
