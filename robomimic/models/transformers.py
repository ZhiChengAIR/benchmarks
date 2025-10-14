"""
Implementation of transformers, mostly based on Andrej's minGPT model.
See https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
for more details.
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import RmsNorm, Mlp
from .attention import Attention, CrossAttention

from robomimic.models.base_nets import Module
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

class GEGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    Implementation: https://github.com/pfnet-research/deep-table/blob/237c8be8a405349ce6ab78075234c60d9bfe60b7/deep_table/nn/layers/activation.py
    """

    def geglu(self, x):
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x):
        return self.geglu(x)


class PositionalEncoding(nn.Module):
    """
    Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(self, embed_dim):
        """
        Standard sinusoidal positional encoding scheme in transformers.

        Positional encoding of the k'th position in the sequence is given by:
            p(k, 2i) = sin(k/n^(i/d))
            p(k, 2i+1) = sin(k/n^(i/d))

        n: set to 10K in original Transformer paper
        d: the embedding dimension
        i: positions along the projected embedding space (ranges from 0 to d/2)

        Args:
            embed_dim: The number of dimensions to project the timesteps into.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Input timestep of shape BxT
        """
        position = x

        # computing 1/n^(i/d) in log space and then exponentiating and fixing the shape
        div_term = (
            torch.exp(
                torch.arange(0, self.embed_dim, 2, device=x.device)
                * (-math.log(10000.0) / self.embed_dim)
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(x.shape[0], x.shape[1], 1)
        )
        pe = torch.zeros((x.shape[0], x.shape[1], self.embed_dim), device=x.device)
        pe[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
        return pe.detach()


class CausalSelfAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attn_dropout=0.1,
        output_dropout=0.1,
    ):
        """
        Multi-head masked self-attention layer + projection (MLP layer).

        For normal self-attention (@num_heads = 1), every single input in the sequence is
        mapped to a key, query, and value embedding of size @embed_dim. For each input,
        its query vector is compared (using dot-product) with all other key vectors in the
        sequence, and softmax normalized to compute an attention over all members of the
        sequence. This is used to take a linear combination of corresponding value embeddings.

        The @num_heads argument is for multi-head attention, where the self-attention operation above
        is performed in parallel over equal size partitions of the @embed_dim, allowing for different
        portions of the embedding dimension to model different kinds of attention. The attention
        output for each head is concatenated together.

        Finally, we use a causal mask here to ensure that each output only depends on inputs that come
        before it.

        Args:
            embed_dim (int): dimension of embeddings to use for keys, queries, and values
                used in self-attention

            num_heads (int): number of attention heads - must divide @embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            context_length (int): expected length of input sequences

            attn_dropout (float): dropout probability for attention outputs

            output_dropout (float): dropout probability for final outputs
        """
        super(CausalSelfAttention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "num_heads: {} does not divide embed_dim: {} exactly".format(num_heads, embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.nets = nn.ModuleDict()

        # projection layers for key, query, value, across all attention heads
        self.nets["qkv"] = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        # dropout layers
        self.nets["attn_dropout"] = nn.Dropout(self.attn_dropout)
        self.nets["output_dropout"] = nn.Dropout(self.output_dropout)

        # output layer
        self.nets["output"] = nn.Linear(self.embed_dim, self.embed_dim)

        # causal mask (ensures attention is only over previous inputs) - just a lower triangular matrix of 1s
        mask = torch.tril(torch.ones(context_length, context_length)).view(
            1, 1, context_length, context_length
        )
        self.register_buffer("mask", mask)

    def forward(self, x):
        """
        Forward pass through Self-Attention block.
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
        """

        # enforce shape consistency
        assert len(x.shape) == 3
        B, T, D = x.shape
        assert (
            T <= self.context_length
        ), "self-attention module can only handle sequences up to {} in length but got length {}".format(
            self.context_length, T
        )
        assert D == self.embed_dim
        NH = self.num_heads  # number of attention heads
        DH = D // NH  # embed dimension for each attention head

        # compute key, query, and value vectors for each member of sequence, and split across attention heads
        qkv = self.nets["qkv"](x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        k = k.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]
        q = q.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]
        v = v.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]

        # causal self-attention mechanism

        # batched matrix multiplication between queries and keys to get all pair-wise dot-products.
        # We broadcast across batch and attention heads and get pair-wise dot-products between all pairs of timesteps
        # [B, NH, T, DH] x [B, NH, DH, T] -> [B, NH, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # use mask to replace entries in dot products with negative inf to ensure they don't contribute to softmax,
        # then take softmax over last dimension to end up with attention score for each member of sequence.
        # Note the use of [:T, :T] -  this makes it so we can handle sequences less than @self.context_length in length.
        att = att.masked_fill(self.mask[..., :T, :T] == 0, float("-inf"))
        att = F.softmax(
            att, dim=-1
        )  # shape [B, NH, T, T], last dimension has score over all T for each sequence member

        # dropout on attention
        att = self.nets["attn_dropout"](att)

        # take weighted sum of value vectors over whole sequence according to attention, with batched matrix multiplication
        # [B, NH, T, T] x [B, NH, T, DH] -> [B, NH, T, DH]
        y = att @ v
        # reshape [B, NH, T, DH] -> [B, T, NH, DH] -> [B, T, NH * DH] = [B, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, D)

        # pass through output layer + dropout
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)
        return y

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # this module doesn't modify the size of the input, it goes from (B, T, D) -> (B, T, D)
        return list(input_shape)


class SelfAttentionBlock(Module):
    """
    A single Transformer Block, that can be chained together repeatedly.
    It consists of a @CausalSelfAttention module and a small MLP, along with
    layer normalization and residual connections on each input.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attn_dropout=0.1,
        output_dropout=0.1,
        activation=nn.GELU(),
    ):
        """
        Args:
            embed_dim (int): dimension of embeddings to use for keys, queries, and values
                used in self-attention

            num_heads (int): number of attention heads - must divide @embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            context_length (int): expected length of input sequences

            attn_dropout (float): dropout probability for attention outputs

            output_dropout (float): dropout probability for final outputs

            activation (str): string denoting the activation function to use in each transformer block
        """
        super(SelfAttentionBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.nets = nn.ModuleDict()

        # self-attention block
        self.nets["attention"] = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            attn_dropout=attn_dropout,
            output_dropout=output_dropout,
        )

        if type(activation) == GEGLU:
            mult = 2
        else:
            mult = 1

        # small 2-layer MLP
        self.nets["mlp"] = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim * mult),
            activation,
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(output_dropout)
        )

        # layer normalization for inputs to self-attention module and MLP
        self.nets["ln1"] = nn.LayerNorm(embed_dim)
        self.nets["ln2"] = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass - chain self-attention + MLP blocks, with residual connections and layer norms.
        """
        x = x + self.nets["attention"](self.nets["ln1"](x))
        x = x + self.nets["mlp"](self.nets["ln2"](x))
        return x

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # this module doesn't modify the size of the input, it goes from (B, T, D) -> (B, T, D)
        return list(input_shape)


class GPT_Backbone(Module):
    """the GPT model, with a context size of block_size"""

    def __init__(
        self,
        embed_dim,
        context_length,
        attn_dropout=0.1,
        block_output_dropout=0.1,
        num_layers=6,
        num_heads=8,
        activation="gelu",
    ):
        """
        Args:
            embed_dim (int): dimension of embeddings to use for keys, queries, and values
                used in self-attention

            context_length (int): expected length of input sequences

            attn_dropout (float): dropout probability for attention outputs for each transformer block

            block_output_dropout (float): dropout probability for final outputs for each transformer block

            num_layers (int): number of transformer blocks to stack

            num_heads (int): number of attention heads - must divide @embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            activation (str): string denoting the activation function to use in each transformer block

        """
        super(GPT_Backbone, self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.block_output_dropout = block_output_dropout

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "geglu":
            self.activation = GEGLU()

        # create networks
        self._create_networks()

        # initialize weights
        self.apply(self._init_weights)

        print(
            "Created {} model with number of parameters: {}".format(
                self.__class__.__name__, sum(p.numel() for p in self.parameters())
            )
        )

    def _create_networks(self):
        """
        Helper function to create networks.
        """
        self.nets = nn.ModuleDict()

        # transformer - cascaded transformer blocks
        self.nets["transformer"] = nn.Sequential(
            *[
                SelfAttentionBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    context_length=self.context_length,
                    attn_dropout=self.attn_dropout,
                    output_dropout=self.block_output_dropout,
                    activation=self.activation,
                )
                for _ in range(self.num_layers)
            ]
        )

        # decoder head
        self.nets["output_ln"] = nn.LayerNorm(self.embed_dim)

    def _init_weights(self, module):
        """
        Weight initializer.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # this module takes inputs (B, T, @self.input_dim) and produces outputs (B, T, @self.output_dim)
        return input_shape[:-1] + [self.output_dim]

    def forward(self, inputs):
        assert inputs.shape[1:] == (self.context_length, self.embed_dim), inputs.shape
        x = self.nets["transformer"](inputs)
        transformer_output = self.nets["output_ln"](x)
        return transformer_output


class SpatioTemporalBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 proj_drop,
                 attn_drop,
                 t,
                 **block_kwargs):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **block_kwargs
        )
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.ffn = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            act_layer=approx_gelu,
            drop=proj_drop
        )

    def forward(self, x, mask):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = x + origin_x

        origin_x = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + origin_x

        return x


class SpatioTemporalEncoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        output_dim,
        attn_drop,
        proj_drop,
        n_obs_steps
    ) -> None:
        super().__init__()
        self.num_heads = heads
        self.t = n_obs_steps
        self.layers = nn.ModuleList([
            SpatioTemporalBlock(
                hidden_size=dim,
                num_heads=heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                t=n_obs_steps
            )
            for _ in range(depth)
        ])

        self._init_weights()

    def forward(self, x, mask):
        for block in self.layers:
            x = block(x, mask)

        return x

    def _init_weights(self) -> None:
        """
        Initialises all the ffn weights in the DiT encoder and separation
        special token with xavier uniform/normal distribution, and the
        positional embeddings to sin-cos
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiTFinalLayer(nn.Module):
    """
    The final layer of the diffusion model.
    """

    def __init__(self, hidden_size, out_channels, drop):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.linear = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            act_layer=approx_gelu,
            drop=drop,
            bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
        x = self.norm_final(x)
        x = modulate(x, shift, scale)
        x = self.linear(x)

        return x


class EBTFinalLayer(nn.Module):
    """
    The final layer of the diffusion model.
    """

    def __init__(self, hidden_size, out_channels, drop):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.linear = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            act_layer=approx_gelu,
            drop=drop,
            bias=True
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)

        return x


class DiTBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 proj_drop,
                 attn_drop,
                 **block_kwargs):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size, num_heads=num_heads,
            qkv_bias=True, qk_norm=True,
            norm_layer=RmsNorm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **block_kwargs)
        self.cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads,
            qkv_bias=True, qk_norm=True,
            norm_layer=RmsNorm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **block_kwargs)

        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.ffn = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            act_layer=approx_gelu,
            drop=proj_drop
        )
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, t, mask=None, memory_mask=None):
        adaln_features = self.adaLN_modulation(t).chunk(9, dim=-1)
        shift_msa, scale_msa, gate_msa = adaln_features[:3]
        shift_mca, scale_mca, gate_mca = adaln_features[3:6]
        shift_mlp, scale_mlp, gate_mlp = adaln_features[6:]
        origin_x = x
        x = self.norm1(x)
        x = modulate(x, shift_msa, scale_msa)
        x = self.attn(x, mask) * gate_msa
        x = x + origin_x

        origin_x = x
        x = self.norm2(x)
        x = modulate(x, shift_mca, scale_mca)
        x = self.cross_attn(x, c, memory_mask) * gate_mca
        x = x + origin_x

        origin_x = x
        x = self.norm3(x)
        x = modulate(x, shift_mlp, scale_mlp)
        x = self.ffn(x) * gate_mlp
        x = x + origin_x

        return x


class DiT(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        output_dim,
        attn_drop,
        proj_drop
    ) -> None:
        super().__init__()
        self.num_heads = heads
        self.layers = nn.ModuleList([
            DiTBlock(
                hidden_size=dim,
                num_heads=heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop
            )
            for _ in range(depth)
        ])
        self.final_layer = DiTFinalLayer(
            hidden_size=dim,
            out_channels=output_dim,
            drop=proj_drop
        )

        self._init_weights()

    def forward(self, x, c, t, mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, c, t, mask, memory_mask)
        x = self.final_layer(x, t)

        return x

    def _init_weights(self) -> None:
        """
        Initialises all the ffn weights in the DiT encoder and separation
        special token with xavier uniform/normal distribution, and the
        positional embeddings to sin-cos
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        self._init_adaln_zero_weights()

    def _init_adaln_zero_weights(self) -> None:
        # Zero-out adaLN modulation layers in DiT blocks:
        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)


class EBTBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 proj_drop,
                 attn_drop,
                 **block_kwargs):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size, num_heads=num_heads,
            qkv_bias=True, qk_norm=True,
            norm_layer=RmsNorm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **block_kwargs)
        self.cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads,
            qkv_bias=True, qk_norm=True,
            norm_layer=RmsNorm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **block_kwargs)

        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.ffn = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            act_layer=approx_gelu,
            drop=proj_drop
        )

        self.norm3 = RmsNorm(hidden_size, eps=1e-6)

    def forward(self, x, c, mask=None, memory_mask=None):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = x + origin_x

        origin_x = x
        x = self.norm2(x)
        x = self.cross_attn(x, c, memory_mask)
        x = x + origin_x

        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + origin_x

        return x


class EBT(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        output_dim,
        attn_drop,
        proj_drop
    ) -> None:
        super().__init__()
        self.num_heads = heads
        self.layers = nn.ModuleList([
            EBTBlock(
                hidden_size=dim,
                num_heads=heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop
            )
            for _ in range(depth)
        ])
        self.final_layer = EBTFinalLayer(
            hidden_size=dim,
            out_channels=output_dim,
            drop=proj_drop
        )

        self._init_weights()

    def forward(self, x, c, mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, c, mask, memory_mask)
        x = self.final_layer(x)

        return x

    def _init_weights(self) -> None:
        """
        Initialises all the ffn weights in the DiT encoder and separation
        special token with xavier uniform/normal distribution, and the
        positional embeddings to sin-cos
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
