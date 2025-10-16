""" This file contains nets used for Diffusion Policy. """
import math
from typing import Union, Optional

import torch
import torch.nn as nn

import robomimic.models.base_nets as BaseNets

from robomimic.models.transformers import SpatioTemporalEncoder, DiT, EBT
from robomimic.models.sincos_pos_emb import (
    get_1d_sincos_pos_embed_from_grid
)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ObsTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        attn_dropout,
        proj_dropout,
        num_layers,
        num_heads,
        n_obs_steps
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        # input embedding stem
        self.input_emb = nn.Linear(input_dim, embed_dim)
        self.input_pos_emb = nn.Parameter(torch.zeros(1, n_obs_steps, embed_dim))

        self.encoder = SpatioTemporalEncoder(
            dim=embed_dim,
            depth=num_layers,
            heads=num_heads,
            output_dim=embed_dim,
            attn_drop=attn_dropout,
            proj_drop=proj_dropout,
            n_obs_steps=n_obs_steps
        )
        mask = torch.tril(torch.ones(n_obs_steps, n_obs_steps)).view(
            1, 1, n_obs_steps, n_obs_steps
        )
        self.register_buffer("mask", mask)

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        x = self.input_emb(x) + self.input_pos_emb
        x = self.encoder(x, self.mask)

        return x


class EBTTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        embed_dim,
        output_dim,
        attn_dropout,
        proj_dropout,
        num_layers,
        num_heads,
        n_obs_steps,
        horizon
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        # input embedding stem
        self.pos_emb = nn.Parameter(
            torch.zeros(1, horizon, embed_dim),
            requires_grad=False
        )
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.input_emb = BaseNets.MLP(
            input_dim=input_dim,
            output_dim=embed_dim,
            layer_dims=(embed_dim,),
            activation=approx_gelu,
            dropouts=(proj_dropout,),
            normalization=True,
            output_activation=approx_gelu
        )
        self.decoder = EBT(
            dim=embed_dim,
            depth=num_layers,
            heads=num_heads,
            output_dim=output_dim,
            attn_drop=attn_dropout,
            proj_drop=proj_dropout
        )
        self.dim = embed_dim

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )
        self._init_pos_emb(self.pos_emb, horizon)

    def forward(
        self,
        sample: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ):
        """
        x: (B,T,input_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,C,T)

        # 1. time
        x = self.input_emb(sample) + self.pos_emb
        c = cond
        x = self.decoder(x=x, c=c, mask=mask, memory_mask=memory_mask)

        # (B,T,C)
        return x

    def _init_pos_emb(
        self,
        module: nn.Parameter,
        timesteps: int,
    ) -> None:
        """
        Initialise all the postional embeddings to be sin-cos positional
        embeddings.

        Args:
            module: The positional embedding that will be initialised
            timesteps: The number of timesteps to consider in the position
        """
        pos = torch.arange(timesteps)
        pos_emb = get_1d_sincos_pos_embed_from_grid(self.dim, pos)

        pos_emb = pos_emb.unsqueeze(0)
        module.data.copy_(pos_emb)
